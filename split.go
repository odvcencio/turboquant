package turboquant

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"sort"
	"sync"
)

// splitOutlierSeedMix decorrelates the outlier-side sub-quantizer's
// randomness from the regular side's. Never change this constant: split
// quantizers are reconstructed from (dim, spec, seed) alone.
const splitOutlierSeedMix = 0x517cc1b727220a95

// SplitSpec partitions vector channels into an outlier set quantized at
// OutlierBits and a regular remainder quantized at RegularBits. Two
// independent TurboQuant instances cover the two sets, which realizes the
// paper's section 4.3 mixed-precision scheme and its fractional effective
// bit widths (for example 32 channels at 4 bits plus 96 channels at 2 bits
// over dimension 128 is the 2.5-bit KV configuration).
//
// An empty outlier set selects a single uniform instance at RegularBits;
// OutlierBits must then be 0.
type SplitSpec struct {
	Outliers    []int // sorted, unique channel indices
	OutlierBits int
	RegularBits int
}

// EffectiveBits returns the average code bits per channel for dimension dim.
// Norm and sign side-band storage is not included; it is the same per-vector
// overhead the uniform quantizers carry.
func (s SplitSpec) EffectiveBits(dim int) float64 {
	n := len(s.Outliers)
	return (float64(n)*float64(s.OutlierBits) + float64(dim-n)*float64(s.RegularBits)) / float64(dim)
}

// validateSplitSpec checks the channel partition. Bit-width validation stays
// with the caller because the IP and MSE ranges differ.
func validateSplitSpec(dim int, spec SplitSpec) error {
	n := len(spec.Outliers)
	if n == 0 {
		if spec.OutlierBits != 0 {
			return fmt.Errorf("turboquant: split spec has no outlier channels; OutlierBits must be 0, got %d", spec.OutlierBits)
		}
		return nil
	}
	if n < 2 || dim-n < 2 {
		return fmt.Errorf("turboquant: split spec needs at least 2 outlier and 2 regular channels, got %d outlier / %d regular", n, dim-n)
	}
	prev := -1
	for _, ch := range spec.Outliers {
		if ch < 0 || ch >= dim {
			return fmt.Errorf("turboquant: split outlier channel %d out of range [0,%d)", ch, dim)
		}
		if ch <= prev {
			return fmt.Errorf("turboquant: split outlier channels must be sorted and unique")
		}
		prev = ch
	}
	return nil
}

func cloneSplitSpec(spec SplitSpec) SplitSpec {
	return SplitSpec{
		Outliers:    append([]int(nil), spec.Outliers...),
		OutlierBits: spec.OutlierBits,
		RegularBits: spec.RegularBits,
	}
}

func splitComplement(dim int, outliers []int) []int {
	member := make([]bool, dim)
	for _, ch := range outliers {
		member[ch] = true
	}
	out := make([]int, 0, dim-len(outliers))
	for ch := range dim {
		if !member[ch] {
			out = append(out, ch)
		}
	}
	return out
}

func gatherInto(dst, src []float32, idx []int) {
	for i, ch := range idx {
		dst[i] = src[ch]
	}
}

func scatterInto(dst, src []float32, idx []int) {
	for i, ch := range idx {
		dst[ch] = src[i]
	}
}

// SelectOutlierChannels ranks channels by mean absolute magnitude over the
// calibration samples and returns the count highest-magnitude channel
// indices in ascending order. This is the light-touch outlier detection the
// paper's KV experiments rely on; the quantizers themselves stay
// data-oblivious.
func SelectOutlierChannels(samples [][]float32, count int) []int {
	if len(samples) == 0 || count <= 0 {
		return nil
	}
	dim := len(samples[0])
	if count > dim {
		count = dim
	}
	scores := make([]float64, dim)
	for _, s := range samples {
		if len(s) != dim {
			panic(fmt.Sprintf("turboquant.SelectOutlierChannels: sample length %d, want %d", len(s), dim))
		}
		for i, v := range s {
			if v < 0 {
				v = -v
			}
			scores[i] += float64(v)
		}
	}
	order := make([]int, dim)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return scores[order[a]] > scores[order[b]]
	})
	selected := append([]int(nil), order[:count]...)
	sort.Ints(selected)
	return selected
}

type splitScratch struct {
	out []float32
	reg []float32
}

// SplitIPQuantizer runs two independent IPQuantizer instances over disjoint
// channel subsets. The composed inner-product estimator is the sum of the
// two sub-estimates, so it stays unbiased, and the effective code rate is
// SplitSpec.EffectiveBits.
type SplitIPQuantizer struct {
	dim    int
	seed   int64
	spec   SplitSpec
	outIdx []int
	regIdx []int
	out    *IPQuantizer // nil when the spec has no outlier channels
	reg    *IPQuantizer
	pool   sync.Pool
}

// SplitIPQuantized holds the two sub-quantizations. Out is the zero value
// when the spec has no outlier channels.
type SplitIPQuantized struct {
	Out IPQuantized
	Reg IPQuantized
}

// SplitPreparedQuery holds per-side prepared query state.
type SplitPreparedQuery struct {
	Out PreparedQuery
	Reg PreparedQuery
}

// NewSplitIP creates a split inner-product quantizer with a random seed.
func NewSplitIP(dim int, spec SplitSpec) *SplitIPQuantizer {
	var seedBytes [8]byte
	_, _ = rand.Read(seedBytes[:])
	return NewSplitIPWithSeed(dim, spec, int64(binary.LittleEndian.Uint64(seedBytes[:])))
}

// NewSplitIPWithSeed creates a deterministic split inner-product quantizer.
// Two quantizers with identical (dim, spec, seed) produce identical output.
func NewSplitIPWithSeed(dim int, spec SplitSpec, seed int64) *SplitIPQuantizer {
	panicOnInvalid("turboquant.NewSplitIP", validateDim(dim))
	panicOnInvalid("turboquant.NewSplitIP", validateSplitSpec(dim, spec))
	panicOnInvalid("turboquant.NewSplitIP", validateIPBitWidth(spec.RegularBits))
	stored := cloneSplitSpec(spec)
	q := &SplitIPQuantizer{
		dim:    dim,
		seed:   seed,
		spec:   stored,
		outIdx: stored.Outliers,
		regIdx: splitComplement(dim, stored.Outliers),
	}
	if len(q.outIdx) > 0 {
		panicOnInvalid("turboquant.NewSplitIP", validateIPBitWidth(spec.OutlierBits))
		q.out = NewIPHadamardWithSeed(len(q.outIdx), spec.OutlierBits, seed^splitOutlierSeedMix)
	}
	q.reg = NewIPHadamardWithSeed(len(q.regIdx), spec.RegularBits, seed)
	outLen, regLen := len(q.outIdx), len(q.regIdx)
	q.pool.New = func() any {
		return &splitScratch{out: make([]float32, outLen), reg: make([]float32, regLen)}
	}
	return q
}

// Dim returns the full vector dimension.
func (q *SplitIPQuantizer) Dim() int { return q.dim }

// Seed returns the seed used to construct this quantizer.
func (q *SplitIPQuantizer) Seed() int64 { return q.seed }

// Spec returns a copy of the channel split specification.
func (q *SplitIPQuantizer) Spec() SplitSpec { return cloneSplitSpec(q.spec) }

// EffectiveBits returns the average code bits per channel.
func (q *SplitIPQuantizer) EffectiveBits() float64 { return q.spec.EffectiveBits(q.dim) }

// OutlierQuantizer returns the outlier-side sub-quantizer, or nil when the
// spec has no outlier channels.
func (q *SplitIPQuantizer) OutlierQuantizer() *IPQuantizer { return q.out }

// RegularQuantizer returns the regular-side sub-quantizer.
func (q *SplitIPQuantizer) RegularQuantizer() *IPQuantizer { return q.reg }

// AllocQuantized allocates storage shaped for this quantizer.
func (q *SplitIPQuantizer) AllocQuantized() SplitIPQuantized {
	var qx SplitIPQuantized
	if q.out != nil {
		qx.Out = AllocIPQuantized(len(q.outIdx), q.spec.OutlierBits)
	}
	qx.Reg = AllocIPQuantized(len(q.regIdx), q.spec.RegularBits)
	return qx
}

// Quantize compresses vec with both sub-quantizers.
func (q *SplitIPQuantizer) Quantize(vec []float32) SplitIPQuantized {
	qx := q.AllocQuantized()
	q.QuantizeTo(&qx, vec)
	return qx
}

// QuantizeTo compresses vec into caller-owned storage.
func (q *SplitIPQuantizer) QuantizeTo(dst *SplitIPQuantized, vec []float32) {
	if dst == nil {
		panic("(*SplitIPQuantizer).QuantizeTo: turboquant: nil destination")
	}
	panicOnInvalid("(*SplitIPQuantizer).QuantizeTo", ValidateVector(q.dim, vec))
	buf := q.pool.Get().(*splitScratch)
	defer q.pool.Put(buf)
	if q.out != nil {
		gatherInto(buf.out, vec, q.outIdx)
		q.out.QuantizeTo(&dst.Out, buf.out)
	}
	gatherInto(buf.reg, vec, q.regIdx)
	q.reg.QuantizeTo(&dst.Reg, buf.reg)
}

// InnerProduct estimates <x, y> as the sum of the two unbiased sub-estimates.
func (q *SplitIPQuantizer) InnerProduct(qx SplitIPQuantized, y []float32) float32 {
	panicOnInvalid("(*SplitIPQuantizer).InnerProduct", ValidateVector(q.dim, y))
	buf := q.pool.Get().(*splitScratch)
	defer q.pool.Put(buf)
	gatherInto(buf.reg, y, q.regIdx)
	est := q.reg.InnerProduct(qx.Reg, buf.reg)
	if q.out != nil {
		gatherInto(buf.out, y, q.outIdx)
		est += q.out.InnerProduct(qx.Out, buf.out)
	}
	return est
}

// PrepareQuery precomputes per-side query state for repeated lookups.
func (q *SplitIPQuantizer) PrepareQuery(y []float32) SplitPreparedQuery {
	panicOnInvalid("(*SplitIPQuantizer).PrepareQuery", ValidateVector(q.dim, y))
	buf := q.pool.Get().(*splitScratch)
	defer q.pool.Put(buf)
	var spq SplitPreparedQuery
	if q.out != nil {
		gatherInto(buf.out, y, q.outIdx)
		spq.Out = q.out.PrepareQuery(buf.out)
	}
	gatherInto(buf.reg, y, q.regIdx)
	spq.Reg = q.reg.PrepareQuery(buf.reg)
	return spq
}

// validatePreparedQuery checks that spq's per-side prepared queries match this
// quantizer's sub-quantizer dimensions.
func (q *SplitIPQuantizer) validatePreparedQuery(spq SplitPreparedQuery) error {
	if err := ValidatePreparedQuery(len(q.regIdx), spq.Reg); err != nil {
		return fmt.Errorf("turboquant: split regular prepared query: %w", err)
	}
	if q.out != nil {
		if err := ValidatePreparedQuery(len(q.outIdx), spq.Out); err != nil {
			return fmt.Errorf("turboquant: split outlier prepared query: %w", err)
		}
	}
	return nil
}

// InnerProductPrepared estimates <x, y> using per-side prepared query state.
func (q *SplitIPQuantizer) InnerProductPrepared(qx SplitIPQuantized, spq SplitPreparedQuery) float32 {
	est := q.reg.InnerProductPrepared(qx.Reg, spq.Reg)
	if q.out != nil {
		est += q.out.InnerProductPrepared(qx.Out, spq.Out)
	}
	return est
}

// Dequantize reconstructs an approximate vector at the original input scale
// by scattering the per-side reconstructions back to their channels.
func (q *SplitIPQuantizer) Dequantize(qx SplitIPQuantized) []float32 {
	recon := make([]float32, q.dim)
	scatterInto(recon, q.reg.Dequantize(qx.Reg), q.regIdx)
	if q.out != nil {
		scatterInto(recon, q.out.Dequantize(qx.Out), q.outIdx)
	}
	return recon
}

// SplitQuantizer runs two independent MSE-optimal Quantizer instances over
// disjoint channel subsets (the value-side counterpart of SplitIPQuantizer).
type SplitQuantizer struct {
	dim    int
	seed   int64
	spec   SplitSpec
	outIdx []int
	regIdx []int
	out    *Quantizer // nil when the spec has no outlier channels
	reg    *Quantizer
	pool   sync.Pool
}

// SplitQuantized holds the packed codes and per-side norms.
type SplitQuantized struct {
	Out     []byte
	OutNorm float32
	Reg     []byte
	RegNorm float32
}

// NewSplitMSE creates a split MSE quantizer with a random seed.
func NewSplitMSE(dim int, spec SplitSpec) *SplitQuantizer {
	var seedBytes [8]byte
	_, _ = rand.Read(seedBytes[:])
	return NewSplitMSEWithSeed(dim, spec, int64(binary.LittleEndian.Uint64(seedBytes[:])))
}

// NewSplitMSEWithSeed creates a deterministic split MSE quantizer.
func NewSplitMSEWithSeed(dim int, spec SplitSpec, seed int64) *SplitQuantizer {
	panicOnInvalid("turboquant.NewSplitMSE", validateDim(dim))
	panicOnInvalid("turboquant.NewSplitMSE", validateSplitSpec(dim, spec))
	panicOnInvalid("turboquant.NewSplitMSE", validateBitWidth(spec.RegularBits))
	stored := cloneSplitSpec(spec)
	q := &SplitQuantizer{
		dim:    dim,
		seed:   seed,
		spec:   stored,
		outIdx: stored.Outliers,
		regIdx: splitComplement(dim, stored.Outliers),
	}
	if len(q.outIdx) > 0 {
		panicOnInvalid("turboquant.NewSplitMSE", validateBitWidth(spec.OutlierBits))
		q.out = NewHadamardWithSeed(len(q.outIdx), spec.OutlierBits, seed^splitOutlierSeedMix)
	}
	q.reg = NewHadamardWithSeed(len(q.regIdx), spec.RegularBits, seed)
	outLen, regLen := len(q.outIdx), len(q.regIdx)
	q.pool.New = func() any {
		return &splitScratch{out: make([]float32, outLen), reg: make([]float32, regLen)}
	}
	return q
}

// Dim returns the full vector dimension.
func (q *SplitQuantizer) Dim() int { return q.dim }

// Seed returns the seed used to construct this quantizer.
func (q *SplitQuantizer) Seed() int64 { return q.seed }

// Spec returns a copy of the channel split specification.
func (q *SplitQuantizer) Spec() SplitSpec { return cloneSplitSpec(q.spec) }

// EffectiveBits returns the average code bits per channel.
func (q *SplitQuantizer) EffectiveBits() float64 { return q.spec.EffectiveBits(q.dim) }

// OutlierQuantizer returns the outlier-side sub-quantizer, or nil when the
// spec has no outlier channels.
func (q *SplitQuantizer) OutlierQuantizer() *Quantizer { return q.out }

// RegularQuantizer returns the regular-side sub-quantizer.
func (q *SplitQuantizer) RegularQuantizer() *Quantizer { return q.reg }

// AllocQuantized allocates storage shaped for this quantizer.
func (q *SplitQuantizer) AllocQuantized() SplitQuantized {
	var qx SplitQuantized
	if q.out != nil {
		qx.Out = make([]byte, PackedSize(len(q.outIdx), q.spec.OutlierBits))
	}
	qx.Reg = make([]byte, PackedSize(len(q.regIdx), q.spec.RegularBits))
	return qx
}

// Quantize compresses vec with both sub-quantizers.
func (q *SplitQuantizer) Quantize(vec []float32) SplitQuantized {
	qx := q.AllocQuantized()
	q.QuantizeTo(&qx, vec)
	return qx
}

// QuantizeTo compresses vec into caller-owned storage.
func (q *SplitQuantizer) QuantizeTo(dst *SplitQuantized, vec []float32) {
	if dst == nil {
		panic("(*SplitQuantizer).QuantizeTo: turboquant: nil destination")
	}
	panicOnInvalid("(*SplitQuantizer).QuantizeTo", ValidateVector(q.dim, vec))
	buf := q.pool.Get().(*splitScratch)
	defer q.pool.Put(buf)
	if q.out != nil {
		gatherInto(buf.out, vec, q.outIdx)
		dst.OutNorm = q.out.QuantizeTo(dst.Out, buf.out)
	}
	gatherInto(buf.reg, vec, q.regIdx)
	dst.RegNorm = q.reg.QuantizeTo(dst.Reg, buf.reg)
}

// Dequantize reconstructs an approximate vector at the original input scale.
func (q *SplitQuantizer) Dequantize(qx SplitQuantized) []float32 {
	recon := make([]float32, q.dim)
	q.DequantizeTo(recon, qx)
	return recon
}

// DequantizeTo reconstructs into caller-owned storage of length Dim().
func (q *SplitQuantizer) DequantizeTo(dst []float32, qx SplitQuantized) {
	if len(dst) != q.dim {
		panic(fmt.Sprintf("(*SplitQuantizer).DequantizeTo: turboquant: expected destination length %d, got %d", q.dim, len(dst)))
	}
	buf := q.pool.Get().(*splitScratch)
	defer q.pool.Put(buf)
	q.reg.DequantizeTo(buf.reg, qx.Reg)
	for i, ch := range q.regIdx {
		dst[ch] = qx.RegNorm * buf.reg[i]
	}
	if q.out != nil {
		q.out.DequantizeTo(buf.out, qx.Out)
		for i, ch := range q.outIdx {
			dst[ch] = qx.OutNorm * buf.out[i]
		}
	}
}
