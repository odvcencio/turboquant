package turboquant

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	mrand "math/rand"
	"sync"
)

// Quantizer compresses float32 vectors to packed bit representations using
// the TurboQuant MSE-optimal algorithm. Safe for concurrent use after construction.
type Quantizer struct {
	dim      int
	bitWidth int
	seed     int64
	rotation rotationState
	portable rotationKind
	cb       codebook
	pool     sync.Pool
}

type scratchBuf struct {
	rotated []float32
	tmp     []float32
	work    []float32
	indices []int
}

// New creates a TurboQuant MSE-optimal quantizer with a random seed.
// The default constructor uses the fast structured Walsh-Hadamard rotation.
func New(dim, bitWidth int) *Quantizer {
	var seedBytes [8]byte
	rand.Read(seedBytes[:])
	seed := int64(binary.LittleEndian.Uint64(seedBytes[:]))
	return NewHadamardWithSeed(dim, bitWidth, seed)
}

// NewHadamard creates a TurboQuant quantizer backed by a fast structured
// Walsh-Hadamard rotation instead of a dense QR rotation.
func NewHadamard(dim, bitWidth int) *Quantizer {
	var seedBytes [8]byte
	rand.Read(seedBytes[:])
	seed := int64(binary.LittleEndian.Uint64(seedBytes[:]))
	return NewHadamardWithSeed(dim, bitWidth, seed)
}

// NewDense creates a TurboQuant quantizer with the legacy dense QR rotation.
func NewDense(dim, bitWidth int) *Quantizer {
	var seedBytes [8]byte
	rand.Read(seedBytes[:])
	seed := int64(binary.LittleEndian.Uint64(seedBytes[:]))
	return NewDenseWithSeed(dim, bitWidth, seed)
}

// NewWithSeed creates a deterministic MSE-optimal quantizer using the default
// fast structured Walsh-Hadamard rotation. Two quantizers with the same dim,
// bitWidth, and seed produce identical output.
func NewWithSeed(dim, bitWidth int, seed int64) *Quantizer {
	return NewHadamardWithSeed(dim, bitWidth, seed)
}

// NewDenseWithSeed creates a deterministic MSE-optimal quantizer with the
// legacy dense QR rotation.
func NewDenseWithSeed(dim, bitWidth int, seed int64) *Quantizer {
	rng := mrand.New(mrand.NewSource(seed))
	return newQuantizerWithRotation(dim, bitWidth, seed, newDenseRotation(dim, rng), cachedCodebook(dim, bitWidth))
}

// NewHadamardWithSeed creates a deterministic MSE-optimal quantizer with a
// structured Walsh-Hadamard rotation.
func NewHadamardWithSeed(dim, bitWidth int, seed int64) *Quantizer {
	rng := mrand.New(mrand.NewSource(seed))
	return newQuantizerWithRotation(dim, bitWidth, seed, newHadamardRotation(dim, rng), cachedCodebook(dim, bitWidth))
}

func newQuantizerWithRotation(dim, bitWidth int, seed int64, rotation rotationState, cb codebook) *Quantizer {
	panicOnInvalid("turboquant.New", validateDim(dim))
	panicOnInvalid("turboquant.New", validateBitWidth(bitWidth))
	q := &Quantizer{
		dim:      dim,
		bitWidth: bitWidth,
		seed:     seed,
		rotation: rotation,
		portable: rotation.kind,
		cb:       cb,
	}
	q.pool.New = func() any {
		return &scratchBuf{
			rotated: make([]float32, dim),
			tmp:     make([]float32, dim),
			work:    make([]float32, dim),
			indices: make([]int, dim),
		}
	}
	return q
}

// Dim returns the vector dimension.
func (q *Quantizer) Dim() int { return q.dim }

// BitWidth returns the bits per coordinate.
func (q *Quantizer) BitWidth() int { return q.bitWidth }

// Seed returns the seed used to construct this quantizer.
func (q *Quantizer) Seed() int64 { return q.seed }

// RotationKind returns the internal rotation family used by this quantizer.
func (q *Quantizer) RotationKind() string { return q.rotation.kindString() }

// Quantize compresses a float32 vector to packed bytes.
// Returns packed indices and the original vector norm (for rescaling on dequantize).
func (q *Quantizer) Quantize(vec []float32) ([]byte, float32) {
	packed := make([]byte, packedSize(q.dim, q.bitWidth))
	norm := q.QuantizeTo(packed, vec)
	return packed, norm
}

// QuantizeTo compresses vec into caller-provided storage and returns the
// original vector norm. dst must be length PackedSize(q.Dim(), q.BitWidth()).
func (q *Quantizer) QuantizeTo(dst []byte, vec []float32) float32 {
	panicOnInvalid("(*Quantizer).QuantizeTo", ValidateVector(q.dim, vec))
	panicOnInvalid("(*Quantizer).QuantizeTo", ValidatePacked(q.dim, q.bitWidth, dst))
	buf := q.pool.Get().(*scratchBuf)
	defer q.pool.Put(buf)

	norm := q.quantizeToBuf(dst, vec, buf)
	return norm
}

func (q *Quantizer) quantizeToBuf(dst []byte, vec []float32, buf *scratchBuf) float32 {
	norm := vecNorm(vec)

	scale := float32(1.0)
	if norm > 1e-12 {
		scale = 1.0 / norm
	}
	for i, v := range vec {
		buf.tmp[i] = v * scale
	}

	q.rotation.apply(buf.rotated, buf.tmp, buf.work)
	for i := 0; i < q.dim; i++ {
		buf.indices[i] = q.cb.nearestCentroid(buf.rotated[i])
	}
	packIndices(dst, buf.indices, q.bitWidth)
	return norm
}

// Dequantize reconstructs an approximate unit-norm float32 vector.
// Multiply by the norm from Quantize to recover original scale.
func (q *Quantizer) Dequantize(packed []byte) []float32 {
	result := make([]float32, q.dim)
	q.DequantizeTo(result, packed)
	return result
}

// DequantizeTo reconstructs an approximate unit-norm vector into caller-owned
// storage. dst must be length q.Dim().
func (q *Quantizer) DequantizeTo(dst []float32, packed []byte) {
	if len(dst) != q.dim {
		panic(fmt.Sprintf("(*Quantizer).DequantizeTo: turboquant: expected destination length %d, got %d", q.dim, len(dst)))
	}
	panicOnInvalid("(*Quantizer).DequantizeTo", ValidatePacked(q.dim, q.bitWidth, packed))
	buf := q.pool.Get().(*scratchBuf)
	defer q.pool.Put(buf)
	q.dequantizeToBuf(dst, packed, buf)
}

func (q *Quantizer) dequantizeToBuf(dst []float32, packed []byte, buf *scratchBuf) {
	unpackIndices(buf.indices, packed, q.dim, q.bitWidth)
	for i := 0; i < q.dim; i++ {
		buf.rotated[i] = q.cb.centroidValue(buf.indices[i])
	}
	q.rotation.applyInverse(dst, buf.rotated, buf.work)
}

// InnerProduct estimates <x, y> from MSE-quantized x and raw y.
// Computes in rotated domain without allocating a full dequantized vector.
func (q *Quantizer) InnerProduct(packed []byte, norm float32, y []float32) float32 {
	panicOnInvalid("(*Quantizer).InnerProduct", ValidatePacked(q.dim, q.bitWidth, packed))
	panicOnInvalid("(*Quantizer).InnerProduct", ValidateVector(q.dim, y))
	if math.IsNaN(float64(norm)) || math.IsInf(float64(norm), 0) {
		panic(fmt.Sprintf("(*Quantizer).InnerProduct: turboquant: invalid norm %v", norm))
	}
	buf := q.pool.Get().(*scratchBuf)
	defer q.pool.Put(buf)

	q.rotation.apply(buf.rotated, y, buf.work)
	return q.innerProductRotatedBuf(packed, norm, buf.rotated, buf.indices)
}

func (q *Quantizer) innerProductRotated(packed []byte, norm float32, rotY []float32) float32 {
	buf := q.pool.Get().(*scratchBuf)
	defer q.pool.Put(buf)
	return q.innerProductRotatedBuf(packed, norm, rotY, buf.indices)
}

func (q *Quantizer) innerProductPrepared(packed []byte, norm float32, rotY, lut []float32, lutBitWidth int) float32 {
	if len(rotY) != q.dim {
		panic(fmt.Sprintf("turboquant: expected rotated query length %d, got %d", q.dim, len(rotY)))
	}
	wantLUT := preparedQueryMSELUTLen(q.dim, q.bitWidth)
	if wantLUT != 0 && lutBitWidth == q.bitWidth && len(lut) == wantLUT {
		var dot float32
		for i, packedByte := range packed {
			dot += lut[i*256+int(packedByte)]
		}
		return dot * norm
	}
	return q.innerProductRotated(packed, norm, rotY)
}

func (q *Quantizer) innerProductRotatedBuf(packed []byte, norm float32, rotY []float32, indices []int) float32 {
	if len(rotY) != q.dim {
		panic(fmt.Sprintf("turboquant: expected rotated query length %d, got %d", q.dim, len(rotY)))
	}
	unpackIndices(indices, packed, q.dim, q.bitWidth)
	var dot float32
	for i := 0; i < q.dim; i++ {
		dot += q.cb.centroidValue(indices[i]) * rotY[i]
	}
	return dot * norm
}

func (q *Quantizer) buildPreparedDotLUT(dst, rotY []float32) {
	if len(rotY) != q.dim {
		panic(fmt.Sprintf("turboquant: expected rotated query length %d, got %d", q.dim, len(rotY)))
	}
	want := preparedQueryMSELUTLen(q.dim, q.bitWidth)
	if len(dst) != want {
		panic(fmt.Sprintf("turboquant: expected prepared MSE LUT length %d, got %d", want, len(dst)))
	}

	switch q.bitWidth {
	case 1:
		for byteIdx := 0; byteIdx < len(dst)/256; byteIdx++ {
			base := byteIdx * 8
			var low [16]float32
			var high [16]float32
			if base+8 <= q.dim {
				buildBinaryNibbleTable4(&low, q.cb.centroids, rotY[base], rotY[base+1], rotY[base+2], rotY[base+3])
				buildBinaryNibbleTable4(&high, q.cb.centroids, rotY[base+4], rotY[base+5], rotY[base+6], rotY[base+7])
			} else {
				lowCount := q.dim - base
				if lowCount > 4 {
					lowCount = 4
				}
				if lowCount < 0 {
					lowCount = 0
				}
				highBase := base + 4
				highCount := q.dim - highBase
				if highCount > 4 {
					highCount = 4
				}
				if highCount < 0 {
					highCount = 0
				}
				buildPackedNibbleTable(&low, q.cb.centroids, rotY, base, lowCount, q.bitWidth)
				buildPackedNibbleTable(&high, q.cb.centroids, rotY, highBase, highCount, q.bitWidth)
			}
			combineNibbleTables256(dst[byteIdx*256:(byteIdx+1)*256], &low, &high)
		}
	case 2:
		for byteIdx := 0; byteIdx < len(dst)/256; byteIdx++ {
			base := byteIdx * 4
			var low [16]float32
			var high [16]float32
			if base+4 <= q.dim {
				buildTwoBitNibbleTable2(&low, q.cb.centroids, rotY[base], rotY[base+1])
				buildTwoBitNibbleTable2(&high, q.cb.centroids, rotY[base+2], rotY[base+3])
			} else {
				lowCount := q.dim - base
				if lowCount > 2 {
					lowCount = 2
				}
				if lowCount < 0 {
					lowCount = 0
				}
				highBase := base + 2
				highCount := q.dim - highBase
				if highCount > 2 {
					highCount = 2
				}
				if highCount < 0 {
					highCount = 0
				}
				buildPackedNibbleTable(&low, q.cb.centroids, rotY, base, lowCount, q.bitWidth)
				buildPackedNibbleTable(&high, q.cb.centroids, rotY, highBase, highCount, q.bitWidth)
			}
			combineNibbleTables256(dst[byteIdx*256:(byteIdx+1)*256], &low, &high)
		}
	case 4:
		for byteIdx := 0; byteIdx < len(dst)/256; byteIdx++ {
			base := byteIdx * 2
			var low [16]float32
			var high [16]float32
			if base+2 <= q.dim {
				buildFourBitNibbleTable1(&low, q.cb.centroids, rotY[base])
				buildFourBitNibbleTable1(&high, q.cb.centroids, rotY[base+1])
			} else {
				lowCount := q.dim - base
				if lowCount > 1 {
					lowCount = 1
				}
				if lowCount < 0 {
					lowCount = 0
				}
				highBase := base + 1
				highCount := q.dim - highBase
				if highCount > 1 {
					highCount = 1
				}
				if highCount < 0 {
					highCount = 0
				}
				buildPackedNibbleTable(&low, q.cb.centroids, rotY, base, lowCount, q.bitWidth)
				buildPackedNibbleTable(&high, q.cb.centroids, rotY, highBase, highCount, q.bitWidth)
			}
			combineNibbleTables256(dst[byteIdx*256:(byteIdx+1)*256], &low, &high)
		}
	case 8:
		for byteIdx := 0; byteIdx < len(dst)/256; byteIdx++ {
			table := dst[byteIdx*256 : (byteIdx+1)*256]
			weight := rotY[byteIdx]
			for value := 0; value < 256; value++ {
				table[value] = q.cb.centroids[value] * weight
			}
		}
	default:
		panic(fmt.Sprintf("turboquant: unsupported prepared MSE LUT bit width %d", q.bitWidth))
	}
}

func buildBinaryNibbleTable4(dst *[16]float32, centroids []float32, y0, y1, y2, y3 float32) {
	var p0 [2]float32
	var p1 [2]float32
	var p2 [2]float32
	var p3 [2]float32
	p0[0], p0[1] = centroids[0]*y0, centroids[1]*y0
	p1[0], p1[1] = centroids[0]*y1, centroids[1]*y1
	p2[0], p2[1] = centroids[0]*y2, centroids[1]*y2
	p3[0], p3[1] = centroids[0]*y3, centroids[1]*y3
	for b3 := 0; b3 < 2; b3++ {
		v3 := p3[b3]
		base3 := b3 << 3
		for b2 := 0; b2 < 2; b2++ {
			v23 := v3 + p2[b2]
			base23 := base3 | (b2 << 2)
			for b1 := 0; b1 < 2; b1++ {
				v231 := v23 + p1[b1]
				base231 := base23 | (b1 << 1)
				dst[base231] = v231 + p0[0]
				dst[base231|1] = v231 + p0[1]
			}
		}
	}
}

func buildTwoBitNibbleTable2(dst *[16]float32, centroids []float32, y0, y1 float32) {
	var p0 [4]float32
	var p1 [4]float32
	for i := 0; i < 4; i++ {
		p0[i] = centroids[i] * y0
		p1[i] = centroids[i] * y1
	}
	for d1 := 0; d1 < 4; d1++ {
		v1 := p1[d1]
		row := d1 << 2
		for d0 := 0; d0 < 4; d0++ {
			dst[row|d0] = p0[d0] + v1
		}
	}
}

func buildFourBitNibbleTable1(dst *[16]float32, centroids []float32, y float32) {
	for value := 0; value < 16; value++ {
		dst[value] = centroids[value] * y
	}
}

func buildPackedNibbleTable(dst *[16]float32, centroids, rotY []float32, base, count, bitWidth int) {
	size := 1 << (count * bitWidth)
	maskWidth := (1 << bitWidth) - 1
	for value := 0; value < size; value++ {
		mask := value
		var sum float32
		for i := 0; i < count; i++ {
			sum += centroids[mask&maskWidth] * rotY[base+i]
			mask >>= bitWidth
		}
		dst[value] = sum
	}
	for offset := size; offset < len(dst); offset += size {
		copy(dst[offset:offset+size], dst[:size])
	}
}

// QuantizeBatch compresses multiple vectors.
func (q *Quantizer) QuantizeBatch(vecs [][]float32) ([][]byte, []float32) {
	packed := make([][]byte, len(vecs))
	norms := make([]float32, len(vecs))
	for i, v := range vecs {
		packed[i], norms[i] = q.Quantize(v)
	}
	return packed, norms
}

// DequantizeBatch reconstructs multiple vectors.
func (q *Quantizer) DequantizeBatch(packed [][]byte) [][]float32 {
	result := make([][]float32, len(packed))
	for i, p := range packed {
		result[i] = q.Dequantize(p)
	}
	return result
}
