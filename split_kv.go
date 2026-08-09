package turboquant

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"sync"
)

// SplitKVCachePage is an append-only quantized KV page whose keys and values
// use channel-split quantizers (SplitIPQuantizer / SplitQuantizer). It is the
// building block for the paper's fractional-bit KV configurations, for
// example 2.5-bit keys (32 channels at 4 bits + 96 at 2 bits over head
// dimension 128). Scoring runs on the CPU; the GPU scorer backends do not
// support split pages yet.
type SplitKVCachePage struct {
	mu sync.RWMutex

	keyQ   *SplitIPQuantizer
	valueQ *SplitQuantizer

	keyOut  splitKeyStore // unused when the key spec has no outliers
	keyReg  splitKeyStore
	valOut  splitValueStore // unused when the value spec has no outliers
	valReg  splitValueStore

	length int
	cap    int

	tmpPool sync.Pool
}

type splitKeyStore struct {
	mseBytes  int
	signBytes int
	mse       []byte
	signs     []byte
	resNorms  []float32
	norms     []float32
}

func newSplitKeyStore(q *IPQuantizer, capacity int) splitKeyStore {
	if q == nil {
		return splitKeyStore{}
	}
	mseBytes, signBytes := IPQuantizedSizes(q.Dim(), q.BitWidth())
	return splitKeyStore{
		mseBytes:  mseBytes,
		signBytes: signBytes,
		mse:       make([]byte, capacity*mseBytes),
		signs:     make([]byte, capacity*signBytes),
		resNorms:  make([]float32, capacity),
		norms:     make([]float32, capacity),
	}
}

func (s *splitKeyStore) at(pos int) IPQuantized {
	return IPQuantized{
		MSE:     s.mse[pos*s.mseBytes : (pos+1)*s.mseBytes],
		Signs:   s.signs[pos*s.signBytes : (pos+1)*s.signBytes],
		ResNorm: s.resNorms[pos],
		Norm:    s.norms[pos],
	}
}

func (s *splitKeyStore) store(pos int, qx IPQuantized) {
	s.resNorms[pos] = qx.ResNorm
	s.norms[pos] = qx.Norm
}

func (s *splitKeyStore) grow(length, newCap int) {
	mse := make([]byte, newCap*s.mseBytes)
	copy(mse, s.mse[:length*s.mseBytes])
	s.mse = mse
	signs := make([]byte, newCap*s.signBytes)
	copy(signs, s.signs[:length*s.signBytes])
	s.signs = signs
	resNorms := make([]float32, newCap)
	copy(resNorms, s.resNorms[:length])
	s.resNorms = resNorms
	norms := make([]float32, newCap)
	copy(norms, s.norms[:length])
	s.norms = norms
}

type splitValueStore struct {
	packedBytes int
	packed      []byte
	norms       []float32
}

func newSplitValueStore(q *Quantizer, bits, capacity int) splitValueStore {
	if q == nil {
		return splitValueStore{}
	}
	packedBytes := PackedSize(q.Dim(), bits)
	return splitValueStore{
		packedBytes: packedBytes,
		packed:      make([]byte, capacity*packedBytes),
		norms:       make([]float32, capacity),
	}
}

func (s *splitValueStore) at(pos int) []byte {
	return s.packed[pos*s.packedBytes : (pos+1)*s.packedBytes]
}

func (s *splitValueStore) grow(length, newCap int) {
	packed := make([]byte, newCap*s.packedBytes)
	copy(packed, s.packed[:length*s.packedBytes])
	s.packed = packed
	norms := make([]float32, newCap)
	copy(norms, s.norms[:length])
	s.norms = norms
}

// NewSplitKVCachePage creates a split KV page with a random seed.
func NewSplitKVCachePage(keyDim int, keySpec SplitSpec, valueDim int, valueSpec SplitSpec, capacity int) *SplitKVCachePage {
	var seedBytes [8]byte
	_, _ = rand.Read(seedBytes[:])
	return NewSplitKVCachePageWithSeed(keyDim, keySpec, valueDim, valueSpec, capacity, int64(binary.LittleEndian.Uint64(seedBytes[:])))
}

// NewSplitKVCachePageWithSeed creates a deterministic split KV page. The
// value quantizer derives its seed the same way KVCachePage does, so key and
// value randomness stay independent.
func NewSplitKVCachePageWithSeed(keyDim int, keySpec SplitSpec, valueDim int, valueSpec SplitSpec, capacity int, seed int64) *SplitKVCachePage {
	if capacity < 0 {
		panic("turboquant: split KV cache page capacity must be >= 0")
	}
	keyQ := NewSplitIPWithSeed(keyDim, keySpec, seed)
	valueQ := NewSplitMSEWithSeed(valueDim, valueSpec, seed^0x6b765f76616c7565)
	p := &SplitKVCachePage{
		keyQ:   keyQ,
		valueQ: valueQ,
		keyReg: newSplitKeyStore(keyQ.RegularQuantizer(), capacity),
		valReg: newSplitValueStore(valueQ.RegularQuantizer(), valueQ.Spec().RegularBits, capacity),
		cap:    capacity,
	}
	if outQ := keyQ.OutlierQuantizer(); outQ != nil {
		p.keyOut = newSplitKeyStore(outQ, capacity)
	}
	if outQ := valueQ.OutlierQuantizer(); outQ != nil {
		p.valOut = newSplitValueStore(outQ, valueQ.Spec().OutlierBits, capacity)
	}
	p.tmpPool.New = func() any {
		return &kvValueScratch{tmp: make([]float32, valueDim)}
	}
	return p
}

// KeyQuantizer returns the split key quantizer used by this page.
func (p *SplitKVCachePage) KeyQuantizer() *SplitIPQuantizer { return p.keyQ }

// ValueQuantizer returns the split value quantizer used by this page.
func (p *SplitKVCachePage) ValueQuantizer() *SplitQuantizer { return p.valueQ }

// EffectiveKeyBits returns the average key code bits per channel.
func (p *SplitKVCachePage) EffectiveKeyBits() float64 { return p.keyQ.EffectiveBits() }

// EffectiveValueBits returns the average value code bits per channel.
func (p *SplitKVCachePage) EffectiveValueBits() float64 { return p.valueQ.EffectiveBits() }

// Len returns the number of stored KV entries.
func (p *SplitKVCachePage) Len() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.length
}

// Cap returns the current storage capacity in entries.
func (p *SplitKVCachePage) Cap() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.cap
}

// Reset clears the page length. Storage is retained.
func (p *SplitKVCachePage) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.length = 0
}

// Append quantizes and appends one key/value pair.
func (p *SplitKVCachePage) Append(key, value []float32) {
	panicOnInvalid("(*SplitKVCachePage).Append", ValidateVector(p.keyQ.Dim(), key))
	panicOnInvalid("(*SplitKVCachePage).Append", ValidateVector(p.valueQ.Dim(), value))

	p.mu.Lock()
	defer p.mu.Unlock()
	p.growLocked(p.length + 1)
	slot := p.length

	kq := SplitIPQuantized{Reg: p.keyReg.at(slot)}
	if p.keyQ.OutlierQuantizer() != nil {
		kq.Out = p.keyOut.at(slot)
	}
	p.keyQ.QuantizeTo(&kq, key)
	p.keyReg.store(slot, kq.Reg)
	if p.keyQ.OutlierQuantizer() != nil {
		p.keyOut.store(slot, kq.Out)
	}

	vq := SplitQuantized{Reg: p.valReg.at(slot)}
	if p.valueQ.OutlierQuantizer() != nil {
		vq.Out = p.valOut.at(slot)
	}
	p.valueQ.QuantizeTo(&vq, value)
	p.valReg.norms[slot] = vq.RegNorm
	if p.valueQ.OutlierQuantizer() != nil {
		p.valOut.norms[slot] = vq.OutNorm
	}

	p.length++
}

// AddBatch quantizes and appends multiple key/value pairs.
func (p *SplitKVCachePage) AddBatch(keys, values [][]float32) {
	if len(keys) != len(values) {
		panic("(*SplitKVCachePage).AddBatch: turboquant: key/value length mismatch")
	}
	for i := range keys {
		p.Append(keys[i], values[i])
	}
}

// PrepareQuery builds reusable key-side query state for repeated lookups.
func (p *SplitKVCachePage) PrepareQuery(query []float32) SplitPreparedQuery {
	return p.keyQ.PrepareQuery(query)
}

// TopKPreparedTo scores all stored keys against spq and writes the top
// len(indices) results in descending score order. Ties break toward the
// lower slot index.
func (p *SplitKVCachePage) TopKPreparedTo(indices []uint32, scores []float32, spq SplitPreparedQuery) {
	if len(indices) != len(scores) {
		panic("(*SplitKVCachePage).TopKPreparedTo: turboquant: indices/scores length mismatch")
	}
	panicOnInvalid("(*SplitKVCachePage).TopKPreparedTo", p.keyQ.validatePreparedQuery(spq))
	for i := range indices {
		indices[i] = 0
		scores[i] = 0
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	if len(indices) > p.length {
		panic(fmt.Sprintf("(*SplitKVCachePage).TopKPreparedTo: turboquant: requested top-k %d exceeds page length %d", len(indices), p.length))
	}

	regQ := p.keyQ.RegularQuantizer()
	outQ := p.keyQ.OutlierQuantizer()
	filled := 0
	for slot := 0; slot < p.length; slot++ {
		score := regQ.InnerProductPreparedTrusted(p.keyReg.at(slot), spq.Reg)
		if outQ != nil {
			score += outQ.InnerProductPreparedTrusted(p.keyOut.at(slot), spq.Out)
		}
		candidate := uint32(slot)
		insert := filled
		for pos := 0; pos < filled; pos++ {
			if score > scores[pos] || (score == scores[pos] && candidate < indices[pos]) {
				insert = pos
				break
			}
		}
		if filled < len(indices) {
			filled++
		} else if insert == filled {
			continue
		}
		for move := filled - 1; move > insert; move-- {
			indices[move] = indices[move-1]
			scores[move] = scores[move-1]
		}
		indices[insert] = candidate
		scores[insert] = score
	}
}

// AttentionOutputPreparedTo runs top-k selection, softmax normalization, and
// weighted approximate value reconstruction into dst. It returns the
// selected positions and their attention weights.
//
// Like KVCachePage, this page is append-only and its read methods take the
// read lock per step; a concurrent Reset is not safe against an in-flight
// read. Serialize Reset against readers externally if you need both.
func (p *SplitKVCachePage) AttentionOutputPreparedTo(dst []float32, spq SplitPreparedQuery, k int) ([]uint32, []float32) {
	if len(dst) != p.valueQ.Dim() {
		panic(fmt.Sprintf("(*SplitKVCachePage).AttentionOutputPreparedTo: turboquant: expected destination length %d, got %d", p.valueQ.Dim(), len(dst)))
	}
	for i := range dst {
		dst[i] = 0
	}
	if k <= 0 {
		return nil, nil
	}
	if n := p.Len(); k > n {
		k = n
	}
	if k == 0 {
		return nil, nil
	}
	indices := make([]uint32, k)
	weights := make([]float32, k)
	p.TopKPreparedTo(indices, weights, spq)
	softmaxInPlace(weights)

	scratch := p.tmpPool.Get().(*kvValueScratch)
	tmp := scratch.tmp
	defer p.tmpPool.Put(scratch)

	p.mu.RLock()
	defer p.mu.RUnlock()
	regQ := p.valueQ.RegularQuantizer()
	outQ := p.valueQ.OutlierQuantizer()
	for i, idx := range indices {
		slot := int(idx)
		if slot < 0 || slot >= p.length {
			panic(fmt.Sprintf("turboquant: split value index %d out of bounds for page length %d", slot, p.length))
		}
		vq := SplitQuantized{
			Reg:     p.valReg.at(slot),
			RegNorm: p.valReg.norms[slot],
		}
		if outQ != nil {
			vq.Out = p.valOut.at(slot)
			vq.OutNorm = p.valOut.norms[slot]
		}
		regQ.DequantizeTo(tmp[:regQ.Dim()], vq.Reg)
		w := weights[i]
		for j, ch := range p.valueQ.regIdx {
			dst[ch] += w * vq.RegNorm * tmp[j]
		}
		if outQ != nil {
			outQ.DequantizeTo(tmp[:outQ.Dim()], vq.Out)
			for j, ch := range p.valueQ.outIdx {
				dst[ch] += w * vq.OutNorm * tmp[j]
			}
		}
	}
	return indices, weights
}

func (p *SplitKVCachePage) growLocked(want int) {
	if want <= p.cap {
		return
	}
	newCap := p.cap * 2
	if newCap < want {
		newCap = want
	}
	if newCap < 4 {
		newCap = 4
	}
	if p.keyQ.OutlierQuantizer() != nil {
		p.keyOut.grow(p.length, newCap)
	}
	p.keyReg.grow(p.length, newCap)
	if p.valueQ.OutlierQuantizer() != nil {
		p.valOut.grow(p.length, newCap)
	}
	p.valReg.grow(p.length, newCap)
	p.cap = newCap
}

// StorageBytes reports the currently allocated storage footprint.
func (p *SplitKVCachePage) StorageBytes() uint64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	total := uint64(len(p.keyReg.mse) + len(p.keyReg.signs) + len(p.valReg.packed))
	total += uint64(len(p.keyReg.resNorms)+len(p.keyReg.norms)+len(p.valReg.norms)) * 4
	if p.keyQ.OutlierQuantizer() != nil {
		total += uint64(len(p.keyOut.mse) + len(p.keyOut.signs))
		total += uint64(len(p.keyOut.resNorms)+len(p.keyOut.norms)) * 4
	}
	if p.valueQ.OutlierQuantizer() != nil {
		total += uint64(len(p.valOut.packed))
		total += uint64(len(p.valOut.norms)) * 4
	}
	return total
}
