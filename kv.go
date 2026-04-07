package turboquant

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

type kvValueScratch struct {
	tmp []float32
}

type kvBatchScratch struct {
	scores  []float32
	filled  []int
	rotated []float32
	work    []float32
}

// KVCachePage stores quantized key/value vectors for one append-only cache page.
// Keys use IP quantization for fast query scoring; values use MSE quantization
// for reconstruction of approximate attention outputs.
type KVCachePage struct {
	mu sync.RWMutex

	keyQ   *IPQuantizer
	valueQ *Quantizer

	keyMSE      []byte
	keySigns    []byte
	keyResNorms []float32

	valuePacked []byte
	valueNorms  []float32

	length       int
	keyMSEBytes  int
	keySignBytes int
	valueBytes   int

	gpuKeys   *GPUPreparedScorer
	gpuValues kvGPUValueBackend
	tmpPool   sync.Pool
	batchPool sync.Pool
}

// KVPreparedQueryBatch keeps a prepared query batch resident on the same GPU
// page state as its parent KV cache page for repeated batch attention calls.
type KVPreparedQueryBatch struct {
	page      *KVCachePage
	pqs       []PreparedQuery
	gpuKeys   *GPUPreparedScorer
	gpuValues kvGPUValueBackend
	gpuBatch  *GPUPreparedQueryBatch
	count     int
	closed    bool
}

// NewKVCachePage creates an append-only quantized KV page using fast hadamard
// quantizers for both keys and values.
func NewKVCachePage(keyDim, keyBits, valueDim, valueBits, capacity int) *KVCachePage {
	var seedBytes [8]byte
	_, _ = rand.Read(seedBytes[:])
	seed := int64(binary.LittleEndian.Uint64(seedBytes[:]))
	return NewKVCachePageWithSeed(keyDim, keyBits, valueDim, valueBits, capacity, seed)
}

// NewKVCachePageWithSeed creates a deterministic append-only quantized KV page.
func NewKVCachePageWithSeed(keyDim, keyBits, valueDim, valueBits, capacity int, seed int64) *KVCachePage {
	return NewKVCachePageWithQuantizers(
		NewIPHadamardWithSeed(keyDim, keyBits, seed),
		NewHadamardWithSeed(valueDim, valueBits, seed^0x6b765f76616c7565),
		capacity,
	)
}

// NewKVCachePageWithQuantizers creates a page using caller-provided quantizers.
func NewKVCachePageWithQuantizers(keyQ *IPQuantizer, valueQ *Quantizer, capacity int) *KVCachePage {
	if keyQ == nil || valueQ == nil {
		panic("turboquant: KV cache page requires non-nil key and value quantizers")
	}
	if capacity < 0 {
		panic("turboquant: KV cache page capacity must be >= 0")
	}
	keyMSEBytes, keySignBytes := IPQuantizedSizes(keyQ.Dim(), keyQ.BitWidth())
	page := &KVCachePage{
		keyQ:         keyQ,
		valueQ:       valueQ,
		keyMSE:       make([]byte, capacity*keyMSEBytes),
		keySigns:     make([]byte, capacity*keySignBytes),
		keyResNorms:  make([]float32, capacity),
		valuePacked:  make([]byte, capacity*PackedSize(valueQ.Dim(), valueQ.BitWidth())),
		valueNorms:   make([]float32, capacity),
		keyMSEBytes:  keyMSEBytes,
		keySignBytes: keySignBytes,
		valueBytes:   PackedSize(valueQ.Dim(), valueQ.BitWidth()),
	}
	page.tmpPool.New = func() any {
		return &kvValueScratch{tmp: make([]float32, valueQ.Dim())}
	}
	page.batchPool.New = func() any {
		return &kvBatchScratch{}
	}
	return page
}

// KeyQuantizer returns the key quantizer shape used by this page.
func (p *KVCachePage) KeyQuantizer() *IPQuantizer { return p.keyQ }

// ValueQuantizer returns the value quantizer shape used by this page.
func (p *KVCachePage) ValueQuantizer() *Quantizer { return p.valueQ }

// Len returns the number of stored KV entries.
func (p *KVCachePage) Len() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.length
}

// Cap returns the current storage capacity in entries.
func (p *KVCachePage) Cap() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return cap(p.keyResNorms)
}

// GPUKeysEnabled reports whether keys are currently uploaded into the GPU scorer.
func (p *KVCachePage) GPUKeysEnabled() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.gpuKeys != nil
}

// GPUValuesEnabled reports whether values are currently uploaded into the GPU
// accumulation backend.
func (p *KVCachePage) GPUValuesEnabled() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.gpuValues != nil
}

// Reset clears the page length and releases any uploaded GPU state.
func (p *KVCachePage) Reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.disableGPUBackendsLocked()
	p.length = 0
}

// Append quantizes and appends one key/value pair.
func (p *KVCachePage) Append(key, value []float32) {
	panicOnInvalid("(*KVCachePage).Append", ValidateVector(p.keyQ.Dim(), key))
	panicOnInvalid("(*KVCachePage).Append", ValidateVector(p.valueQ.Dim(), value))

	p.mu.Lock()
	defer p.mu.Unlock()
	p.disableGPUBackendsLocked()
	p.growLocked(p.length + 1)
	slot := p.length
	keyDst := p.keyAt(slot)
	p.keyQ.QuantizeTo(&keyDst, key)
	p.valueNorms[slot] = p.valueQ.QuantizeTo(p.valuePackedAt(slot), value)
	p.length++
}

// AddBatch quantizes and appends multiple key/value pairs.
func (p *KVCachePage) AddBatch(keys, values [][]float32) {
	if len(keys) != len(values) {
		panic("(*KVCachePage).AddBatch: turboquant: key/value length mismatch")
	}
	for i := range keys {
		panicOnInvalid("(*KVCachePage).AddBatch", ValidateVector(p.keyQ.Dim(), keys[i]))
		panicOnInvalid("(*KVCachePage).AddBatch", ValidateVector(p.valueQ.Dim(), values[i]))
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.disableGPUBackendsLocked()
	p.growLocked(p.length + len(keys))
	for i := range keys {
		slot := p.length + i
		keyDst := p.keyAt(slot)
		p.keyQ.QuantizeTo(&keyDst, keys[i])
		p.valueNorms[slot] = p.valueQ.QuantizeTo(p.valuePackedAt(slot), values[i])
	}
	p.length += len(keys)
}

// PrepareQuery builds reusable key-side query state for repeated lookups.
func (p *KVCachePage) PrepareQuery(query []float32) PreparedQuery {
	return p.keyQ.PrepareQuery(query)
}

// PrepareQueryTo writes reusable key-side query state into caller-owned storage.
func (p *KVCachePage) PrepareQueryTo(dst *PreparedQuery, query []float32) {
	p.keyQ.PrepareQueryTo(dst, query)
}

// UploadPreparedQueries uploads a prepared-query batch into the currently
// enabled GPU page state for repeated batch top-k or attention calls.
func (p *KVCachePage) UploadPreparedQueries(pqs []PreparedQuery) (*KVPreparedQueryBatch, error) {
	for i := range pqs {
		if err := ValidatePreparedQuery(p.keyQ.Dim(), pqs[i]); err != nil {
			return nil, err
		}
	}
	return p.UploadPreparedQueriesTrusted(pqs)
}

// UploadPreparedQueriesTrusted uploads a caller-validated prepared-query batch
// into the currently enabled GPU page state.
func (p *KVCachePage) UploadPreparedQueriesTrusted(pqs []PreparedQuery) (*KVPreparedQueryBatch, error) {
	if len(pqs) == 0 {
		return nil, fmt.Errorf("turboquant: cannot upload an empty KV prepared-query batch")
	}
	p.mu.RLock()
	gpuKeys := p.gpuKeys
	gpuValues := p.gpuValues
	p.mu.RUnlock()
	if gpuKeys == nil || gpuValues == nil {
		return nil, ErrGPUBackendUnavailable
	}
	gpuBatch, err := gpuKeys.UploadPreparedQueriesTrusted(pqs)
	if err != nil {
		return nil, err
	}
	stored := make([]PreparedQuery, len(pqs))
	copy(stored, pqs)
	return &KVPreparedQueryBatch{
		page:      p,
		pqs:       stored,
		gpuKeys:   gpuKeys,
		gpuValues: gpuValues,
		gpuBatch:  gpuBatch,
		count:     len(pqs),
	}, nil
}

// EnableGPUKeys uploads the current key corpus into the experimental GPU scorer.
// Native CUDA builds also upload value-side state for device-side rotated-domain
// accumulation during AttentionOutputPreparedTo.
func (p *KVCachePage) EnableGPUKeys() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.disableGPUBackendsLocked()
	if p.length == 0 {
		return nil
	}
	scorer, err := p.keyQ.NewGPUPreparedScorerFromData(p.packGPUKeyDataLocked())
	if err != nil {
		return err
	}
	p.gpuKeys = scorer
	values, err := newKVGPUValueBackend(p)
	if err != nil {
		p.disableGPUBackendsLocked()
		return err
	}
	p.gpuValues = values
	return nil
}

// DisableGPUKeys releases any uploaded GPU page state.
func (p *KVCachePage) DisableGPUKeys() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.disableGPUBackendsLocked()
}

// Len reports the number of prepared queries kept in this uploaded batch.
func (b *KVPreparedQueryBatch) Len() int {
	if b == nil {
		return 0
	}
	return b.count
}

func (b *KVPreparedQueryBatch) validateLivePage() error {
	if b == nil {
		return fmt.Errorf("turboquant: nil KV prepared-query batch")
	}
	if b.closed {
		return fmt.Errorf("turboquant: KV prepared-query batch is closed")
	}
	if b.page == nil || b.gpuBatch == nil {
		return fmt.Errorf("turboquant: KV prepared-query batch has no backing GPU state")
	}
	b.page.mu.RLock()
	live := b.page.gpuKeys == b.gpuKeys && b.page.gpuValues == b.gpuValues && b.page.gpuKeys != nil && b.page.gpuValues != nil
	b.page.mu.RUnlock()
	if !live {
		return fmt.Errorf("turboquant: KV prepared-query batch was invalidated by page mutation or GPU reset")
	}
	return nil
}

// TopKTo scores the uploaded query batch against the current page keys and
// writes flattened query-major top-k results into caller-owned storage.
func (b *KVPreparedQueryBatch) TopKTo(indices []uint32, scores []float32) error {
	if err := b.validateLivePage(); err != nil {
		return err
	}
	if len(indices) != len(scores) {
		return fmt.Errorf("turboquant: uploaded KV batch destination length mismatch: %d indices vs %d scores", len(indices), len(scores))
	}
	if b.count == 0 {
		if len(indices) != 0 {
			return fmt.Errorf("turboquant: empty uploaded KV batch requires empty destinations")
		}
		return nil
	}
	if len(indices)%b.count != 0 {
		return fmt.Errorf("turboquant: uploaded KV batch top-k buffer must be divisible by query count %d", b.count)
	}
	k := len(indices) / b.count
	return b.gpuBatch.ScoreTopKTo(indices, scores, k)
}

// AttentionOutputInto computes approximate attention outputs for the uploaded
// query batch, reusing the uploaded GPU query state and page GPU value state.
func (b *KVPreparedQueryBatch) AttentionOutputInto(dst []float32, indices []uint32, weights []float32) error {
	if err := b.validateLivePage(); err != nil {
		return err
	}
	if b.page == nil {
		return fmt.Errorf("turboquant: uploaded KV batch has no page")
	}
	page := b.page
	dim := page.valueQ.Dim()
	if len(dst) != b.count*dim {
		return fmt.Errorf("turboquant: expected uploaded KV batch destination length %d, got %d", b.count*dim, len(dst))
	}
	if len(indices) != len(weights) {
		return fmt.Errorf("turboquant: uploaded KV batch index/weight length mismatch")
	}
	if b.count == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return nil
	}
	if len(indices)%b.count != 0 {
		return fmt.Errorf("turboquant: uploaded KV batch top-k buffer must be divisible by query count %d", b.count)
	}
	k := len(indices) / b.count
	if k == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return nil
	}
	if err := b.gpuBatch.ScoreTopKTo(indices, weights, k); err != nil {
		return err
	}
	for queryIdx := 0; queryIdx < b.count; queryIdx++ {
		baseK := queryIdx * k
		softmaxInPlace(weights[baseK : baseK+k])
	}
	if page.tryGPUWeightedValueSumBatch(dst, indices, weights, b.count) {
		return nil
	}
	for queryIdx := 0; queryIdx < b.count; queryIdx++ {
		baseK := queryIdx * k
		baseDim := queryIdx * dim
		page.weightedValueSumTo(dst[baseDim:baseDim+dim], indices[baseK:baseK+k], weights[baseK:baseK+k])
	}
	return nil
}

// Close releases the uploaded GPU query batch.
func (b *KVPreparedQueryBatch) Close() error {
	if b == nil || b.closed {
		return nil
	}
	b.closed = true
	if b.gpuBatch != nil {
		return b.gpuBatch.Close()
	}
	return nil
}

// TopKPrepared returns the highest-scoring cached key positions for a prepared query.
func (p *KVCachePage) TopKPrepared(pq PreparedQuery, k int) ([]uint32, []float32) {
	if k <= 0 {
		return nil, nil
	}
	indices := make([]uint32, minInt(k, maxInt(p.Len(), 0)))
	scores := make([]float32, len(indices))
	p.TopKPreparedTo(indices, scores, pq)
	return indices, scores
}

// TopKPreparedTo writes top-k key positions and scores into caller-owned storage.
func (p *KVCachePage) TopKPreparedTo(indices []uint32, scores []float32, pq PreparedQuery) {
	if len(indices) != len(scores) {
		panic("(*KVCachePage).TopKPreparedTo: turboquant: index/score length mismatch")
	}
	if len(indices) == 0 {
		return
	}
	panicOnInvalid("(*KVCachePage).TopKPreparedTo", ValidatePreparedQuery(p.keyQ.Dim(), pq))

	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.length == 0 {
		for i := range indices {
			indices[i] = 0
			scores[i] = 0
		}
		return
	}
	k := len(indices)
	if k > p.length {
		panic(fmt.Sprintf("(*KVCachePage).TopKPreparedTo: turboquant: requested top-k %d exceeds page length %d", k, p.length))
	}
	if p.gpuKeys != nil && p.gpuKeys.Len() == p.length {
		if err := p.gpuKeys.ScorePreparedQueryTopKToTrusted(indices, scores, pq); err == nil {
			return
		}
	}
	topKPreparedCPULocked(indices, scores, p, pq)
}

// TopKPreparedBatch returns top-k key positions and scores for multiple prepared
// queries. Results are flattened query-major with k entries per query.
func (p *KVCachePage) TopKPreparedBatch(pqs []PreparedQuery, k int) ([]uint32, []float32) {
	if k <= 0 || len(pqs) == 0 {
		return nil, nil
	}
	indices := make([]uint32, len(pqs)*k)
	scores := make([]float32, len(indices))
	p.TopKPreparedBatchTo(indices, scores, pqs)
	return indices, scores
}

// TopKPreparedBatchTo writes top-k key positions and scores for multiple
// prepared queries into caller-owned storage. Results are flattened query-major
// with k entries per query.
func (p *KVCachePage) TopKPreparedBatchTo(indices []uint32, scores []float32, pqs []PreparedQuery) {
	if len(indices) != len(scores) {
		panic("(*KVCachePage).TopKPreparedBatchTo: turboquant: index/score length mismatch")
	}
	if len(pqs) == 0 || len(indices) == 0 {
		return
	}
	if len(indices)%len(pqs) != 0 {
		panic("(*KVCachePage).TopKPreparedBatchTo: turboquant: flattened top-k buffer must be divisible by query count")
	}
	k := len(indices) / len(pqs)
	if k == 0 {
		return
	}
	for i := range pqs {
		panicOnInvalid("(*KVCachePage).TopKPreparedBatchTo", ValidatePreparedQuery(p.keyQ.Dim(), pqs[i]))
	}

	p.mu.RLock()
	defer p.mu.RUnlock()
	if p.length == 0 {
		for i := range indices {
			indices[i] = 0
			scores[i] = 0
		}
		return
	}
	if k > p.length {
		panic(fmt.Sprintf("(*KVCachePage).TopKPreparedBatchTo: turboquant: requested top-k %d exceeds page length %d", k, p.length))
	}
	if p.gpuKeys != nil && p.gpuKeys.Len() == p.length {
		var err error
		if len(pqs) == 1 {
			err = p.gpuKeys.ScorePreparedQueryTopKToTrusted(indices[:k], scores[:k], pqs[0])
		} else {
			err = p.gpuKeys.ScorePreparedQueriesTopKToTrusted(indices, scores, pqs, k)
		}
		if err == nil {
			return
		}
	}
	topKPreparedBatchCPULocked(indices, scores, p, pqs, k)
}

// AttentionOutputPreparedTo computes an approximate attention output over the
// top-k matching keys and writes it into dst. It returns the selected token
// positions and normalized attention weights.
func (p *KVCachePage) AttentionOutputPreparedInto(dst []float32, indices []uint32, weights []float32, pq PreparedQuery) {
	if len(dst) != p.valueQ.Dim() {
		panic(fmt.Sprintf("(*KVCachePage).AttentionOutputPreparedInto: turboquant: expected destination length %d, got %d", p.valueQ.Dim(), len(dst)))
	}
	if len(indices) != len(weights) {
		panic("(*KVCachePage).AttentionOutputPreparedInto: turboquant: index/weight length mismatch")
	}
	if len(indices) == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	p.TopKPreparedTo(indices, weights, pq)
	softmaxInPlace(weights)
	p.weightedValueSumTo(dst, indices, weights)
}

// AttentionOutputPreparedBatchInto computes approximate attention outputs for
// multiple prepared queries into caller-owned storage. Results are flattened
// query-major with dim values per output and k values per query for indices and
// weights.
func (p *KVCachePage) AttentionOutputPreparedBatchInto(dst []float32, indices []uint32, weights []float32, pqs []PreparedQuery) {
	if len(pqs) == 0 {
		if len(dst) != 0 || len(indices) != 0 || len(weights) != 0 {
			panic("(*KVCachePage).AttentionOutputPreparedBatchInto: turboquant: empty query batch requires empty destination buffers")
		}
		return
	}
	if len(indices) != len(weights) {
		panic("(*KVCachePage).AttentionOutputPreparedBatchInto: turboquant: index/weight length mismatch")
	}
	if len(indices)%len(pqs) != 0 {
		panic("(*KVCachePage).AttentionOutputPreparedBatchInto: turboquant: flattened top-k buffer must be divisible by query count")
	}
	if len(dst) != len(pqs)*p.valueQ.Dim() {
		panic(fmt.Sprintf("(*KVCachePage).AttentionOutputPreparedBatchInto: turboquant: expected destination length %d, got %d", len(pqs)*p.valueQ.Dim(), len(dst)))
	}
	k := len(indices) / len(pqs)
	if k == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	p.TopKPreparedBatchTo(indices, weights, pqs)
	dim := p.valueQ.Dim()
	for queryIdx := range pqs {
		baseK := queryIdx * k
		softmaxInPlace(weights[baseK : baseK+k])
	}
	if p.tryGPUWeightedValueSumBatch(dst, indices, weights, len(pqs)) {
		return
	}
	for queryIdx := range pqs {
		baseK := queryIdx * k
		baseDim := queryIdx * dim
		p.weightedValueSumTo(dst[baseDim:baseDim+dim], indices[baseK:baseK+k], weights[baseK:baseK+k])
	}
}

// AttentionOutputPreparedBatchTo computes approximate attention outputs for a
// batch of prepared queries and writes them into dst. It returns flattened
// query-major top-k indices and normalized weights.
func (p *KVCachePage) AttentionOutputPreparedBatchTo(dst []float32, pqs []PreparedQuery, k int) ([]uint32, []float32) {
	if len(pqs) == 0 {
		if len(dst) != 0 {
			panic("(*KVCachePage).AttentionOutputPreparedBatchTo: turboquant: empty query batch requires empty destination")
		}
		return nil, nil
	}
	if k <= 0 {
		for i := range dst {
			dst[i] = 0
		}
		return nil, nil
	}
	indices := make([]uint32, len(pqs)*k)
	weights := make([]float32, len(indices))
	p.AttentionOutputPreparedBatchInto(dst, indices, weights, pqs)
	return indices, weights
}

// AttentionOutputPreparedTo computes an approximate attention output over the
// top-k matching keys and writes it into dst. It returns the selected token
// positions and normalized attention weights.
func (p *KVCachePage) AttentionOutputPreparedTo(dst []float32, pq PreparedQuery, k int) ([]uint32, []float32) {
	if len(dst) != p.valueQ.Dim() {
		panic(fmt.Sprintf("(*KVCachePage).AttentionOutputPreparedTo: turboquant: expected destination length %d, got %d", p.valueQ.Dim(), len(dst)))
	}
	if k <= 0 {
		for i := range dst {
			dst[i] = 0
		}
		return nil, nil
	}
	indices := make([]uint32, k)
	weights := make([]float32, k)
	p.AttentionOutputPreparedInto(dst, indices, weights, pq)
	return indices, weights
}

func (p *KVCachePage) weightedValueSumTo(dst []float32, indices []uint32, weights []float32) {
	if len(indices) != len(weights) {
		panic("turboquant: weighted value sum requires matching indices and weights")
	}
	for i := range dst {
		dst[i] = 0
	}
	if p.tryGPUWeightedValueSum(dst, indices, weights) {
		return
	}
	scratch := p.tmpPool.Get().(*kvValueScratch)
	tmp := scratch.tmp
	defer p.tmpPool.Put(scratch)
	p.mu.RLock()
	defer p.mu.RUnlock()
	for i, idx := range indices {
		slot := int(idx)
		if slot < 0 || slot >= p.length {
			panic(fmt.Sprintf("turboquant: value index %d out of bounds for page length %d", slot, p.length))
		}
		p.valueQ.DequantizeTo(tmp, p.valuePackedAt(slot))
		scale := weights[i] * p.valueNorms[slot]
		for j := range dst {
			dst[j] += scale * tmp[j]
		}
	}
}

func (p *KVCachePage) tryGPUWeightedValueSum(dst []float32, indices []uint32, weights []float32) bool {
	p.mu.RLock()
	backend := p.gpuValues
	p.mu.RUnlock()
	if backend == nil {
		return false
	}
	buf := p.valueQ.pool.Get().(*scratchBuf)
	defer p.valueQ.pool.Put(buf)
	if err := backend.accumulateRotatedTo(buf.rotated[:p.valueQ.dim], indices, weights); err != nil {
		return false
	}
	p.valueQ.rotation.applyInverse(dst, buf.rotated[:p.valueQ.dim], buf.work[:p.valueQ.dim])
	return true
}

func (p *KVCachePage) tryGPUWeightedValueSumBatch(dst []float32, indices []uint32, weights []float32, queryCount int) bool {
	p.mu.RLock()
	backend := p.gpuValues
	p.mu.RUnlock()
	if backend == nil || queryCount <= 0 {
		return false
	}
	dim := p.valueQ.dim
	scratch := p.batchPool.Get().(*kvBatchScratch)
	if cap(scratch.rotated) < len(dst) {
		scratch.rotated = make([]float32, len(dst))
	}
	if cap(scratch.work) < dim {
		scratch.work = make([]float32, dim)
	}
	rotated := scratch.rotated[:len(dst)]
	work := scratch.work[:dim]
	defer p.batchPool.Put(scratch)
	if err := backend.accumulateRotatedBatchTo(rotated, indices, weights, queryCount); err != nil {
		return false
	}
	for queryIdx := 0; queryIdx < queryCount; queryIdx++ {
		base := queryIdx * dim
		p.valueQ.rotation.applyInverse(dst[base:base+dim], rotated[base:base+dim], work)
	}
	return true
}

func (p *KVCachePage) keyAt(pos int) IPQuantized {
	mseBase := pos * p.keyMSEBytes
	signBase := pos * p.keySignBytes
	return IPQuantized{
		MSE:     p.keyMSE[mseBase : mseBase+p.keyMSEBytes],
		Signs:   p.keySigns[signBase : signBase+p.keySignBytes],
		ResNorm: p.keyResNorms[pos],
	}
}

func (p *KVCachePage) valuePackedAt(pos int) []byte {
	base := pos * p.valueBytes
	return p.valuePacked[base : base+p.valueBytes]
}

func (p *KVCachePage) disableGPUBackendsLocked() {
	if p.gpuKeys == nil {
		if p.gpuValues != nil {
			_ = p.gpuValues.Close()
			p.gpuValues = nil
		}
		return
	}
	if p.gpuValues != nil {
		_ = p.gpuValues.Close()
		p.gpuValues = nil
	}
	_ = p.gpuKeys.Close()
	p.gpuKeys = nil
}

func (p *KVCachePage) growLocked(want int) {
	if want <= cap(p.keyResNorms) {
		return
	}
	newCap := cap(p.keyResNorms) * 2
	if newCap < want {
		newCap = want
	}
	if newCap < 4 {
		newCap = 4
	}
	keyMSE := make([]byte, newCap*p.keyMSEBytes)
	copy(keyMSE, p.keyMSE[:p.length*p.keyMSEBytes])
	p.keyMSE = keyMSE

	keySigns := make([]byte, newCap*p.keySignBytes)
	copy(keySigns, p.keySigns[:p.length*p.keySignBytes])
	p.keySigns = keySigns

	keyResNorms := make([]float32, newCap)
	copy(keyResNorms, p.keyResNorms[:p.length])
	p.keyResNorms = keyResNorms

	valuePacked := make([]byte, newCap*p.valueBytes)
	copy(valuePacked, p.valuePacked[:p.length*p.valueBytes])
	p.valuePacked = valuePacked

	valueNorms := make([]float32, newCap)
	copy(valueNorms, p.valueNorms[:p.length])
	p.valueNorms = valueNorms
}

func (p *KVCachePage) packGPUKeyDataLocked() GPUPreparedData {
	data := GPUPreparedData{
		MSE:           make([]byte, p.length*p.keyMSEBytes),
		Signs:         make([]byte, p.length*p.keySignBytes),
		ResNorms:      make([]float32, p.length),
		TieBreakRanks: make([]uint32, p.length),
	}
	copy(data.MSE, p.keyMSE[:p.length*p.keyMSEBytes])
	copy(data.Signs, p.keySigns[:p.length*p.keySignBytes])
	copy(data.ResNorms, p.keyResNorms[:p.length])
	for i := 0; i < p.length; i++ {
		data.TieBreakRanks[i] = uint32(i)
	}
	return data
}

func topKPreparedCPULocked(indices []uint32, scores []float32, p *KVCachePage, pq PreparedQuery) {
	for i := range indices {
		indices[i] = 0
		scores[i] = 0
	}
	filled := 0
	for slot := 0; slot < p.length; slot++ {
		score := p.keyQ.InnerProductPreparedTrusted(p.keyAt(slot), pq)
		candidateIdx := uint32(slot)
		insert := filled
		for pos := 0; pos < filled; pos++ {
			if score > scores[pos] || (score == scores[pos] && candidateIdx < indices[pos]) {
				insert = pos
				break
			}
		}
		if filled < len(indices) {
			filled++
		} else if insert == filled {
			continue
		}
		limit := filled - 1
		if limit >= len(indices) {
			limit = len(indices) - 1
		}
		for pos := limit; pos > insert; pos-- {
			indices[pos] = indices[pos-1]
			scores[pos] = scores[pos-1]
		}
		indices[insert] = candidateIdx
		scores[insert] = score
	}
}

func topKPreparedBatchCPULocked(indices []uint32, scores []float32, p *KVCachePage, pqs []PreparedQuery, k int) {
	for i := range indices {
		indices[i] = 0
		scores[i] = 0
	}
	scratch := p.batchPool.Get().(*kvBatchScratch)
	if cap(scratch.scores) < len(pqs) {
		scratch.scores = make([]float32, len(pqs))
	}
	queryScores := scratch.scores[:len(pqs)]
	if cap(scratch.filled) < len(pqs) {
		scratch.filled = make([]int, len(pqs))
	}
	filled := scratch.filled[:len(pqs)]
	for i := range filled {
		filled[i] = 0
	}
	defer p.batchPool.Put(scratch)

	for slot := 0; slot < p.length; slot++ {
		p.keyQ.InnerProductPreparedBatchToTrusted(queryScores, p.keyAt(slot), pqs)
		candidateIdx := uint32(slot)
		for queryIdx := range pqs {
			score := queryScores[queryIdx]
			base := queryIdx * k
			insert := filled[queryIdx]
			for pos := 0; pos < filled[queryIdx]; pos++ {
				offset := base + pos
				if score > scores[offset] || (score == scores[offset] && candidateIdx < indices[offset]) {
					insert = pos
					break
				}
			}
			if filled[queryIdx] < k {
				filled[queryIdx]++
			} else if insert == filled[queryIdx] {
				continue
			}
			limit := filled[queryIdx] - 1
			if limit >= k {
				limit = k - 1
			}
			for pos := limit; pos > insert; pos-- {
				dst := base + pos
				src := dst - 1
				indices[dst] = indices[src]
				scores[dst] = scores[src]
			}
			offset := base + insert
			indices[offset] = candidateIdx
			scores[offset] = score
		}
	}
}

func softmaxInPlace(scores []float32) []float32 {
	if len(scores) == 0 {
		return scores
	}
	maxScore := scores[0]
	for _, score := range scores[1:] {
		if score > maxScore {
			maxScore = score
		}
	}
	var sum float64
	for i, score := range scores {
		weight := math.Exp(float64(score - maxScore))
		scores[i] = float32(weight)
		sum += weight
	}
	if sum == 0 {
		return scores
	}
	inv := float32(1 / sum)
	for i := range scores {
		scores[i] *= inv
	}
	return scores
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
