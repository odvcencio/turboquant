package turboquant

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"sync"
)

// TransformerLayerKVCache stores quantized transformer K/V tensors for one
// attention layer. Each appended token contributes one key and one value vector
// per head, flattened head-major as [heads*headDim].
//
// This is the lowest-level honest runtime integration slice for model KV in
// TurboQuant today: real transformer-shaped per-head K/V storage plus
// approximate attention output over quantized keys and values.
type TransformerLayerKVCache struct {
	mu sync.RWMutex

	heads   int
	headDim int
	keyBits int
	valBits int
	pages   []*KVCachePage
}

// TransformerLayerPreparedQuery holds reusable prepared-query state for one
// transformer attention layer, with one prepared query per head.
type TransformerLayerPreparedQuery struct {
	heads []PreparedQuery
}

// NewTransformerLayerKVCache creates a transformer-layer KV cache with a random
// seed.
func NewTransformerLayerKVCache(heads, headDim, keyBits, valueBits, capacity int) *TransformerLayerKVCache {
	var seedBytes [8]byte
	_, _ = rand.Read(seedBytes[:])
	seed := int64(binary.LittleEndian.Uint64(seedBytes[:]))
	return NewTransformerLayerKVCacheWithSeed(heads, headDim, keyBits, valueBits, capacity, seed)
}

// NewTransformerLayerKVCacheWithSeed creates a deterministic transformer-layer
// KV cache. Keys and values are quantized independently per head using the
// existing KVCachePage substrate.
func NewTransformerLayerKVCacheWithSeed(heads, headDim, keyBits, valueBits, capacity int, seed int64) *TransformerLayerKVCache {
	if heads <= 0 {
		panic("turboquant: transformer KV cache heads must be > 0")
	}
	panicOnInvalid("turboquant.NewTransformerLayerKVCacheWithSeed", validateDim(headDim))
	panicOnInvalid("turboquant.NewTransformerLayerKVCacheWithSeed", validateIPBitWidth(keyBits))
	panicOnInvalid("turboquant.NewTransformerLayerKVCacheWithSeed", validateBitWidth(valueBits))
	if capacity < 0 {
		panic("turboquant: transformer KV cache capacity must be >= 0")
	}

	pages := make([]*KVCachePage, heads)
	for head := range pages {
		headSeed := seed ^ (int64(head+1) * 1315423911)
		pages[head] = NewKVCachePageWithSeed(headDim, keyBits, headDim, valueBits, capacity, headSeed)
	}
	return &TransformerLayerKVCache{
		heads:   heads,
		headDim: headDim,
		keyBits: keyBits,
		valBits: valueBits,
		pages:   pages,
	}
}

// Heads reports the number of attention heads in this layer cache.
func (c *TransformerLayerKVCache) Heads() int {
	if c == nil {
		return 0
	}
	return c.heads
}

// HeadDim reports the per-head key/value dimension.
func (c *TransformerLayerKVCache) HeadDim() int {
	if c == nil {
		return 0
	}
	return c.headDim
}

// KeyBits reports the configured key quantization bit width for this layer.
func (c *TransformerLayerKVCache) KeyBits() int {
	if c == nil {
		return 0
	}
	return c.keyBits
}

// ValueBits reports the configured value quantization bit width for this layer.
func (c *TransformerLayerKVCache) ValueBits() int {
	if c == nil {
		return 0
	}
	return c.valBits
}

// Len reports the number of cached tokens per head.
func (c *TransformerLayerKVCache) Len() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if len(c.pages) == 0 {
		return 0
	}
	return c.pages[0].Len()
}

// Cap reports the token capacity per head.
func (c *TransformerLayerKVCache) Cap() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	if len(c.pages) == 0 {
		return 0
	}
	return c.pages[0].Cap()
}

// StorageBytes reports the aggregate allocated storage across all heads,
// excluding optional GPU-resident mirrors.
func (c *TransformerLayerKVCache) StorageBytes() uint64 {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	var total uint64
	for _, page := range c.pages {
		total += page.StorageBytes()
	}
	return total
}

// LiveBytes reports the aggregate populated storage across all heads, excluding
// optional GPU-resident mirrors.
func (c *TransformerLayerKVCache) LiveBytes() uint64 {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	var total uint64
	for _, page := range c.pages {
		total += page.LiveBytes()
	}
	return total
}

// Reset clears all cached tokens and releases any uploaded GPU state.
func (c *TransformerLayerKVCache) Reset() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, page := range c.pages {
		page.Reset()
	}
}

// GPUKeysEnabled reports whether all heads currently have their key corpus
// uploaded into the experimental GPU scorer.
func (c *TransformerLayerKVCache) GPUKeysEnabled() bool {
	if c == nil {
		return false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, page := range c.pages {
		if !page.GPUKeysEnabled() {
			return false
		}
	}
	return len(c.pages) != 0
}

// GPUValuesEnabled reports whether all heads currently have their value corpus
// uploaded into the experimental GPU accumulation backend.
func (c *TransformerLayerKVCache) GPUValuesEnabled() bool {
	if c == nil {
		return false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	for _, page := range c.pages {
		if !page.GPUValuesEnabled() {
			return false
		}
	}
	return len(c.pages) != 0
}

// EnableGPUKeys uploads every head into the experimental GPU backends.
func (c *TransformerLayerKVCache) EnableGPUKeys() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for i, page := range c.pages {
		if err := page.EnableGPUKeys(); err != nil {
			for j := 0; j < i; j++ {
				c.pages[j].DisableGPUKeys()
			}
			return err
		}
	}
	return nil
}

// DisableGPUKeys releases any uploaded GPU state for every head.
func (c *TransformerLayerKVCache) DisableGPUKeys() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, page := range c.pages {
		page.DisableGPUKeys()
	}
}

// Append quantizes and appends one transformer token worth of K/V tensors. Both
// keys and values must be flattened head-major as [heads*headDim].
func (c *TransformerLayerKVCache) Append(keys, values []float32) {
	if c == nil {
		panic("(*TransformerLayerKVCache).Append: turboquant: nil cache")
	}
	panicOnInvalid("(*TransformerLayerKVCache).Append", c.validateTokenVectors(keys, values))

	c.mu.Lock()
	defer c.mu.Unlock()
	for head, page := range c.pages {
		base := head * c.headDim
		page.Append(keys[base:base+c.headDim], values[base:base+c.headDim])
	}
}

// AllocPreparedQuery allocates reusable per-head prepared-query state for this
// layer cache.
func (c *TransformerLayerKVCache) AllocPreparedQuery() TransformerLayerPreparedQuery {
	if c == nil {
		return TransformerLayerPreparedQuery{}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	pq := TransformerLayerPreparedQuery{
		heads: make([]PreparedQuery, len(c.pages)),
	}
	for i, page := range c.pages {
		pq.heads[i] = page.keyQ.AllocPreparedQuery()
	}
	return pq
}

// PrepareQuery allocates and fills reusable per-head prepared-query state for a
// flattened head-major query tensor [heads*headDim].
func (c *TransformerLayerKVCache) PrepareQuery(query []float32) TransformerLayerPreparedQuery {
	pq := c.AllocPreparedQuery()
	c.PrepareQueryTo(&pq, query)
	return pq
}

// PrepareQueryTo writes reusable per-head prepared-query state into
// caller-owned storage.
func (c *TransformerLayerKVCache) PrepareQueryTo(dst *TransformerLayerPreparedQuery, query []float32) {
	if c == nil {
		panic("(*TransformerLayerKVCache).PrepareQueryTo: turboquant: nil cache")
	}
	if dst == nil {
		panic("(*TransformerLayerKVCache).PrepareQueryTo: turboquant: nil destination")
	}
	panicOnInvalid("(*TransformerLayerKVCache).PrepareQueryTo", c.validateQuery(query))

	c.mu.RLock()
	defer c.mu.RUnlock()
	panicOnInvalid("(*TransformerLayerKVCache).PrepareQueryTo", c.validatePreparedQuery(*dst))
	for head, page := range c.pages {
		base := head * c.headDim
		page.PrepareQueryTo(&dst.heads[head], query[base:base+c.headDim])
	}
}

// TopKPrepared returns flattened head-major top-k token positions and scores
// for a prepared query batch, with k entries per head.
func (c *TransformerLayerKVCache) TopKPrepared(pq TransformerLayerPreparedQuery, k int) ([]uint32, []float32) {
	if c == nil || k <= 0 {
		return nil, nil
	}
	k = minInt(k, c.Len())
	if k <= 0 {
		return nil, nil
	}
	indices := make([]uint32, c.heads*k)
	scores := make([]float32, len(indices))
	c.TopKPreparedTo(indices, scores, pq)
	return indices, scores
}

// TopKPreparedTo writes flattened head-major top-k token positions and scores
// into caller-owned storage, with k entries per head.
func (c *TransformerLayerKVCache) TopKPreparedTo(indices []uint32, scores []float32, pq TransformerLayerPreparedQuery) {
	if c == nil {
		panic("(*TransformerLayerKVCache).TopKPreparedTo: turboquant: nil cache")
	}
	if len(indices) != len(scores) {
		panic("(*TransformerLayerKVCache).TopKPreparedTo: turboquant: index/score length mismatch")
	}
	if len(indices) == 0 {
		return
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	panicOnInvalid("(*TransformerLayerKVCache).TopKPreparedTo", c.validatePreparedQuery(pq))
	if len(indices)%c.heads != 0 {
		panic("(*TransformerLayerKVCache).TopKPreparedTo: turboquant: flattened top-k buffer must be divisible by head count")
	}
	k := len(indices) / c.heads
	for head, page := range c.pages {
		base := head * k
		page.TopKPreparedTo(indices[base:base+k], scores[base:base+k], pq.heads[head])
	}
}

// AttentionOutputInto computes approximate per-head attention outputs from a
// flattened head-major query tensor [heads*headDim] and writes them into dst.
// Indices and weights are flattened head-major with k entries per head.
func (c *TransformerLayerKVCache) AttentionOutputInto(dst []float32, indices []uint32, weights []float32, query []float32) {
	pq := c.PrepareQuery(query)
	c.AttentionOutputPreparedInto(dst, indices, weights, pq)
}

// AttentionOutputTo computes approximate per-head attention outputs from a
// flattened head-major query tensor [heads*headDim], writing the result into
// dst and returning flattened head-major indices and weights.
func (c *TransformerLayerKVCache) AttentionOutputTo(dst []float32, query []float32, k int) ([]uint32, []float32) {
	pq := c.PrepareQuery(query)
	return c.AttentionOutputPreparedTo(dst, pq, k)
}

// AttentionOutputPreparedInto computes approximate per-head attention outputs
// for a prepared query batch and writes them into dst. Indices and weights are
// flattened head-major with k entries per head.
func (c *TransformerLayerKVCache) AttentionOutputPreparedInto(dst []float32, indices []uint32, weights []float32, pq TransformerLayerPreparedQuery) {
	if c == nil {
		panic("(*TransformerLayerKVCache).AttentionOutputPreparedInto: turboquant: nil cache")
	}
	if len(dst) != c.heads*c.headDim {
		panic(fmt.Sprintf("(*TransformerLayerKVCache).AttentionOutputPreparedInto: turboquant: expected destination length %d, got %d", c.heads*c.headDim, len(dst)))
	}
	if len(indices) != len(weights) {
		panic("(*TransformerLayerKVCache).AttentionOutputPreparedInto: turboquant: index/weight length mismatch")
	}
	if len(indices) == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	panicOnInvalid("(*TransformerLayerKVCache).AttentionOutputPreparedInto", c.validatePreparedQuery(pq))
	if len(indices)%c.heads != 0 {
		panic("(*TransformerLayerKVCache).AttentionOutputPreparedInto: turboquant: flattened top-k buffer must be divisible by head count")
	}
	k := len(indices) / c.heads
	for head, page := range c.pages {
		baseDim := head * c.headDim
		baseK := head * k
		page.AttentionOutputPreparedInto(
			dst[baseDim:baseDim+c.headDim],
			indices[baseK:baseK+k],
			weights[baseK:baseK+k],
			pq.heads[head],
		)
	}
}

// AttentionOutputPreparedTo computes approximate per-head attention outputs for
// a prepared query batch, writing the result into dst and returning flattened
// head-major indices and weights.
func (c *TransformerLayerKVCache) AttentionOutputPreparedTo(dst []float32, pq TransformerLayerPreparedQuery, k int) ([]uint32, []float32) {
	if c == nil {
		panic("(*TransformerLayerKVCache).AttentionOutputPreparedTo: turboquant: nil cache")
	}
	if len(dst) != c.heads*c.headDim {
		panic(fmt.Sprintf("(*TransformerLayerKVCache).AttentionOutputPreparedTo: turboquant: expected destination length %d, got %d", c.heads*c.headDim, len(dst)))
	}
	if k <= 0 {
		for i := range dst {
			dst[i] = 0
		}
		return nil, nil
	}
	k = minInt(k, c.Len())
	if k <= 0 {
		for i := range dst {
			dst[i] = 0
		}
		return nil, nil
	}
	indices := make([]uint32, c.heads*k)
	weights := make([]float32, len(indices))
	c.AttentionOutputPreparedInto(dst, indices, weights, pq)
	return indices, weights
}

func (c *TransformerLayerKVCache) validateTokenVectors(keys, values []float32) error {
	if err := c.validateQuery(keys); err != nil {
		return err
	}
	return c.validateQuery(values)
}

func (c *TransformerLayerKVCache) validateQuery(query []float32) error {
	if c == nil {
		return fmt.Errorf("turboquant: nil transformer KV cache")
	}
	return ValidateVector(c.heads*c.headDim, query)
}

func (c *TransformerLayerKVCache) validatePreparedQuery(pq TransformerLayerPreparedQuery) error {
	if c == nil {
		return fmt.Errorf("turboquant: nil transformer KV cache")
	}
	if len(pq.heads) != c.heads {
		return fmt.Errorf("turboquant: expected %d prepared heads, got %d", c.heads, len(pq.heads))
	}
	for head := range pq.heads {
		if err := ValidatePreparedQuery(c.headDim, pq.heads[head]); err != nil {
			return err
		}
	}
	return nil
}
