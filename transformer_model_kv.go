package turboquant

import "fmt"

// TransformerLayerKVProfile describes the storage shape and quantization policy
// for one transformer attention layer in a multi-layer KV stack.
type TransformerLayerKVProfile struct {
	Layer     int   `json:"layer"`
	Heads     int   `json:"heads"`
	KVHeads   int   `json:"kv_heads,omitempty"`
	HeadDim   int   `json:"head_dim"`
	KeyBits   int   `json:"key_bits"`
	ValueBits int   `json:"value_bits"`
	Capacity  int   `json:"capacity,omitempty"`
	Seed      int64 `json:"seed,omitempty"`
}

// KVHeadCount reports the number of KV heads used for storage. When omitted in
// JSON, it defaults to the full query-head count for MHA models.
func (p TransformerLayerKVProfile) KVHeadCount() int {
	if p.KVHeads > 0 {
		return p.KVHeads
	}
	return p.Heads
}

// QueryHeadsPerKVHead reports how many query heads share one KV head.
func (p TransformerLayerKVProfile) QueryHeadsPerKVHead() int {
	kvHeads := p.KVHeadCount()
	if kvHeads <= 0 {
		return 0
	}
	return p.Heads / kvHeads
}

// Validate checks whether the profile can be used to construct a layer cache.
func (p TransformerLayerKVProfile) Validate() error {
	if p.Layer < 0 {
		return fmt.Errorf("turboquant: transformer layer index must be >= 0")
	}
	if p.Heads <= 0 {
		return fmt.Errorf("turboquant: transformer layer profile heads must be > 0")
	}
	if err := validateDim(p.HeadDim); err != nil {
		return err
	}
	if err := validateIPBitWidth(p.KeyBits); err != nil {
		return err
	}
	if err := validateBitWidth(p.ValueBits); err != nil {
		return err
	}
	kvHeads := p.KVHeadCount()
	if kvHeads <= 0 {
		return fmt.Errorf("turboquant: transformer layer profile kv_heads must be > 0")
	}
	if p.Heads%kvHeads != 0 {
		return fmt.Errorf("turboquant: transformer layer profile heads %d must be divisible by kv_heads %d", p.Heads, kvHeads)
	}
	if p.Capacity < 0 {
		return fmt.Errorf("turboquant: transformer layer profile capacity must be >= 0")
	}
	return nil
}

// TransformerModelKVCache groups one TransformerLayerKVCache per layer, allowing
// each layer to use its own bit allocation policy.
type TransformerModelKVCache struct {
	profiles     []TransformerLayerKVProfile
	layers       []*TransformerLayerKVCache
	indexByLayer map[int]int
}

// NewTransformerModelKVCache constructs a multi-layer transformer KV stack from
// explicit layer profiles. Duplicate layer ids and invalid profiles panic.
func NewTransformerModelKVCache(profiles []TransformerLayerKVProfile) *TransformerModelKVCache {
	if len(profiles) == 0 {
		panic("turboquant: transformer model KV cache requires at least one layer profile")
	}
	layerProfiles := make([]TransformerLayerKVProfile, len(profiles))
	copy(layerProfiles, profiles)

	stack := &TransformerModelKVCache{
		profiles:     layerProfiles,
		layers:       make([]*TransformerLayerKVCache, len(layerProfiles)),
		indexByLayer: make(map[int]int, len(layerProfiles)),
	}
	for i, profile := range layerProfiles {
		panicOnInvalid("turboquant.NewTransformerModelKVCache", profile.Validate())
		if _, exists := stack.indexByLayer[profile.Layer]; exists {
			panic(fmt.Sprintf("turboquant: duplicate transformer layer profile %d", profile.Layer))
		}
		stack.indexByLayer[profile.Layer] = i
		stack.layers[i] = NewTransformerLayerKVCacheWithSeed(
			profile.KVHeadCount(),
			profile.HeadDim,
			profile.KeyBits,
			profile.ValueBits,
			profile.Capacity,
			profile.Seed,
		)
	}
	return stack
}

// Layers reports the number of configured transformer layers.
func (c *TransformerModelKVCache) Layers() int {
	if c == nil {
		return 0
	}
	return len(c.layers)
}

// LayerIDs reports the configured layer ids in profile order.
func (c *TransformerModelKVCache) LayerIDs() []int {
	if c == nil {
		return nil
	}
	ids := make([]int, len(c.profiles))
	for i, profile := range c.profiles {
		ids[i] = profile.Layer
	}
	return ids
}

// Profiles returns a copy of the configured layer profiles.
func (c *TransformerModelKVCache) Profiles() []TransformerLayerKVProfile {
	if c == nil {
		return nil
	}
	out := make([]TransformerLayerKVProfile, len(c.profiles))
	copy(out, c.profiles)
	return out
}

// HasLayer reports whether the stack contains the requested layer id.
func (c *TransformerModelKVCache) HasLayer(layer int) bool {
	if c == nil {
		return false
	}
	_, ok := c.indexByLayer[layer]
	return ok
}

// LayerProfile returns the configured profile for one layer.
func (c *TransformerModelKVCache) LayerProfile(layer int) (TransformerLayerKVProfile, bool) {
	if c == nil {
		return TransformerLayerKVProfile{}, false
	}
	idx, ok := c.indexByLayer[layer]
	if !ok {
		return TransformerLayerKVProfile{}, false
	}
	return c.profiles[idx], true
}

// Layer returns the per-layer cache for the requested layer id, or nil when the
// layer is not configured.
func (c *TransformerModelKVCache) Layer(layer int) *TransformerLayerKVCache {
	if c == nil {
		return nil
	}
	idx, ok := c.indexByLayer[layer]
	if !ok {
		return nil
	}
	return c.layers[idx]
}

// Append writes one token's K/V tensors into the requested layer cache.
func (c *TransformerModelKVCache) Append(layer int, keys, values []float32) {
	cache := c.Layer(layer)
	if cache == nil {
		panic(fmt.Sprintf("(*TransformerModelKVCache).Append: turboquant: unknown layer %d", layer))
	}
	cache.Append(keys, values)
}

// Reset clears all configured layer caches.
func (c *TransformerModelKVCache) Reset() {
	if c == nil {
		return
	}
	for _, layer := range c.layers {
		layer.Reset()
	}
}

// StorageBytes reports the aggregate allocated storage across all configured
// transformer layers.
func (c *TransformerModelKVCache) StorageBytes() uint64 {
	if c == nil {
		return 0
	}
	var total uint64
	for _, layer := range c.layers {
		total += layer.StorageBytes()
	}
	return total
}

// LiveBytes reports the aggregate populated storage across all configured
// transformer layers.
func (c *TransformerModelKVCache) LiveBytes() uint64 {
	if c == nil {
		return 0
	}
	var total uint64
	for _, layer := range c.layers {
		total += layer.LiveBytes()
	}
	return total
}
