package turboquant

import (
	"errors"
	"math/rand"
	"testing"
)

func TestTransformerLayerKVCacheAppendAndLen(t *testing.T) {
	cache := NewTransformerLayerKVCacheWithSeed(3, 16, 3, 2, 16, 42)
	if cache.Heads() != 3 {
		t.Fatalf("Heads = %d want 3", cache.Heads())
	}
	if cache.HeadDim() != 16 {
		t.Fatalf("HeadDim = %d want 16", cache.HeadDim())
	}
	if cache.KeyBits() != 3 || cache.ValueBits() != 2 {
		t.Fatalf("bit widths = (%d,%d) want (3,2)", cache.KeyBits(), cache.ValueBits())
	}
	if cache.Len() != 0 {
		t.Fatalf("Len = %d want 0", cache.Len())
	}

	rng := newTestRNG()
	for i := 0; i < 8; i++ {
		cache.Append(randomMultiHeadUnitVector(3, 16, rng), randomMultiHeadUnitVector(3, 16, rng))
	}
	if cache.Len() != 8 {
		t.Fatalf("Len = %d want 8", cache.Len())
	}
	if cache.StorageBytes() <= cache.LiveBytes() {
		t.Fatalf("storage bytes = %d live bytes = %d", cache.StorageBytes(), cache.LiveBytes())
	}
}

func TestTransformerModelKVCacheProfiles(t *testing.T) {
	stack := NewTransformerModelKVCache([]TransformerLayerKVProfile{
		{Layer: 0, Heads: 2, HeadDim: 16, KeyBits: 2, ValueBits: 2, Capacity: 8, Seed: 11},
		{Layer: 15, Heads: 2, HeadDim: 16, KeyBits: 4, ValueBits: 4, Capacity: 8, Seed: 29},
	})
	if stack.Layers() != 2 {
		t.Fatalf("Layers = %d want 2", stack.Layers())
	}
	if !stack.HasLayer(0) || !stack.HasLayer(15) || stack.HasLayer(7) {
		t.Fatalf("HasLayer mismatch for configured stack")
	}
	if got := stack.LayerIDs(); len(got) != 2 || got[0] != 0 || got[1] != 15 {
		t.Fatalf("LayerIDs = %v want [0 15]", got)
	}
	layer0 := stack.Layer(0)
	layer15 := stack.Layer(15)
	if layer0 == nil || layer15 == nil {
		t.Fatal("Layer() returned nil for configured layers")
	}
	if layer0.KeyBits() != 2 || layer0.ValueBits() != 2 {
		t.Fatalf("layer0 bit widths = (%d,%d) want (2,2)", layer0.KeyBits(), layer0.ValueBits())
	}
	if layer15.KeyBits() != 4 || layer15.ValueBits() != 4 {
		t.Fatalf("layer15 bit widths = (%d,%d) want (4,4)", layer15.KeyBits(), layer15.ValueBits())
	}
	profile15, ok := stack.LayerProfile(15)
	if !ok {
		t.Fatal("LayerProfile(15) = false want true")
	}
	if profile15.KeyBits != 4 || profile15.ValueBits != 4 {
		t.Fatalf("LayerProfile(15) = (%d,%d) want (4,4)", profile15.KeyBits, profile15.ValueBits)
	}

	rng := newTestRNG()
	stack.Append(0, randomMultiHeadUnitVector(2, 16, rng), randomMultiHeadUnitVector(2, 16, rng))
	stack.Append(15, randomMultiHeadUnitVector(2, 16, rng), randomMultiHeadUnitVector(2, 16, rng))
	if stack.LiveBytes() != layer0.LiveBytes()+layer15.LiveBytes() {
		t.Fatalf("LiveBytes = %d want %d", stack.LiveBytes(), layer0.LiveBytes()+layer15.LiveBytes())
	}
	if stack.StorageBytes() != layer0.StorageBytes()+layer15.StorageBytes() {
		t.Fatalf("StorageBytes = %d want %d", stack.StorageBytes(), layer0.StorageBytes()+layer15.StorageBytes())
	}

	stack.Reset()
	if layer0.Len() != 0 || layer15.Len() != 0 {
		t.Fatalf("lens after reset = (%d,%d) want (0,0)", layer0.Len(), layer15.Len())
	}
}

func TestTransformerModelKVCacheProfilesGroupedKVHeads(t *testing.T) {
	stack := NewTransformerModelKVCache([]TransformerLayerKVProfile{
		{Layer: 9, Heads: 4, KVHeads: 2, HeadDim: 16, KeyBits: 3, ValueBits: 2, Capacity: 8, Seed: 11},
	})
	layer := stack.Layer(9)
	if layer == nil {
		t.Fatal("Layer(9) returned nil")
	}
	if layer.Heads() != 2 {
		t.Fatalf("Layer(9).Heads = %d want 2", layer.Heads())
	}
	profile, ok := stack.LayerProfile(9)
	if !ok {
		t.Fatal("LayerProfile(9) = false want true")
	}
	if profile.KVHeadCount() != 2 {
		t.Fatalf("profile.KVHeadCount = %d want 2", profile.KVHeadCount())
	}
	if profile.QueryHeadsPerKVHead() != 2 {
		t.Fatalf("profile.QueryHeadsPerKVHead = %d want 2", profile.QueryHeadsPerKVHead())
	}

	rng := newTestRNG()
	stack.Append(9, randomMultiHeadUnitVector(2, 16, rng), randomMultiHeadUnitVector(2, 16, rng))
	if layer.Len() != 1 {
		t.Fatalf("layer.Len = %d want 1", layer.Len())
	}
}

func TestTransformerLayerKVCachePrepareQueryMatchesPerHeadPages(t *testing.T) {
	cache := NewTransformerLayerKVCacheWithSeed(2, 16, 3, 2, 8, 42)
	rng := newTestRNG()
	query := randomMultiHeadUnitVector(2, 16, rng)

	got := cache.PrepareQuery(query)
	if len(got.heads) != 2 {
		t.Fatalf("prepared heads = %d want 2", len(got.heads))
	}
	for head := range cache.pages {
		base := head * cache.HeadDim()
		want := cache.pages[head].PrepareQuery(query[base : base+cache.HeadDim()])
		if err := ValidatePreparedQuery(cache.HeadDim(), got.heads[head]); err != nil {
			t.Fatalf("ValidatePreparedQuery(head=%d): %v", head, err)
		}
		if !samePreparedQuery(got.heads[head], want) {
			t.Fatalf("prepared query mismatch for head %d", head)
		}
	}
}

func TestTransformerLayerKVCacheAttentionOutputPreparedMatchesPerHeadPages(t *testing.T) {
	cache := NewTransformerLayerKVCacheWithSeed(2, 16, 3, 2, 64, 42)
	rng := newTestRNG()
	for i := 0; i < 32; i++ {
		cache.Append(randomMultiHeadUnitVector(2, 16, rng), randomMultiHeadUnitVector(2, 16, rng))
	}
	pq := cache.PrepareQuery(randomMultiHeadUnitVector(2, 16, rng))

	gotOut := make([]float32, cache.Heads()*cache.HeadDim())
	gotIdx := make([]uint32, cache.Heads()*4)
	gotWeights := make([]float32, len(gotIdx))
	cache.AttentionOutputPreparedInto(gotOut, gotIdx, gotWeights, pq)

	for head, page := range cache.pages {
		wantOut := make([]float32, cache.HeadDim())
		wantIdx := make([]uint32, 4)
		wantWeights := make([]float32, 4)
		page.AttentionOutputPreparedInto(wantOut, wantIdx, wantWeights, pq.heads[head])

		baseDim := head * cache.HeadDim()
		baseK := head * 4
		for i := 0; i < 4; i++ {
			if gotIdx[baseK+i] != wantIdx[i] {
				t.Fatalf("head %d index[%d] = %d want %d", head, i, gotIdx[baseK+i], wantIdx[i])
			}
			if !closeKVFloat32(gotWeights[baseK+i], wantWeights[i], 1e-6) {
				t.Fatalf("head %d weight[%d] = %v want %v", head, i, gotWeights[baseK+i], wantWeights[i])
			}
		}
		for i := 0; i < cache.HeadDim(); i++ {
			if !closeKVFloat32(gotOut[baseDim+i], wantOut[i], 1e-6) {
				t.Fatalf("head %d output[%d] = %v want %v", head, i, gotOut[baseDim+i], wantOut[i])
			}
		}
	}
}

func TestTransformerLayerKVCacheAttentionOutputPreparedIntoZeroAllocs(t *testing.T) {
	skipAllocsUnderRace(t)
	cache := NewTransformerLayerKVCacheWithSeed(4, 16, 3, 2, 64, 42)
	rng := newTestRNG()
	for i := 0; i < 64; i++ {
		cache.Append(randomMultiHeadUnitVector(4, 16, rng), randomMultiHeadUnitVector(4, 16, rng))
	}
	pq := cache.AllocPreparedQuery()
	cache.PrepareQueryTo(&pq, randomMultiHeadUnitVector(4, 16, rng))

	dst := make([]float32, cache.Heads()*cache.HeadDim())
	indices := make([]uint32, cache.Heads()*8)
	weights := make([]float32, len(indices))
	allocs := testing.AllocsPerRun(100, func() {
		cache.AttentionOutputPreparedInto(dst, indices, weights, pq)
	})
	if allocs != 0 {
		t.Fatalf("allocs = %v want 0", allocs)
	}
}

func TestTransformerLayerKVCacheGPUParityWhenAvailable(t *testing.T) {
	cache := NewTransformerLayerKVCacheWithSeed(2, 16, 3, 2, 64, 42)
	rng := newTestRNG()
	for i := 0; i < 64; i++ {
		cache.Append(randomMultiHeadUnitVector(2, 16, rng), randomMultiHeadUnitVector(2, 16, rng))
	}
	pq := cache.PrepareQuery(randomMultiHeadUnitVector(2, 16, rng))

	cpuOut := make([]float32, cache.Heads()*cache.HeadDim())
	cpuIndices, cpuWeights := cache.AttentionOutputPreparedTo(cpuOut, pq, 6)

	err := cache.EnableGPUKeys()
	if err != nil {
		if errors.Is(err, ErrGPUBackendUnavailable) {
			t.Skip("GPU transformer KV unavailable on this platform")
		}
		t.Fatalf("EnableGPUKeys: %v", err)
	}
	if !cache.GPUKeysEnabled() {
		t.Fatal("GPUKeysEnabled = false want true")
	}
	if !cache.GPUValuesEnabled() {
		t.Fatal("GPUValuesEnabled = false want true")
	}

	gpuOut := make([]float32, cache.Heads()*cache.HeadDim())
	gpuIndices, gpuWeights := cache.AttentionOutputPreparedTo(gpuOut, pq, 6)
	for i := range cpuIndices {
		if gpuIndices[i] != cpuIndices[i] {
			t.Fatalf("gpu index[%d] = %d want %d", i, gpuIndices[i], cpuIndices[i])
		}
		if !closeKVFloat32(gpuWeights[i], cpuWeights[i], 1e-5) {
			t.Fatalf("gpu weight[%d] = %v want %v", i, gpuWeights[i], cpuWeights[i])
		}
	}
	for i := range cpuOut {
		if !closeKVFloat32(gpuOut[i], cpuOut[i], 1e-5) {
			t.Fatalf("gpu output[%d] = %v want %v", i, gpuOut[i], cpuOut[i])
		}
	}

	cache.Append(randomMultiHeadUnitVector(2, 16, rng), randomMultiHeadUnitVector(2, 16, rng))
	if cache.GPUKeysEnabled() {
		t.Fatal("GPU keys should be invalidated after append")
	}
	if cache.GPUValuesEnabled() {
		t.Fatal("GPU values should be invalidated after append")
	}
}

func randomMultiHeadUnitVector(heads, headDim int, rng *rand.Rand) []float32 {
	flat := make([]float32, heads*headDim)
	for head := 0; head < heads; head++ {
		base := head * headDim
		copy(flat[base:base+headDim], randomUnitVector(headDim, rng))
	}
	return flat
}

func samePreparedQuery(left, right PreparedQuery) bool {
	if len(left.signLUT) != len(right.signLUT) || len(left.mseLUT) != len(right.mseLUT) || len(left.rotY) != len(right.rotY) || left.mseBitWidth != right.mseBitWidth {
		return false
	}
	for i := range left.signLUT {
		if left.signLUT[i] != right.signLUT[i] {
			return false
		}
	}
	for i := range left.mseLUT {
		if left.mseLUT[i] != right.mseLUT[i] {
			return false
		}
	}
	for i := range left.rotY {
		if left.rotY[i] != right.rotY[i] {
			return false
		}
	}
	return true
}
