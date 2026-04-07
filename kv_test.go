package turboquant

import (
	"errors"
	"math"
	"testing"
)

func TestKVCachePageAttentionOutputPrepared(t *testing.T) {
	page := NewKVCachePageWithSeed(32, 3, 32, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 12; i++ {
		page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	}
	pq := page.PrepareQuery(randomUnitVector(32, rng))

	got := make([]float32, 32)
	indices, weights := page.AttentionOutputPreparedTo(got, pq, 4)
	if len(indices) != 4 || len(weights) != 4 {
		t.Fatalf("got %d indices and %d weights want 4/4", len(indices), len(weights))
	}
	var sum float64
	for i := range weights {
		sum += float64(weights[i])
	}
	if math.Abs(sum-1) > 1e-5 {
		t.Fatalf("attention weights sum = %v want 1", sum)
	}

	manualIndices, manualScores := page.TopKPrepared(pq, 4)
	for i := range indices {
		if indices[i] != manualIndices[i] {
			t.Fatalf("index[%d] = %d want %d", i, indices[i], manualIndices[i])
		}
	}
	manualWeights := append([]float32(nil), manualScores...)
	softmaxInPlace(manualWeights)
	for i := range weights {
		if !closeKVFloat32(weights[i], manualWeights[i], 1e-6) {
			t.Fatalf("weight[%d] = %v want %v", i, weights[i], manualWeights[i])
		}
	}

	want := make([]float32, 32)
	tmp := make([]float32, 32)
	for i, idx := range manualIndices {
		page.valueQ.DequantizeTo(tmp, page.valuePackedAt(int(idx)))
		scale := manualWeights[i] * page.valueNorms[idx]
		for j := range want {
			want[j] += scale * tmp[j]
		}
	}
	for i := range want {
		if !closeKVFloat32(got[i], want[i], 1e-6) {
			t.Fatalf("output[%d] = %v want %v", i, got[i], want[i])
		}
	}
}

func TestKVCachePageAttentionOutputPreparedIntoMatchesAllocatingAPI(t *testing.T) {
	page := NewKVCachePageWithSeed(32, 3, 32, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 32; i++ {
		page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	}
	pq := page.PrepareQuery(randomUnitVector(32, rng))

	wantOut := make([]float32, 32)
	wantIdx, wantWeights := page.AttentionOutputPreparedTo(wantOut, pq, 6)

	gotOut := make([]float32, 32)
	gotIdx := make([]uint32, 6)
	gotWeights := make([]float32, 6)
	page.AttentionOutputPreparedInto(gotOut, gotIdx, gotWeights, pq)

	for i := range wantIdx {
		if gotIdx[i] != wantIdx[i] {
			t.Fatalf("index[%d] = %d want %d", i, gotIdx[i], wantIdx[i])
		}
		if !closeKVFloat32(gotWeights[i], wantWeights[i], 1e-6) {
			t.Fatalf("weight[%d] = %v want %v", i, gotWeights[i], wantWeights[i])
		}
	}
	for i := range wantOut {
		if !closeKVFloat32(gotOut[i], wantOut[i], 1e-6) {
			t.Fatalf("output[%d] = %v want %v", i, gotOut[i], wantOut[i])
		}
	}
}

func TestKVCachePageAttentionOutputPreparedIntoZeroAllocs(t *testing.T) {
	page := NewKVCachePageWithSeed(32, 3, 32, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 64; i++ {
		page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	}
	pq := page.PrepareQuery(randomUnitVector(32, rng))
	dst := make([]float32, 32)
	indices := make([]uint32, 8)
	weights := make([]float32, 8)
	allocs := testing.AllocsPerRun(100, func() {
		page.AttentionOutputPreparedInto(dst, indices, weights, pq)
	})
	if allocs != 0 {
		t.Fatalf("allocs = %v want 0", allocs)
	}
}

func TestKVCachePageAttentionOutputPreparedBatchIntoMatchesSingleQueryAPI(t *testing.T) {
	page := NewKVCachePageWithSeed(32, 3, 32, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 64; i++ {
		page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	}
	pqs := make([]PreparedQuery, 4)
	for i := range pqs {
		pqs[i] = page.PrepareQuery(randomUnitVector(32, rng))
	}
	gotOut := make([]float32, len(pqs)*32)
	gotIdx := make([]uint32, len(pqs)*6)
	gotWeights := make([]float32, len(gotIdx))
	page.AttentionOutputPreparedBatchInto(gotOut, gotIdx, gotWeights, pqs)

	for i := range pqs {
		wantOut := make([]float32, 32)
		wantIdx := make([]uint32, 6)
		wantWeights := make([]float32, 6)
		page.AttentionOutputPreparedInto(wantOut, wantIdx, wantWeights, pqs[i])
		baseK := i * 6
		baseDim := i * 32
		for j := 0; j < 6; j++ {
			if gotIdx[baseK+j] != wantIdx[j] {
				t.Fatalf("query %d index[%d] = %d want %d", i, j, gotIdx[baseK+j], wantIdx[j])
			}
			if !closeKVFloat32(gotWeights[baseK+j], wantWeights[j], 1e-6) {
				t.Fatalf("query %d weight[%d] = %v want %v", i, j, gotWeights[baseK+j], wantWeights[j])
			}
		}
		for j := 0; j < 32; j++ {
			if !closeKVFloat32(gotOut[baseDim+j], wantOut[j], 1e-6) {
				t.Fatalf("query %d output[%d] = %v want %v", i, j, gotOut[baseDim+j], wantOut[j])
			}
		}
	}
}

func TestKVCachePageAttentionOutputPreparedBatchIntoZeroAllocs(t *testing.T) {
	page := NewKVCachePageWithSeed(32, 3, 32, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 64; i++ {
		page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	}
	pqs := make([]PreparedQuery, 4)
	for i := range pqs {
		pqs[i] = page.PrepareQuery(randomUnitVector(32, rng))
	}
	dst := make([]float32, len(pqs)*32)
	indices := make([]uint32, len(pqs)*8)
	weights := make([]float32, len(indices))
	allocs := testing.AllocsPerRun(100, func() {
		page.AttentionOutputPreparedBatchInto(dst, indices, weights, pqs)
	})
	if allocs != 0 {
		t.Fatalf("allocs = %v want 0", allocs)
	}
}

func TestKVCachePageGPUParityWhenAvailable(t *testing.T) {
	page := NewKVCachePageWithSeed(32, 3, 32, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 64; i++ {
		page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	}
	pq := page.PrepareQuery(randomUnitVector(32, rng))

	cpuOut := make([]float32, 32)
	cpuIndices, cpuWeights := page.AttentionOutputPreparedTo(cpuOut, pq, 6)

	err := page.EnableGPUKeys()
	if err != nil {
		if errors.Is(err, ErrGPUBackendUnavailable) {
			t.Skip("GPU KV keys unavailable on this platform")
		}
		t.Fatalf("EnableGPUKeys: %v", err)
	}
	if !page.GPUKeysEnabled() {
		t.Fatal("GPUKeysEnabled = false want true")
	}
	if page.GPUValuesEnabled() != page.GPUKeysEnabled() {
		t.Fatalf("GPUValuesEnabled = %v want %v", page.GPUValuesEnabled(), page.GPUKeysEnabled())
	}

	gpuOut := make([]float32, 32)
	gpuIndices, gpuWeights := page.AttentionOutputPreparedTo(gpuOut, pq, 6)
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

	page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	if page.GPUKeysEnabled() {
		t.Fatal("GPU keys should be invalidated after append")
	}
	if page.GPUValuesEnabled() {
		t.Fatal("GPU values should be invalidated after append")
	}
}

func TestKVCachePageGPUBatchParityWhenAvailable(t *testing.T) {
	page := NewKVCachePageWithSeed(32, 3, 32, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 64; i++ {
		page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	}
	pqs := make([]PreparedQuery, 4)
	for i := range pqs {
		pqs[i] = page.PrepareQuery(randomUnitVector(32, rng))
	}

	cpuOut := make([]float32, len(pqs)*32)
	cpuIndices := make([]uint32, len(pqs)*6)
	cpuWeights := make([]float32, len(cpuIndices))
	page.AttentionOutputPreparedBatchInto(cpuOut, cpuIndices, cpuWeights, pqs)

	err := page.EnableGPUKeys()
	if err != nil {
		if errors.Is(err, ErrGPUBackendUnavailable) {
			t.Skip("GPU KV keys unavailable on this platform")
		}
		t.Fatalf("EnableGPUKeys: %v", err)
	}

	gpuOut := make([]float32, len(pqs)*32)
	gpuIndices := make([]uint32, len(pqs)*6)
	gpuWeights := make([]float32, len(gpuIndices))
	page.AttentionOutputPreparedBatchInto(gpuOut, gpuIndices, gpuWeights, pqs)

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
}

func TestKVPreparedQueryBatchParityWhenAvailable(t *testing.T) {
	page := NewKVCachePageWithSeed(32, 3, 32, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 64; i++ {
		page.Append(randomUnitVector(32, rng), randomUnitVector(32, rng))
	}
	pqs := make([]PreparedQuery, 4)
	for i := range pqs {
		pqs[i] = page.PrepareQuery(randomUnitVector(32, rng))
	}

	wantOut := make([]float32, len(pqs)*32)
	wantIdx := make([]uint32, len(pqs)*6)
	wantWeights := make([]float32, len(wantIdx))
	page.AttentionOutputPreparedBatchInto(wantOut, wantIdx, wantWeights, pqs)

	err := page.EnableGPUKeys()
	if err != nil {
		if errors.Is(err, ErrGPUBackendUnavailable) {
			t.Skip("GPU KV keys unavailable on this platform")
		}
		t.Fatalf("EnableGPUKeys: %v", err)
	}

	batch, err := page.UploadPreparedQueriesTrusted(pqs)
	if err != nil {
		t.Fatalf("UploadPreparedQueriesTrusted: %v", err)
	}
	defer batch.Close()

	gotOut := make([]float32, len(pqs)*32)
	gotIdx := make([]uint32, len(pqs)*6)
	gotWeights := make([]float32, len(gotIdx))
	if err := batch.AttentionOutputInto(gotOut, gotIdx, gotWeights); err != nil {
		t.Fatalf("AttentionOutputInto: %v", err)
	}

	for i := range wantIdx {
		if gotIdx[i] != wantIdx[i] {
			t.Fatalf("index[%d] = %d want %d", i, gotIdx[i], wantIdx[i])
		}
		if !closeKVFloat32(gotWeights[i], wantWeights[i], 1e-5) {
			t.Fatalf("weight[%d] = %v want %v", i, gotWeights[i], wantWeights[i])
		}
	}
	for i := range wantOut {
		if !closeKVFloat32(gotOut[i], wantOut[i], 1e-5) {
			t.Fatalf("output[%d] = %v want %v", i, gotOut[i], wantOut[i])
		}
	}
}

func TestKVPreparedQueryBatchInvalidatedAfterAppend(t *testing.T) {
	page := NewKVCachePageWithSeed(16, 3, 16, 2, 4, 42)
	rng := newTestRNG()
	for i := 0; i < 16; i++ {
		page.Append(randomUnitVector(16, rng), randomUnitVector(16, rng))
	}
	pqs := []PreparedQuery{
		page.PrepareQuery(randomUnitVector(16, rng)),
		page.PrepareQuery(randomUnitVector(16, rng)),
	}
	err := page.EnableGPUKeys()
	if err != nil {
		if errors.Is(err, ErrGPUBackendUnavailable) {
			t.Skip("GPU KV keys unavailable on this platform")
		}
		t.Fatalf("EnableGPUKeys: %v", err)
	}
	batch, err := page.UploadPreparedQueriesTrusted(pqs)
	if err != nil {
		t.Fatalf("UploadPreparedQueriesTrusted: %v", err)
	}
	defer batch.Close()

	page.Append(randomUnitVector(16, rng), randomUnitVector(16, rng))
	dst := make([]float32, len(pqs)*16)
	indices := make([]uint32, len(pqs)*4)
	weights := make([]float32, len(indices))
	if err := batch.AttentionOutputInto(dst, indices, weights); err == nil {
		t.Fatal("AttentionOutputInto after append = nil error want invalidated batch error")
	}
}

func TestKVCachePageResetClearsLengthAndGPU(t *testing.T) {
	page := NewKVCachePageWithSeed(16, 3, 16, 2, 2, 42)
	rng := newTestRNG()
	page.Append(randomUnitVector(16, rng), randomUnitVector(16, rng))
	page.Append(randomUnitVector(16, rng), randomUnitVector(16, rng))
	if page.Len() != 2 {
		t.Fatalf("Len = %d want 2", page.Len())
	}
	err := page.EnableGPUKeys()
	if err != nil && !errors.Is(err, ErrGPUBackendUnavailable) {
		t.Fatalf("EnableGPUKeys: %v", err)
	}
	page.Reset()
	if page.Len() != 0 {
		t.Fatalf("Len after reset = %d want 0", page.Len())
	}
	if page.GPUKeysEnabled() {
		t.Fatal("GPU keys should be disabled after reset")
	}
	if page.GPUValuesEnabled() {
		t.Fatal("GPU values should be disabled after reset")
	}
}

func closeKVFloat32(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}
