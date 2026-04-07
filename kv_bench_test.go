package turboquant

import (
	"errors"
	"testing"
)

func benchmarkKVCachePageAttentionOutput(b *testing.B, n, dim, keyBits, valueBits, k int, tryGPU bool) {
	page := NewKVCachePageWithSeed(dim, keyBits, dim, valueBits, n, 42)
	rng := newTestRNG()
	for i := 0; i < n; i++ {
		page.Append(randomUnitVector(dim, rng), randomUnitVector(dim, rng))
	}
	if tryGPU {
		if err := page.EnableGPUKeys(); err != nil {
			if errors.Is(err, ErrGPUBackendUnavailable) {
				b.Skip("GPU KV keys unavailable on this platform")
			}
			b.Fatalf("EnableGPUKeys: %v", err)
		}
	}
	pq := page.PrepareQuery(randomUnitVector(dim, rng))
	dst := make([]float32, dim)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		page.AttentionOutputPreparedTo(dst, pq, k)
	}
}

func benchmarkKVCachePageAttentionOutputInto(b *testing.B, n, dim, keyBits, valueBits, k int, tryGPU bool) {
	page := NewKVCachePageWithSeed(dim, keyBits, dim, valueBits, n, 42)
	rng := newTestRNG()
	for i := 0; i < n; i++ {
		page.Append(randomUnitVector(dim, rng), randomUnitVector(dim, rng))
	}
	if tryGPU {
		if err := page.EnableGPUKeys(); err != nil {
			if errors.Is(err, ErrGPUBackendUnavailable) {
				b.Skip("GPU KV keys unavailable on this platform")
			}
			b.Fatalf("EnableGPUKeys: %v", err)
		}
	}
	pq := page.PrepareQuery(randomUnitVector(dim, rng))
	dst := make([]float32, dim)
	indices := make([]uint32, k)
	weights := make([]float32, k)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		page.AttentionOutputPreparedInto(dst, indices, weights, pq)
	}
}

func benchmarkKVCachePageAttentionOutputBatchInto(b *testing.B, n, dim, keyBits, valueBits, k, queries int, tryGPU bool) {
	page := NewKVCachePageWithSeed(dim, keyBits, dim, valueBits, n, 42)
	rng := newTestRNG()
	for i := 0; i < n; i++ {
		page.Append(randomUnitVector(dim, rng), randomUnitVector(dim, rng))
	}
	if tryGPU {
		if err := page.EnableGPUKeys(); err != nil {
			if errors.Is(err, ErrGPUBackendUnavailable) {
				b.Skip("GPU KV keys unavailable on this platform")
			}
			b.Fatalf("EnableGPUKeys: %v", err)
		}
	}
	pqs := make([]PreparedQuery, queries)
	for i := range pqs {
		pqs[i] = page.PrepareQuery(randomUnitVector(dim, rng))
	}
	dst := make([]float32, queries*dim)
	indices := make([]uint32, queries*k)
	weights := make([]float32, queries*k)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		page.AttentionOutputPreparedBatchInto(dst, indices, weights, pqs)
	}
}

func benchmarkKVCachePageAttentionOutputUploadedBatchInto(b *testing.B, n, dim, keyBits, valueBits, k, queries int) {
	page := NewKVCachePageWithSeed(dim, keyBits, dim, valueBits, n, 42)
	rng := newTestRNG()
	for i := 0; i < n; i++ {
		page.Append(randomUnitVector(dim, rng), randomUnitVector(dim, rng))
	}
	if err := page.EnableGPUKeys(); err != nil {
		if errors.Is(err, ErrGPUBackendUnavailable) {
			b.Skip("GPU KV keys unavailable on this platform")
		}
		b.Fatalf("EnableGPUKeys: %v", err)
	}
	pqs := make([]PreparedQuery, queries)
	for i := range pqs {
		pqs[i] = page.PrepareQuery(randomUnitVector(dim, rng))
	}
	batch, err := page.UploadPreparedQueriesTrusted(pqs)
	if err != nil {
		b.Fatalf("UploadPreparedQueriesTrusted: %v", err)
	}
	defer batch.Close()
	dst := make([]float32, queries*dim)
	indices := make([]uint32, queries*k)
	weights := make([]float32, queries*k)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := batch.AttentionOutputInto(dst, indices, weights); err != nil {
			b.Fatalf("AttentionOutputInto: %v", err)
		}
	}
}

func BenchmarkKVCachePageAttentionOutputPrepared100K384_3bit(b *testing.B) {
	benchmarkKVCachePageAttentionOutput(b, 100_000, 384, 3, 2, 16, false)
}

func BenchmarkKVCachePageAttentionOutputPreparedGPU100K384_3bit(b *testing.B) {
	benchmarkKVCachePageAttentionOutput(b, 100_000, 384, 3, 2, 16, true)
}

func BenchmarkKVCachePageAttentionOutputInto100K384_3bit(b *testing.B) {
	benchmarkKVCachePageAttentionOutputInto(b, 100_000, 384, 3, 2, 16, false)
}

func BenchmarkKVCachePageAttentionOutputIntoGPU100K384_3bit(b *testing.B) {
	benchmarkKVCachePageAttentionOutputInto(b, 100_000, 384, 3, 2, 16, true)
}

func BenchmarkKVCachePageAttentionOutputBatchInto100K384_3bitx4(b *testing.B) {
	benchmarkKVCachePageAttentionOutputBatchInto(b, 100_000, 384, 3, 2, 16, 4, false)
}

func BenchmarkKVCachePageAttentionOutputBatchIntoGPU100K384_3bitx4(b *testing.B) {
	benchmarkKVCachePageAttentionOutputBatchInto(b, 100_000, 384, 3, 2, 16, 4, true)
}

func BenchmarkKVCachePageAttentionOutputUploadedBatchIntoGPU100K384_3bitx4(b *testing.B) {
	benchmarkKVCachePageAttentionOutputUploadedBatchInto(b, 100_000, 384, 3, 2, 16, 4)
}
