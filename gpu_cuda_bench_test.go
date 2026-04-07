//go:build linux && amd64 && cgo && cuda

package turboquant

import "testing"

func benchmarkCUDAScorer(b *testing.B, count, queries, k int) {
	q := NewIPHadamardWithSeed(384, 3, 42)
	rng := newTestRNG()
	corpus := make([]IPQuantized, count)
	for i := range corpus {
		corpus[i] = q.Quantize(randomUnitVector(384, rng))
	}
	scorer, err := q.NewGPUPreparedScorer(corpus)
	if err != nil {
		b.Fatalf("NewGPUPreparedScorer: %v", err)
	}
	defer scorer.Close()

	pqs := make([]PreparedQuery, queries)
	for i := range pqs {
		pqs[i] = q.PrepareQuery(randomUnitVector(384, rng))
	}
	uploaded, err := scorer.UploadPreparedQueries(pqs)
	if err != nil {
		b.Fatalf("UploadPreparedQueries: %v", err)
	}
	defer uploaded.Close()

	b.ReportAllocs()
	b.ResetTimer()
	if queries == 1 {
		for i := 0; i < b.N; i++ {
			if _, _, err := scorer.ScorePreparedQueryTopK(pqs[0], k); err != nil {
				b.Fatalf("ScorePreparedQueryTopK: %v", err)
			}
		}
		return
	}
	for i := 0; i < b.N; i++ {
		if _, _, err := uploaded.ScoreTopK(k); err != nil {
			b.Fatalf("ScoreTopK: %v", err)
		}
	}
}

func benchmarkCUDAScorerTo(b *testing.B, count, queries, k int) {
	q := NewIPHadamardWithSeed(384, 3, 42)
	rng := newTestRNG()
	corpus := make([]IPQuantized, count)
	for i := range corpus {
		corpus[i] = q.Quantize(randomUnitVector(384, rng))
	}
	scorer, err := q.NewGPUPreparedScorer(corpus)
	if err != nil {
		b.Fatalf("NewGPUPreparedScorer: %v", err)
	}
	defer scorer.Close()

	pqs := make([]PreparedQuery, queries)
	for i := range pqs {
		pqs[i] = q.PrepareQuery(randomUnitVector(384, rng))
	}
	uploaded, err := scorer.UploadPreparedQueries(pqs)
	if err != nil {
		b.Fatalf("UploadPreparedQueries: %v", err)
	}
	defer uploaded.Close()

	indices := make([]uint32, queries*k)
	scores := make([]float32, queries*k)
	b.ReportAllocs()
	b.ResetTimer()
	if queries == 1 {
		for i := 0; i < b.N; i++ {
			if err := scorer.ScorePreparedQueryTopKTo(indices[:k], scores[:k], pqs[0]); err != nil {
				b.Fatalf("ScorePreparedQueryTopKTo: %v", err)
			}
		}
		return
	}
	for i := 0; i < b.N; i++ {
		if err := uploaded.ScoreTopKTo(indices, scores, k); err != nil {
			b.Fatalf("ScoreTopKTo: %v", err)
		}
	}
}

func BenchmarkCUDAPreparedQueryTopK100K384_3bit(b *testing.B) {
	benchmarkCUDAScorer(b, 100_000, 1, 10)
}

func BenchmarkCUDAUploadedPreparedBatchTopK100K384_3bitx4(b *testing.B) {
	benchmarkCUDAScorer(b, 100_000, 4, 10)
}

func BenchmarkCUDAPreparedQueryTopKTo100K384_3bit(b *testing.B) {
	benchmarkCUDAScorerTo(b, 100_000, 1, 10)
}

func BenchmarkCUDAUploadedPreparedBatchTopKTo100K384_3bitx4(b *testing.B) {
	benchmarkCUDAScorerTo(b, 100_000, 4, 10)
}
