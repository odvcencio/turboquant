//go:build linux && amd64 && cgo && cuda

package turboquant

import (
	"errors"
	"math"
	"math/rand"
	"sort"
	"testing"
)

// TestCUDAScorerParityNonUnitNorms exercises the score kernel's per-row
// input-norm rescaling against the CPU estimator with a corpus of widely
// varying vector norms. Top-k order must agree exactly and scores must match
// to float tolerance.
func TestCUDAScorerParityNonUnitNorms(t *testing.T) {
	const dim, bits, count, k = 64, 3, 96, 8
	q := NewIPHadamardWithSeed(dim, bits, 91)
	rng := rand.New(rand.NewSource(17))

	vectors := make([]IPQuantized, count)
	for i := range vectors {
		scale := float32(0.05 + rng.Float64()*10)
		vectors[i] = q.Quantize(scaledVector(randomUnitVector(dim, rng), scale))
	}

	scorer, err := q.NewGPUPreparedScorer(vectors)
	if errors.Is(err, ErrGPUBackendUnavailable) {
		t.Skip("GPU backend unavailable")
	}
	if err != nil {
		t.Fatal(err)
	}
	defer scorer.Close()

	query := randomUnitVector(dim, rng)
	pq := q.PrepareQuery(query)

	cpuScores := make([]float32, count)
	for i := range vectors {
		cpuScores[i] = q.InnerProductPrepared(vectors[i], pq)
	}
	order := make([]int, count)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		if cpuScores[order[a]] != cpuScores[order[b]] {
			return cpuScores[order[a]] > cpuScores[order[b]]
		}
		return order[a] < order[b]
	})

	gpuIdx := make([]uint32, k)
	gpuScores := make([]float32, k)
	if err := scorer.ScorePreparedQueryTopKTo(gpuIdx, gpuScores, pq); err != nil {
		t.Fatal(err)
	}
	for rank := range k {
		wantIdx := uint32(order[rank])
		wantScore := cpuScores[order[rank]]
		if gpuIdx[rank] != wantIdx {
			t.Fatalf("rank %d: GPU index %d want %d (gpu=%v cpuTop=%v)", rank, gpuIdx[rank], wantIdx, gpuScores, cpuScores[order[0]:order[0]+1])
		}
		if math.Abs(float64(gpuScores[rank]-wantScore)) > 1e-3*(1+math.Abs(float64(wantScore))) {
			t.Fatalf("rank %d: GPU score %v want %v", rank, gpuScores[rank], wantScore)
		}
	}
}
