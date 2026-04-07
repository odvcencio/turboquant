//go:build linux && amd64 && cgo && cuda

package turboquant

import (
	"math"
	"testing"
)

func TestCUDAPreparedScorerMatchesCPU(t *testing.T) {
	q := NewIPHadamardWithSeed(64, 3, 42)
	rng := newTestRNG()
	corpus := make([]IPQuantized, 32)
	for i := range corpus {
		corpus[i] = q.Quantize(randomUnitVector(64, rng))
	}
	scorer, err := q.NewGPUPreparedScorer(corpus)
	if err != nil {
		t.Fatalf("NewGPUPreparedScorer: %v", err)
	}
	defer scorer.Close()

	query := q.PrepareQuery(randomUnitVector(64, rng))
	cpuScores := make([]float32, len(corpus))
	for i := range corpus {
		cpuScores[i] = q.InnerProductPreparedTrusted(corpus[i], query)
	}
	gpuScores, err := scorer.ScorePreparedQuery(query)
	if err != nil {
		t.Fatalf("ScorePreparedQuery: %v", err)
	}
	for i := range cpuScores {
		if !closeFloat32(cpuScores[i], gpuScores[i], 1e-6) {
			t.Fatalf("score[%d] = %v want %v", i, gpuScores[i], cpuScores[i])
		}
	}

	indices, scores, err := scorer.ScorePreparedQueryTopK(query, 8)
	if err != nil {
		t.Fatalf("ScorePreparedQueryTopK: %v", err)
	}
	wantIdx := topKIndicesFloat32(cpuScores, 8)
	for i := range wantIdx {
		if int(indices[i]) != wantIdx[i] || !closeFloat32(scores[i], cpuScores[wantIdx[i]], 1e-6) {
			t.Fatalf("topk[%d] = (%d,%v) want (%d,%v)", i, indices[i], scores[i], wantIdx[i], cpuScores[wantIdx[i]])
		}
	}
}

func TestCUDAUploadedPreparedBatchMatchesCPU(t *testing.T) {
	q := NewIPHadamardWithSeed(64, 3, 42)
	rng := newTestRNG()
	corpus := make([]IPQuantized, 32)
	for i := range corpus {
		corpus[i] = q.Quantize(randomUnitVector(64, rng))
	}
	scorer, err := q.NewGPUPreparedScorer(corpus)
	if err != nil {
		t.Fatalf("NewGPUPreparedScorer: %v", err)
	}
	defer scorer.Close()

	pqs := make([]PreparedQuery, 4)
	for i := range pqs {
		pqs[i] = q.PrepareQuery(randomUnitVector(64, rng))
	}
	uploaded, err := scorer.UploadPreparedQueries(pqs)
	if err != nil {
		t.Fatalf("UploadPreparedQueries: %v", err)
	}
	defer uploaded.Close()

	indices, scores, err := uploaded.ScoreTopK(6)
	if err != nil {
		t.Fatalf("ScoreTopK: %v", err)
	}
	for queryIdx := range pqs {
		cpuScores := make([]float32, len(corpus))
		for i := range corpus {
			cpuScores[i] = q.InnerProductPreparedTrusted(corpus[i], pqs[queryIdx])
		}
		wantIdx := topKIndicesFloat32(cpuScores, 6)
		base := queryIdx * 6
		for i := 0; i < 6; i++ {
			if int(indices[base+i]) != wantIdx[i] || !closeFloat32(scores[base+i], cpuScores[wantIdx[i]], 1e-6) {
				t.Fatalf("query %d topk[%d] = (%d,%v) want (%d,%v)", queryIdx, i, indices[base+i], scores[base+i], wantIdx[i], cpuScores[wantIdx[i]])
			}
		}
	}
}

func TestCUDAPreparedScorerTopKWorksWithoutExplicitRanks(t *testing.T) {
	q := NewIPHadamardWithSeed(64, 3, 42)
	rng := newTestRNG()
	corpus := make([]IPQuantized, 32)
	for i := range corpus {
		corpus[i] = q.Quantize(randomUnitVector(64, rng))
	}
	data := q.PackGPUPreparedData(corpus)
	data.TieBreakRanks = nil
	scorer, err := q.NewGPUPreparedScorerFromData(data)
	if err != nil {
		t.Fatalf("NewGPUPreparedScorerFromData: %v", err)
	}
	defer scorer.Close()

	query := q.PrepareQuery(randomUnitVector(64, rng))
	indices, scores, err := scorer.ScorePreparedQueryTopK(query, 8)
	if err != nil {
		t.Fatalf("ScorePreparedQueryTopK: %v", err)
	}

	cpuScores := make([]float32, len(corpus))
	for i := range corpus {
		cpuScores[i] = q.InnerProductPreparedTrusted(corpus[i], query)
	}
	wantIdx := topKIndicesFloat32(cpuScores, 8)
	for i := range wantIdx {
		if int(indices[i]) != wantIdx[i] || !closeFloat32(scores[i], cpuScores[wantIdx[i]], 1e-6) {
			t.Fatalf("topk[%d] = (%d,%v) want (%d,%v)", i, indices[i], scores[i], wantIdx[i], cpuScores[wantIdx[i]])
		}
	}
}

func closeFloat32(a, b, tol float32) bool {
	return float32(math.Abs(float64(a-b))) <= tol
}

func topKIndicesFloat32(scores []float32, k int) []int {
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}
	for i := 1; i < len(indices); i++ {
		cur := indices[i]
		j := i - 1
		for ; j >= 0; j-- {
			if scores[indices[j]] > scores[cur] || (scores[indices[j]] == scores[cur] && indices[j] < cur) {
				break
			}
			indices[j+1] = indices[j]
		}
		indices[j+1] = cur
	}
	if k > len(indices) {
		k = len(indices)
	}
	return indices[:k]
}
