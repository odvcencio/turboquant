package recall

import (
	"math/rand"
	"sort"
	"testing"
)

func TestTopKMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	n := 200
	scores := make([]float32, n)
	for i := range scores {
		scores[i] = rng.Float32()*2 - 1
	}

	type pair struct {
		idx   uint32
		score float32
	}
	all := make([]pair, n)
	for i, s := range scores {
		all[i] = pair{uint32(i), s}
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].score != all[j].score {
			return all[i].score > all[j].score
		}
		return all[i].idx < all[j].idx
	})

	for k := 1; k <= 32; k++ {
		dstIdx := make([]uint32, k)
		dstScores := make([]float32, k)
		TopK(dstIdx, dstScores, scores)
		for i := 0; i < k; i++ {
			if dstIdx[i] != all[i].idx || dstScores[i] != all[i].score {
				t.Fatalf("k=%d i=%d: got idx=%d score=%v want idx=%d score=%v",
					k, i, dstIdx[i], dstScores[i], all[i].idx, all[i].score)
			}
		}
	}
}

func TestTopKTieBreak(t *testing.T) {
	scores := []float32{1, 1, 1, 1, 1}
	dstIdx := make([]uint32, 3)
	dstScores := make([]float32, 3)
	TopK(dstIdx, dstScores, scores)
	want := []uint32{0, 1, 2}
	for i, w := range want {
		if dstIdx[i] != w {
			t.Fatalf("index[%d] = %d want %d", i, dstIdx[i], w)
		}
		if dstScores[i] != 1 {
			t.Fatalf("score[%d] = %v want 1", i, dstScores[i])
		}
	}
}

func TestTopKPadsShortInput(t *testing.T) {
	scores := []float32{5, 3}
	dstIdx := make([]uint32, 4)
	dstScores := make([]float32, 4)
	TopK(dstIdx, dstScores, scores)
	wantIdx := []uint32{0, 1, 0, 0}
	wantScores := []float32{5, 3, 0, 0}
	for i := range wantIdx {
		if dstIdx[i] != wantIdx[i] || dstScores[i] != wantScores[i] {
			t.Fatalf("slot %d = (%d, %v) want (%d, %v)", i, dstIdx[i], dstScores[i], wantIdx[i], wantScores[i])
		}
	}
}
