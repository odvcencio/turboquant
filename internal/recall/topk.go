package recall

import "sort"

// TopK selects the k highest scores. It writes indices in descending score
// order into dstIndices and the matching scores into dstScores. It breaks
// ties by ascending index, matching the CPU top-k selector in gpu_cuda.go.
// k is len(dstIndices). Slots beyond len(scores) receive index 0 and score 0.
func TopK(dstIndices []uint32, dstScores, scores []float32) {
	k := len(dstIndices)
	if len(dstScores) != k {
		panic("recall: TopK destination length mismatch")
	}
	if k == 0 {
		return
	}

	order := make([]uint32, len(scores))
	for i := range order {
		order[i] = uint32(i)
	}
	sort.Slice(order, func(i, j int) bool {
		a, b := order[i], order[j]
		if scores[a] != scores[b] {
			return scores[a] > scores[b]
		}
		return a < b
	})

	for i := 0; i < k; i++ {
		if i < len(order) {
			dstIndices[i] = order[i]
			dstScores[i] = scores[order[i]]
		} else {
			dstIndices[i] = 0
			dstScores[i] = 0
		}
	}
}
