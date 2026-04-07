//go:build js && wasm

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"syscall/js"

	"github.com/odvcencio/turboquant"
)

type smokeResult struct {
	Count           int
	Top1CPU         int
	Top1GPU         int
	Top10Match      int
	TopKGPU         []int
	TopKPassed      bool
	BatchTopKPassed bool
	BatchTop1CPU    []int
	BatchTop1GPU    []int
	MaxAbsDiff      float32
	MeanAbsDiff     float32
	SampleCPU       []float32
	SampleGPU       []float32
	Passed          bool
}

func main() {
	js.Global().Set("runTurboQuantWebGPUSmoke", js.FuncOf(func(this js.Value, args []js.Value) any {
		promiseCtor := js.Global().Get("Promise")
		executor := js.FuncOf(func(this js.Value, args []js.Value) any {
			resolve := args[0]
			reject := args[1]
			go func() {
				result, err := runSmoke()
				if err != nil {
					reject.Invoke(err.Error())
					return
				}
				resolve.Invoke(result.toJS())
			}()
			return nil
		})
		return promiseCtor.New(executor)
	}))
	select {}
}

func runSmoke() (smokeResult, error) {
	const (
		dim      = 384
		bitWidth = 3
		count    = 128
	)

	q := turboquant.NewIPHadamardWithSeed(dim, bitWidth, 42)
	rng := rand.New(rand.NewSource(99))

	corpus := make([]turboquant.IPQuantized, count)
	for i := 0; i < count; i++ {
		corpus[i] = q.Quantize(randomUnitVector(dim, rng))
	}
	query := randomUnitVector(dim, rng)
	pq := q.PrepareQuery(query)
	batchQueries := make([]turboquant.PreparedQuery, 4)
	for i := range batchQueries {
		batchQueries[i] = q.PrepareQuery(randomUnitVector(dim, rng))
	}

	cpuScores := make([]float32, count)
	for i := range corpus {
		cpuScores[i] = q.InnerProductPreparedTrusted(corpus[i], pq)
	}
	cpuBatchTop1 := make([]int, len(batchQueries))
	cpuBatchTopK := make([][]int, len(batchQueries))
	for qi := range batchQueries {
		scores := make([]float32, count)
		for i := range corpus {
			scores[i] = q.InnerProductPreparedTrusted(corpus[i], batchQueries[qi])
		}
		order := topKIndices(scores, 10)
		cpuBatchTopK[qi] = append([]int(nil), order...)
		cpuBatchTop1[qi] = order[0]
	}

	scorer, err := q.NewGPUPreparedScorer(corpus)
	if err != nil {
		return smokeResult{}, err
	}
	defer scorer.Close()

	gpuScores, err := scorer.ScorePreparedQuery(pq)
	if err != nil {
		return smokeResult{}, err
	}
	if len(gpuScores) != count {
		return smokeResult{}, fmt.Errorf("gpu score length = %d want %d", len(gpuScores), count)
	}
	gpuTopIdx, gpuTopScores, err := scorer.ScorePreparedQueryTopK(pq, 10)
	if err != nil {
		return smokeResult{}, err
	}
	gpuBatchIdx, gpuBatchScores, err := scorer.ScorePreparedQueriesTopK(batchQueries, 10)
	if err != nil {
		return smokeResult{}, err
	}
	uploadedBatch, err := scorer.UploadPreparedQueries(batchQueries)
	if err != nil {
		return smokeResult{}, err
	}
	defer uploadedBatch.Close()
	uploadedBatchIdx, uploadedBatchScores, err := uploadedBatch.ScoreTopK(10)
	if err != nil {
		return smokeResult{}, err
	}

	var maxAbs float32
	var sumAbs float32
	for i := 0; i < count; i++ {
		diff := abs32(cpuScores[i] - gpuScores[i])
		if diff > maxAbs {
			maxAbs = diff
		}
		sumAbs += diff
	}

	cpuOrder := topKIndices(cpuScores, 10)
	gpuOrder := topKIndices(gpuScores, 10)
	top10Match := 0
	for i := 0; i < len(cpuOrder) && i < len(gpuOrder); i++ {
		if cpuOrder[i] == gpuOrder[i] {
			top10Match++
		}
	}
	topKPassed := len(gpuTopIdx) == len(cpuOrder)
	topKGPU := make([]int, len(gpuTopIdx))
	for i := range gpuTopIdx {
		topKGPU[i] = int(gpuTopIdx[i])
		if topKGPU[i] != cpuOrder[i] || gpuTopScores[i] != cpuScores[cpuOrder[i]] {
			topKPassed = false
		}
	}
	batchTopKPassed := len(gpuBatchIdx) == len(batchQueries)*10
	batchTop1GPU := make([]int, len(batchQueries))
	for qi := range batchQueries {
		base := qi * 10
		batchTop1GPU[qi] = int(gpuBatchIdx[base])
		for i := 0; i < 10; i++ {
			gotIdx := int(gpuBatchIdx[base+i])
			wantIdx := cpuBatchTopK[qi][i]
			if gotIdx != wantIdx || gpuBatchScores[base+i] != q.InnerProductPreparedTrusted(corpus[wantIdx], batchQueries[qi]) {
				batchTopKPassed = false
			}
			if int(uploadedBatchIdx[base+i]) != wantIdx || uploadedBatchScores[base+i] != q.InnerProductPreparedTrusted(corpus[wantIdx], batchQueries[qi]) {
				batchTopKPassed = false
			}
		}
	}

	result := smokeResult{
		Count:           count,
		Top1CPU:         cpuOrder[0],
		Top1GPU:         gpuOrder[0],
		Top10Match:      top10Match,
		TopKGPU:         topKGPU,
		TopKPassed:      topKPassed,
		BatchTopKPassed: batchTopKPassed,
		BatchTop1CPU:    cpuBatchTop1,
		BatchTop1GPU:    batchTop1GPU,
		MaxAbsDiff:      maxAbs,
		MeanAbsDiff:     sumAbs / float32(count),
		SampleCPU:       append([]float32(nil), cpuScores[:8]...),
		SampleGPU:       append([]float32(nil), gpuScores[:8]...),
		Passed:          maxAbs <= 1e-4 && cpuOrder[0] == gpuOrder[0] && top10Match >= 9 && topKPassed && batchTopKPassed,
	}
	return result, nil
}

func (r smokeResult) toJS() js.Value {
	return js.ValueOf(map[string]any{
		"count":           r.Count,
		"top1CPU":         r.Top1CPU,
		"top1GPU":         r.Top1GPU,
		"top10Match":      r.Top10Match,
		"topKGPU":         intSliceToAny(r.TopKGPU),
		"topKPassed":      r.TopKPassed,
		"batchTopKPassed": r.BatchTopKPassed,
		"batchTop1CPU":    intSliceToAny(r.BatchTop1CPU),
		"batchTop1GPU":    intSliceToAny(r.BatchTop1GPU),
		"maxAbsDiff":      r.MaxAbsDiff,
		"meanAbsDiff":     r.MeanAbsDiff,
		"sampleCPU":       float32SliceToAny(r.SampleCPU),
		"sampleGPU":       float32SliceToAny(r.SampleGPU),
		"passed":          r.Passed,
	})
}

func float32SliceToAny(values []float32) []any {
	dst := make([]any, len(values))
	for i := range values {
		dst[i] = values[i]
	}
	return dst
}

func intSliceToAny(values []int) []any {
	dst := make([]any, len(values))
	for i := range values {
		dst[i] = values[i]
	}
	return dst
}

func randomUnitVector(dim int, rng *rand.Rand) []float32 {
	vec := make([]float32, dim)
	var norm2 float64
	for i := range vec {
		value := rng.NormFloat64()
		vec[i] = float32(value)
		norm2 += value * value
	}
	if norm2 == 0 {
		return vec
	}
	scale := 1 / math.Sqrt(norm2)
	for i := range vec {
		vec[i] = float32(float64(vec[i]) * scale)
	}
	return vec
}

func topKIndices(scores []float32, k int) []int {
	indices := make([]int, len(scores))
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		li := indices[i]
		lj := indices[j]
		if scores[li] == scores[lj] {
			return li < lj
		}
		return scores[li] > scores[lj]
	})
	if k > len(indices) {
		k = len(indices)
	}
	return indices[:k]
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
