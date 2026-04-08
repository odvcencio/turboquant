package turboquant

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

const (
	// TransformerEvalMethodTurboQuant uses the existing TurboQuant key/value
	// quantizers for approximate attention reconstruction.
	TransformerEvalMethodTurboQuant = "turboquant"
	// TransformerEvalMethodUniform uses a uniform scalar quantization baseline
	// over per-vector normalized coordinates.
	TransformerEvalMethodUniform = "uniform"
)

func normalizeTransformerEvalMethod(raw string) (string, error) {
	method := strings.ToLower(strings.TrimSpace(raw))
	if method == "" {
		return TransformerEvalMethodTurboQuant, nil
	}
	switch method {
	case TransformerEvalMethodTurboQuant, TransformerEvalMethodUniform:
		return method, nil
	default:
		return "", fmt.Errorf("turboquant: unsupported transformer eval method %q", raw)
	}
}

// NormalizeTransformerEvalMethodForCLI validates and normalizes an evaluation
// method name for CLI entry points.
func NormalizeTransformerEvalMethodForCLI(raw string) (string, error) {
	return normalizeTransformerEvalMethod(raw)
}

type transformerApproxEval struct {
	output       []float32
	liveBytes    uint64
	storageBytes uint64
	gpuEnabled   bool
}

func evaluateTransformerCaptureTurboQuant(capture TransformerLayerCapture, cfg TransformerLayerEvalConfig, query []float32, topK int) (transformerApproxEval, error) {
	cache, err := capture.BuildCache(cfg.KeyBits, cfg.ValueBits, cfg.Capacity, cfg.Seed)
	if err != nil {
		return transformerApproxEval{}, err
	}
	usedGPU := false
	// The current GPU scorer needs a prepared-query MSE LUT on the key side.
	// Some valid IP bit widths (for example 4-bit keys, which use a 3-bit MSE
	// stage) do not have that fast path yet, so sweep those configs on CPU
	// instead of aborting the whole run.
	if cfg.TryGPU && preparedQueryMSELUTLen(capture.HeadDim, cfg.KeyBits-1) > 0 {
		err = cache.EnableGPUKeys()
		if err == nil {
			usedGPU = true
		} else if !errors.Is(err, ErrGPUBackendUnavailable) {
			return transformerApproxEval{}, err
		}
	}
	approx := make([]float32, capture.tokenDim())
	if topK > 0 {
		if err := capture.AttentionOutputFromCacheInto(approx, cache, query, topK); err != nil {
			return transformerApproxEval{}, err
		}
	}
	return transformerApproxEval{
		output:       approx,
		liveBytes:    cache.LiveBytes(),
		storageBytes: cache.StorageBytes(),
		gpuEnabled:   usedGPU,
	}, nil
}

func evaluateTransformerCaptureUniform(capture TransformerLayerCapture, cfg TransformerLayerEvalConfig, query []float32, topK int) (transformerApproxEval, error) {
	approx := make([]float32, capture.tokenDim())
	uniformAttentionOutputInto(approx, capture, query, cfg.KeyBits, cfg.ValueBits, topK)

	tokens := capture.TokenCount()
	capacity := maxInt(cfg.Capacity, tokens)
	return transformerApproxEval{
		output:       approx,
		liveBytes:    uniformTransformerLayerBytes(capture.kvHeads(), capture.HeadDim, cfg.KeyBits, cfg.ValueBits, tokens),
		storageBytes: uniformTransformerLayerBytes(capture.kvHeads(), capture.HeadDim, cfg.KeyBits, cfg.ValueBits, capacity),
		gpuEnabled:   false,
	}, nil
}

func groupedTransformerAttentionOutputTurboQuantInto(dst []float32, cache *TransformerLayerKVCache, capture TransformerLayerCapture, query []float32, topK int) {
	for i := range dst {
		dst[i] = 0
	}
	if topK <= 0 {
		return
	}
	perKV := capture.queryHeadsPerKVHead()
	if perKV <= 0 {
		return
	}

	headDim := capture.HeadDim
	pqs := make([]PreparedQuery, perKV)
	indices := make([]uint32, perKV*topK)
	weights := make([]float32, len(indices))
	for kvHead, page := range cache.pages {
		baseHead := kvHead * perKV
		for i := 0; i < perKV; i++ {
			queryHead := baseHead + i
			baseDim := queryHead * headDim
			pqs[i] = page.PrepareQuery(query[baseDim : baseDim+headDim])
		}
		baseDim := baseHead * headDim
		page.AttentionOutputPreparedBatchInto(
			dst[baseDim:baseDim+perKV*headDim],
			indices,
			weights,
			pqs,
		)
	}
}

func uniformTransformerLayerBytes(heads, headDim, keyBits, valueBits, entries int) uint64 {
	if entries <= 0 || heads <= 0 || headDim <= 0 {
		return 0
	}
	perHead := PackedSize(headDim, keyBits) + 4 + PackedSize(headDim, valueBits) + 4
	return uint64(entries * heads * perHead)
}

func uniformAttentionOutputInto(dst []float32, capture TransformerLayerCapture, query []float32, keyBits, valueBits, topK int) {
	for i := range dst {
		dst[i] = 0
	}
	tokens := capture.TokenCount()
	if tokens == 0 || topK <= 0 {
		return
	}
	if topK > tokens {
		topK = tokens
	}

	headDim := capture.HeadDim
	scores := make([]float32, tokens)
	order := make([]int, tokens)
	keyBuf := make([]float32, headDim)
	valueBuf := make([]float32, headDim)
	weights := make([]float32, topK)
	indices := make([]int, topK)

	for head := 0; head < capture.Heads; head++ {
		base := head * headDim
		kvBase := capture.kvHeadForQueryHead(head) * headDim
		q := query[base : base+headDim]
		for token := 0; token < tokens; token++ {
			order[token] = token
			uniformQuantizeReconstructTo(keyBuf, capture.keyToken(token)[kvBase:kvBase+headDim], keyBits)
			scores[token] = dotFloat32(q, keyBuf)
		}
		sort.Slice(order, func(i, j int) bool {
			leftIdx := order[i]
			rightIdx := order[j]
			if scores[leftIdx] != scores[rightIdx] {
				return scores[leftIdx] > scores[rightIdx]
			}
			return leftIdx < rightIdx
		})
		maxScore := scores[order[0]]
		var sum float64
		for i := 0; i < topK; i++ {
			idx := order[i]
			indices[i] = idx
			weight := math.Exp(float64(scores[idx] - maxScore))
			weights[i] = float32(weight)
			sum += weight
		}
		if sum == 0 {
			continue
		}
		inv := float32(1 / sum)
		for i := 0; i < topK; i++ {
			idx := indices[i]
			uniformQuantizeReconstructTo(valueBuf, capture.valueToken(idx)[kvBase:kvBase+headDim], valueBits)
			weight := weights[i] * inv
			for j := 0; j < headDim; j++ {
				dst[base+j] += weight * valueBuf[j]
			}
		}
	}
}

func uniformQuantizeReconstructTo(dst []float32, vec []float32, bitWidth int) {
	if len(dst) != len(vec) {
		panic(fmt.Sprintf("turboquant: uniform reconstruction length mismatch: %d vs %d", len(dst), len(vec)))
	}
	if bitWidth <= 0 {
		panic(fmt.Sprintf("turboquant: invalid uniform bit width %d", bitWidth))
	}
	norm := vecNorm(vec)
	if norm <= 1e-12 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	levelsInt := (1 << uint(bitWidth)) - 1
	levels := float32(levelsInt)
	invNorm := float32(1) / norm
	for i, value := range vec {
		unit := value * invNorm
		if unit > 1 {
			unit = 1
		} else if unit < -1 {
			unit = -1
		}
		idx := int(math.Round(float64((unit + 1) * 0.5 * levels)))
		if idx < 0 {
			idx = 0
		} else if idx > levelsInt {
			idx = levelsInt
		}
		recon := (float32(idx)/levels)*2 - 1
		dst[i] = recon * norm
	}
}
