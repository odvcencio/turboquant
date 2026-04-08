package turboquant

import (
	"fmt"
	"math"
)

// TransformerLayerCaptureFile groups one or more captured transformer-layer
// attention states for offline TurboQuant ingestion and evaluation.
type TransformerLayerCaptureFile struct {
	Samples []TransformerLayerCapture `json:"samples"`
}

// TransformerLayerCapture represents one captured transformer attention step
// for a single layer. Query is flattened head-major as [heads*headDim]. Keys
// and values are flattened token-major, with each token flattened head-major as
// [kv_heads*headDim]. When kv_heads is omitted, it defaults to heads.
type TransformerLayerCapture struct {
	Name           string `json:"name,omitempty"`
	Model          string `json:"model,omitempty"`
	Prompt         string `json:"prompt,omitempty"`
	PromptIndex    int    `json:"prompt_index,omitempty"`
	Layer          int    `json:"layer,omitempty"`
	TokenIndex     int    `json:"token_index,omitempty"`
	TokenPosition  string `json:"token_position,omitempty"`
	SequenceLength int    `json:"sequence_length,omitempty"`
	Heads          int    `json:"heads"`
	KVHeads        int    `json:"kv_heads,omitempty"`
	HeadDim        int    `json:"head_dim"`
	Tokens         int    `json:"tokens,omitempty"`
	// QueryScale optionally stores the model-native attention scaling factor,
	// typically 1/sqrt(head_dim), for faithful offline reconstruction.
	QueryScale float32   `json:"query_scale,omitempty"`
	Query      []float32 `json:"query"`
	Keys       []float32 `json:"keys"`
	Values     []float32 `json:"values"`
}

// TransformerLayerEvalConfig controls offline ingestion and evaluation of one
// captured transformer layer.
type TransformerLayerEvalConfig struct {
	KeyBits    int     `json:"key_bits"`
	ValueBits  int     `json:"value_bits"`
	Method     string  `json:"method,omitempty"`
	TopK       int     `json:"top_k"`
	Capacity   int     `json:"capacity,omitempty"`
	Seed       int64   `json:"seed,omitempty"`
	QueryScale float32 `json:"query_scale,omitempty"`
	TryGPU     bool    `json:"try_gpu,omitempty"`
}

// TransformerLayerHeadEval stores per-head attention reconstruction metrics.
type TransformerLayerHeadEval struct {
	Head         int     `json:"head"`
	OutputMSE    float64 `json:"output_mse"`
	OutputCosine float64 `json:"output_cosine"`
	ExactNorm    float64 `json:"exact_norm"`
	QuantNorm    float64 `json:"quant_norm"`
}

// TransformerLayerEvalResult stores aggregate metrics for one captured layer.
type TransformerLayerEvalResult struct {
	Model             string                     `json:"model,omitempty"`
	Name              string                     `json:"name,omitempty"`
	PromptIndex       int                        `json:"prompt_index,omitempty"`
	Layer             int                        `json:"layer,omitempty"`
	TokenIndex        int                        `json:"token_index,omitempty"`
	TokenPosition     string                     `json:"token_position,omitempty"`
	SequenceLength    int                        `json:"sequence_length,omitempty"`
	Tokens            int                        `json:"tokens"`
	Heads             int                        `json:"heads"`
	KVHeads           int                        `json:"kv_heads,omitempty"`
	HeadDim           int                        `json:"head_dim"`
	Method            string                     `json:"method"`
	KeyBits           int                        `json:"key_bits"`
	ValueBits         int                        `json:"value_bits"`
	RequestedTopK     int                        `json:"requested_top_k"`
	TopK              int                        `json:"top_k"`
	Capacity          int                        `json:"capacity"`
	QueryScale        float32                    `json:"query_scale"`
	GPUEnabled        bool                       `json:"gpu_enabled"`
	RawKVBytes        uint64                     `json:"raw_kv_bytes"`
	CacheLiveBytes    uint64                     `json:"cache_live_bytes"`
	CacheStorageBytes uint64                     `json:"cache_storage_bytes"`
	CompressionRatio  float64                    `json:"compression_ratio"`
	OutputMSE         float64                    `json:"output_mse"`
	OutputCosine      float64                    `json:"output_cosine"`
	HeadsMetrics      []TransformerLayerHeadEval `json:"heads_metrics"`
}

func (c TransformerLayerCapture) tokenDim() int {
	return c.Heads * c.HeadDim
}

func (c TransformerLayerCapture) kvHeads() int {
	if c.KVHeads > 0 {
		return c.KVHeads
	}
	return c.Heads
}

func (c TransformerLayerCapture) kvTokenDim() int {
	return c.kvHeads() * c.HeadDim
}

func (c TransformerLayerCapture) queryHeadsPerKVHead() int {
	kvHeads := c.kvHeads()
	if kvHeads <= 0 {
		return 0
	}
	return c.Heads / kvHeads
}

func (c TransformerLayerCapture) kvHeadForQueryHead(head int) int {
	perKV := c.queryHeadsPerKVHead()
	if perKV <= 0 {
		return 0
	}
	return head / perKV
}

// TokenCount reports the number of cached tokens represented by this capture.
func (c TransformerLayerCapture) TokenCount() int {
	tokenDim := c.kvTokenDim()
	if tokenDim <= 0 {
		return 0
	}
	if c.Tokens > 0 {
		return c.Tokens
	}
	return len(c.Keys) / tokenDim
}

// Validate checks that the captured query/keys/values form a well-shaped
// transformer-layer attention state.
func (c TransformerLayerCapture) Validate() error {
	if c.Heads <= 0 {
		return fmt.Errorf("turboquant: transformer capture heads must be > 0")
	}
	kvHeads := c.kvHeads()
	if kvHeads <= 0 {
		return fmt.Errorf("turboquant: transformer capture kv_heads must be > 0")
	}
	if c.Heads%kvHeads != 0 {
		return fmt.Errorf("turboquant: transformer capture heads %d must be divisible by kv_heads %d", c.Heads, kvHeads)
	}
	if err := validateDim(c.HeadDim); err != nil {
		return err
	}
	if err := ValidateVector(c.tokenDim(), c.Query); err != nil {
		return err
	}
	if len(c.Keys) != len(c.Values) {
		return fmt.Errorf("turboquant: transformer capture key/value payload length mismatch: %d vs %d", len(c.Keys), len(c.Values))
	}
	tokenDim := c.kvTokenDim()
	if len(c.Keys)%tokenDim != 0 {
		return fmt.Errorf("turboquant: transformer capture key/value payload length %d is not divisible by token dim %d", len(c.Keys), tokenDim)
	}
	tokens := len(c.Keys) / tokenDim
	if c.Tokens != 0 && c.Tokens != tokens {
		return fmt.Errorf("turboquant: transformer capture tokens = %d want %d from payload", c.Tokens, tokens)
	}
	for i, v := range c.Keys {
		if math.IsNaN(float64(v)) {
			return fmt.Errorf("turboquant: NaN in keys at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			return fmt.Errorf("turboquant: Inf in keys at index %d", i)
		}
	}
	for i, v := range c.Values {
		if math.IsNaN(float64(v)) {
			return fmt.Errorf("turboquant: NaN in values at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			return fmt.Errorf("turboquant: Inf in values at index %d", i)
		}
	}
	return nil
}

// BuildCache ingests the captured keys and values into a quantized
// TransformerLayerKVCache.
func (c TransformerLayerCapture) BuildCache(keyBits, valueBits, capacity int, seed int64) (*TransformerLayerKVCache, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}
	if err := validateIPBitWidth(keyBits); err != nil {
		return nil, err
	}
	if err := validateBitWidth(valueBits); err != nil {
		return nil, err
	}
	tokens := c.TokenCount()
	if capacity <= 0 || capacity < tokens {
		capacity = tokens
	}
	cache := NewTransformerLayerKVCacheWithSeed(c.kvHeads(), c.HeadDim, keyBits, valueBits, capacity, seed)
	for token := 0; token < tokens; token++ {
		cache.Append(c.keyToken(token), c.valueToken(token))
	}
	return cache, nil
}

// AttentionOutputFromCacheInto computes approximate per-head attention outputs
// from an already-built quantized cache. Query must be flattened head-major as
// [heads*headDim]. For GQA captures, the cache stores only kv_heads while the
// output is expanded back to query-head width.
func (c TransformerLayerCapture) AttentionOutputFromCacheInto(dst []float32, cache *TransformerLayerKVCache, query []float32, topK int) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if cache == nil {
		return fmt.Errorf("turboquant: nil transformer cache")
	}
	if cache.Heads() != c.kvHeads() {
		return fmt.Errorf("turboquant: transformer cache heads = %d want %d", cache.Heads(), c.kvHeads())
	}
	if cache.HeadDim() != c.HeadDim {
		return fmt.Errorf("turboquant: transformer cache head_dim = %d want %d", cache.HeadDim(), c.HeadDim)
	}
	if len(query) != c.tokenDim() {
		return fmt.Errorf("turboquant: expected transformer query length %d, got %d", c.tokenDim(), len(query))
	}
	if len(dst) != c.tokenDim() {
		return fmt.Errorf("turboquant: expected transformer attention destination length %d, got %d", c.tokenDim(), len(dst))
	}
	for i := range dst {
		dst[i] = 0
	}
	if topK <= 0 {
		return nil
	}
	topK = minInt(topK, cache.Len())
	if topK <= 0 {
		return nil
	}
	if c.kvHeads() == c.Heads {
		cache.AttentionOutputTo(dst, query, topK)
		return nil
	}
	groupedTransformerAttentionOutputTurboQuantInto(dst, cache, c, query, topK)
	return nil
}

// ReferenceAttentionOutputInto computes the exact per-head attention output for
// the captured query against the raw captured keys/values. queryScale multiplies
// every dot-product score before softmax; use 1.0 to match the current
// TurboQuant cache API, or 1/sqrt(headDim) to mirror scaled dot-product
// attention.
func (c TransformerLayerCapture) ReferenceAttentionOutputInto(dst []float32, queryScale float32) error {
	if err := c.Validate(); err != nil {
		return err
	}
	if len(dst) != c.tokenDim() {
		return fmt.Errorf("turboquant: expected reference attention destination length %d, got %d", c.tokenDim(), len(dst))
	}
	for i := range dst {
		dst[i] = 0
	}
	tokens := c.TokenCount()
	if tokens == 0 {
		return nil
	}
	if queryScale == 0 {
		queryScale = 1
	}

	scores := make([]float32, tokens)
	for head := 0; head < c.Heads; head++ {
		baseDim := head * c.HeadDim
		kvBaseDim := c.kvHeadForQueryHead(head) * c.HeadDim
		q := c.Query[baseDim : baseDim+c.HeadDim]
		maxScore := float32(math.Inf(-1))
		for token := 0; token < tokens; token++ {
			k := c.keyToken(token)[kvBaseDim : kvBaseDim+c.HeadDim]
			score := queryScale * dotFloat32(q, k)
			scores[token] = score
			if score > maxScore {
				maxScore = score
			}
		}
		var sum float64
		for token := 0; token < tokens; token++ {
			weight := math.Exp(float64(scores[token] - maxScore))
			scores[token] = float32(weight)
			sum += weight
		}
		if sum == 0 {
			continue
		}
		inv := float32(1 / sum)
		for token := 0; token < tokens; token++ {
			weight := scores[token] * inv
			v := c.valueToken(token)[kvBaseDim : kvBaseDim+c.HeadDim]
			for i := 0; i < c.HeadDim; i++ {
				dst[baseDim+i] += weight * v[i]
			}
		}
	}
	return nil
}

// EvaluateTransformerLayerCapture ingests a captured transformer layer into a
// quantized cache and reports attention-output reconstruction metrics against
// the exact raw attention output.
func EvaluateTransformerLayerCapture(capture TransformerLayerCapture, cfg TransformerLayerEvalConfig) (TransformerLayerEvalResult, error) {
	if err := capture.Validate(); err != nil {
		return TransformerLayerEvalResult{}, err
	}
	method, err := normalizeTransformerEvalMethod(cfg.Method)
	if err != nil {
		return TransformerLayerEvalResult{}, err
	}
	cfg.Method = method
	if cfg.QueryScale == 0 {
		cfg.QueryScale = capture.QueryScale
		if cfg.QueryScale == 0 {
			cfg.QueryScale = 1
		}
	}
	query := append([]float32(nil), capture.Query...)
	if cfg.QueryScale != 1 {
		for i := range query {
			query[i] *= cfg.QueryScale
		}
	}

	exact := make([]float32, capture.tokenDim())
	if err := capture.ReferenceAttentionOutputInto(exact, cfg.QueryScale); err != nil {
		return TransformerLayerEvalResult{}, err
	}
	topK := cfg.TopK
	if topK <= 0 || topK > capture.TokenCount() {
		topK = capture.TokenCount()
	}
	var approxEval transformerApproxEval
	switch cfg.Method {
	case TransformerEvalMethodTurboQuant:
		approxEval, err = evaluateTransformerCaptureTurboQuant(capture, cfg, query, topK)
	case TransformerEvalMethodUniform:
		approxEval, err = evaluateTransformerCaptureUniform(capture, cfg, query, topK)
	default:
		err = fmt.Errorf("turboquant: unsupported transformer eval method %q", cfg.Method)
	}
	if err != nil {
		return TransformerLayerEvalResult{}, err
	}

	result := TransformerLayerEvalResult{
		Model:             capture.Model,
		Name:              capture.Name,
		PromptIndex:       capture.PromptIndex,
		Layer:             capture.Layer,
		TokenIndex:        capture.TokenIndex,
		TokenPosition:     capture.TokenPosition,
		SequenceLength:    capture.SequenceLength,
		Tokens:            capture.TokenCount(),
		Heads:             capture.Heads,
		KVHeads:           capture.kvHeads(),
		HeadDim:           capture.HeadDim,
		Method:            cfg.Method,
		KeyBits:           cfg.KeyBits,
		ValueBits:         cfg.ValueBits,
		RequestedTopK:     cfg.TopK,
		TopK:              topK,
		Capacity:          maxInt(cfg.Capacity, capture.TokenCount()),
		QueryScale:        cfg.QueryScale,
		GPUEnabled:        approxEval.gpuEnabled,
		RawKVBytes:        uint64(len(capture.Keys)+len(capture.Values)) * 4,
		CacheLiveBytes:    approxEval.liveBytes,
		CacheStorageBytes: approxEval.storageBytes,
		OutputMSE:         vectorMSEFloat64(exact, approxEval.output),
		OutputCosine:      vectorCosineFloat64(exact, approxEval.output),
		HeadsMetrics:      make([]TransformerLayerHeadEval, 0, capture.Heads),
	}
	if result.CacheStorageBytes != 0 {
		result.CompressionRatio = float64(result.RawKVBytes) / float64(result.CacheStorageBytes)
	}
	for head := 0; head < capture.Heads; head++ {
		base := head * capture.HeadDim
		exactHead := exact[base : base+capture.HeadDim]
		approxHead := approxEval.output[base : base+capture.HeadDim]
		result.HeadsMetrics = append(result.HeadsMetrics, TransformerLayerHeadEval{
			Head:         head,
			OutputMSE:    vectorMSEFloat64(exactHead, approxHead),
			OutputCosine: vectorCosineFloat64(exactHead, approxHead),
			ExactNorm:    vectorNormFloat64(exactHead),
			QuantNorm:    vectorNormFloat64(approxHead),
		})
	}
	return result, nil
}

func (c TransformerLayerCapture) keyToken(token int) []float32 {
	tokenDim := c.kvTokenDim()
	base := token * tokenDim
	return c.Keys[base : base+tokenDim]
}

func (c TransformerLayerCapture) valueToken(token int) []float32 {
	tokenDim := c.kvTokenDim()
	base := token * tokenDim
	return c.Values[base : base+tokenDim]
}

func dotFloat32(left, right []float32) float32 {
	var sum float32
	for i := range left {
		sum += left[i] * right[i]
	}
	return sum
}

func vectorNormFloat64(values []float32) float64 {
	var sum float64
	for _, value := range values {
		sum += float64(value) * float64(value)
	}
	return math.Sqrt(sum)
}

func vectorMSEFloat64(left, right []float32) float64 {
	if len(left) == 0 {
		return 0
	}
	var sum float64
	for i := range left {
		diff := float64(left[i] - right[i])
		sum += diff * diff
	}
	return sum / float64(len(left))
}

func vectorCosineFloat64(left, right []float32) float64 {
	var dot float64
	var leftNorm float64
	var rightNorm float64
	for i := range left {
		dot += float64(left[i]) * float64(right[i])
		leftNorm += float64(left[i]) * float64(left[i])
		rightNorm += float64(right[i]) * float64(right[i])
	}
	if leftNorm == 0 || rightNorm == 0 {
		return 0
	}
	return dot / math.Sqrt(leftNorm*rightNorm)
}
