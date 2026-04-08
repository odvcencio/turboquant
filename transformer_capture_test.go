package turboquant

import (
	"math"
	"testing"
)

func TestTransformerLayerCaptureBuildCache(t *testing.T) {
	capture := TransformerLayerCapture{
		Name:    "layer0-step1",
		Heads:   2,
		HeadDim: 4,
		Query:   []float32{1, 0, 0, 0, 0, 1, 0, 0},
		Keys: []float32{
			1, 0, 0, 0, 0, 1, 0, 0,
			0, 1, 0, 0, 1, 0, 0, 0,
		},
		Values: []float32{
			2, 0, 0, 0, 0, 3, 0, 0,
			0, 4, 0, 0, 5, 0, 0, 0,
		},
	}
	cache, err := capture.BuildCache(3, 2, 0, 42)
	if err != nil {
		t.Fatalf("BuildCache: %v", err)
	}
	if cache.Heads() != capture.Heads {
		t.Fatalf("Heads = %d want %d", cache.Heads(), capture.Heads)
	}
	if cache.HeadDim() != capture.HeadDim {
		t.Fatalf("HeadDim = %d want %d", cache.HeadDim(), capture.HeadDim)
	}
	if cache.Len() != capture.TokenCount() {
		t.Fatalf("Len = %d want %d", cache.Len(), capture.TokenCount())
	}
	if cache.LiveBytes() == 0 {
		t.Fatal("LiveBytes = 0 want > 0")
	}
}

func TestTransformerLayerCaptureReferenceAttentionOutputInto(t *testing.T) {
	capture := TransformerLayerCapture{
		Heads:   1,
		HeadDim: 2,
		Query:   []float32{1, 0},
		Keys: []float32{
			1, 0,
			0, 1,
		},
		Values: []float32{
			2, 0,
			0, 4,
		},
	}
	got := make([]float32, 2)
	if err := capture.ReferenceAttentionOutputInto(got, 1); err != nil {
		t.Fatalf("ReferenceAttentionOutputInto: %v", err)
	}
	weight0 := float32(math.Exp(1) / (math.Exp(1) + math.Exp(0)))
	weight1 := float32(math.Exp(0) / (math.Exp(1) + math.Exp(0)))
	want := []float32{2 * weight0, 4 * weight1}
	for i := range want {
		if !closeKVFloat32(got[i], want[i], 1e-5) {
			t.Fatalf("output[%d] = %v want %v", i, got[i], want[i])
		}
	}
}

func TestEvaluateTransformerLayerCapture(t *testing.T) {
	capture := TransformerLayerCapture{
		Name:       "layer0-step2",
		Heads:      2,
		HeadDim:    4,
		QueryScale: 0.5,
		Query:      []float32{1, 0, 0, 0, 0, 1, 0, 0},
		Keys: []float32{
			1, 0, 0, 0, 0, 1, 0, 0,
			0, 1, 0, 0, 1, 0, 0, 0,
			1, 1, 0, 0, 1, 1, 0, 0,
		},
		Values: []float32{
			2, 0, 0, 0, 0, 3, 0, 0,
			0, 4, 0, 0, 5, 0, 0, 0,
			1, 1, 0, 0, 1, 1, 0, 0,
		},
	}
	result, err := EvaluateTransformerLayerCapture(capture, TransformerLayerEvalConfig{
		KeyBits:   3,
		ValueBits: 2,
		TopK:      2,
		Seed:      42,
	})
	if err != nil {
		t.Fatalf("EvaluateTransformerLayerCapture: %v", err)
	}
	if result.Name != capture.Name {
		t.Fatalf("Name = %q want %q", result.Name, capture.Name)
	}
	if result.Method != TransformerEvalMethodTurboQuant {
		t.Fatalf("Method = %q want %q", result.Method, TransformerEvalMethodTurboQuant)
	}
	if result.Tokens != capture.TokenCount() {
		t.Fatalf("Tokens = %d want %d", result.Tokens, capture.TokenCount())
	}
	if result.TopK != 2 {
		t.Fatalf("TopK = %d want 2", result.TopK)
	}
	if result.RequestedTopK != 2 {
		t.Fatalf("RequestedTopK = %d want 2", result.RequestedTopK)
	}
	if result.QueryScale != 0.5 {
		t.Fatalf("QueryScale = %v want 0.5", result.QueryScale)
	}
	if result.RawKVBytes != uint64(len(capture.Keys)+len(capture.Values))*4 {
		t.Fatalf("RawKVBytes = %d want %d", result.RawKVBytes, uint64(len(capture.Keys)+len(capture.Values))*4)
	}
	if len(result.HeadsMetrics) != capture.Heads {
		t.Fatalf("HeadsMetrics = %d want %d", len(result.HeadsMetrics), capture.Heads)
	}
	if result.CacheLiveBytes == 0 || result.CacheStorageBytes < result.CacheLiveBytes {
		t.Fatalf("cache bytes = (%d,%d)", result.CacheLiveBytes, result.CacheStorageBytes)
	}
	if result.CompressionRatio <= 0 {
		t.Fatalf("CompressionRatio = %v want > 0", result.CompressionRatio)
	}
	if result.OutputCosine < -1 || result.OutputCosine > 1 {
		t.Fatalf("OutputCosine = %v want [-1,1]", result.OutputCosine)
	}
	if result.OutputMSE < 0 {
		t.Fatalf("OutputMSE = %v want >= 0", result.OutputMSE)
	}
}

func TestEvaluateTransformerLayerCaptureUsesExplicitQueryScaleOverride(t *testing.T) {
	capture := TransformerLayerCapture{
		Heads:      1,
		HeadDim:    2,
		QueryScale: 0.5,
		Query:      []float32{1, 0},
		Keys: []float32{
			1, 0,
			0, 1,
		},
		Values: []float32{
			2, 0,
			0, 4,
		},
	}
	result, err := EvaluateTransformerLayerCapture(capture, TransformerLayerEvalConfig{
		KeyBits:    2,
		ValueBits:  2,
		TopK:       2,
		QueryScale: 1,
	})
	if err != nil {
		t.Fatalf("EvaluateTransformerLayerCapture: %v", err)
	}
	if result.QueryScale != 1 {
		t.Fatalf("QueryScale = %v want 1", result.QueryScale)
	}
}

func TestEvaluateTransformerLayerCaptureTryGPUFallsBackForUnsupportedPreparedMSEBitWidth(t *testing.T) {
	capture := TransformerLayerCapture{
		Name:       "gpu-fallback",
		Heads:      1,
		HeadDim:    4,
		QueryScale: 1,
		Query:      []float32{1, 0, 0, 0},
		Keys: []float32{
			1, 0, 0, 0,
			0, 1, 0, 0,
		},
		Values: []float32{
			2, 0, 0, 0,
			0, 3, 0, 0,
		},
	}
	result, err := EvaluateTransformerLayerCapture(capture, TransformerLayerEvalConfig{
		KeyBits:   4,
		ValueBits: 2,
		TopK:      2,
		Seed:      42,
		TryGPU:    true,
	})
	if err != nil {
		t.Fatalf("EvaluateTransformerLayerCapture(TryGPU fallback): %v", err)
	}
	if result.GPUEnabled {
		t.Fatal("GPUEnabled = true want false for unsupported prepared MSE LUT config")
	}
}

func TestEvaluateTransformerLayerCaptureUniformBaseline(t *testing.T) {
	capture := TransformerLayerCapture{
		Name:           "uniform-step",
		Model:          "tiny",
		PromptIndex:    1,
		Layer:          3,
		TokenIndex:     7,
		TokenPosition:  "last",
		SequenceLength: 8,
		Heads:          1,
		HeadDim:        2,
		QueryScale:     1,
		Query:          []float32{1, 0},
		Keys: []float32{
			1, 0,
			0, 1,
		},
		Values: []float32{
			2, 0,
			0, 4,
		},
	}
	result, err := EvaluateTransformerLayerCapture(capture, TransformerLayerEvalConfig{
		Method:    TransformerEvalMethodUniform,
		KeyBits:   2,
		ValueBits: 2,
		TopK:      2,
	})
	if err != nil {
		t.Fatalf("EvaluateTransformerLayerCapture(uniform): %v", err)
	}
	if result.Method != TransformerEvalMethodUniform {
		t.Fatalf("Method = %q want %q", result.Method, TransformerEvalMethodUniform)
	}
	if result.Model != capture.Model || result.PromptIndex != capture.PromptIndex || result.Layer != capture.Layer || result.TokenIndex != capture.TokenIndex {
		t.Fatalf("metadata = (%q,%d,%d,%d) want (%q,%d,%d,%d)", result.Model, result.PromptIndex, result.Layer, result.TokenIndex, capture.Model, capture.PromptIndex, capture.Layer, capture.TokenIndex)
	}
	if result.TokenPosition != capture.TokenPosition || result.SequenceLength != capture.SequenceLength {
		t.Fatalf("token metadata = (%q,%d) want (%q,%d)", result.TokenPosition, result.SequenceLength, capture.TokenPosition, capture.SequenceLength)
	}
	wantBytes := uniformTransformerLayerBytes(capture.Heads, capture.HeadDim, 2, 2, capture.TokenCount())
	if result.CacheLiveBytes != wantBytes || result.CacheStorageBytes != wantBytes {
		t.Fatalf("bytes = (%d,%d) want (%d,%d)", result.CacheLiveBytes, result.CacheStorageBytes, wantBytes, wantBytes)
	}
	if result.OutputMSE < 0 {
		t.Fatalf("OutputMSE = %v want >= 0", result.OutputMSE)
	}
}

func TestTransformerLayerCaptureGroupedKVHeadsReferenceAttentionOutputInto(t *testing.T) {
	capture := TransformerLayerCapture{
		Heads:   4,
		KVHeads: 2,
		HeadDim: 2,
		Query: []float32{
			1, 0,
			0, 1,
			1, 0,
			0, 1,
		},
		Keys: []float32{
			1, 0, 0, 1,
			0, 1, 1, 0,
		},
		Values: []float32{
			10, 0, 0, 20,
			0, 30, 40, 0,
		},
	}
	got := make([]float32, len(capture.Query))
	if err := capture.ReferenceAttentionOutputInto(got, 1); err != nil {
		t.Fatalf("ReferenceAttentionOutputInto: %v", err)
	}
	hi := float32(math.Exp(1) / (math.Exp(1) + math.Exp(0)))
	lo := float32(math.Exp(0) / (math.Exp(1) + math.Exp(0)))
	want := []float32{
		10 * hi, 30 * lo,
		10 * lo, 30 * hi,
		40 * hi, 20 * lo,
		40 * lo, 20 * hi,
	}
	for i := range want {
		if !closeKVFloat32(got[i], want[i], 1e-5) {
			t.Fatalf("output[%d] = %v want %v", i, got[i], want[i])
		}
	}
}

func TestEvaluateTransformerLayerCaptureUniformGroupedKVHeads(t *testing.T) {
	capture := TransformerLayerCapture{
		Heads:   4,
		KVHeads: 2,
		HeadDim: 2,
		Query: []float32{
			1, 0,
			0, 1,
			1, 0,
			0, 1,
		},
		Keys: []float32{
			1, 0, 0, 1,
			0, 1, 1, 0,
		},
		Values: []float32{
			10, 0, 0, 20,
			0, 30, 40, 0,
		},
	}
	cache, err := capture.BuildCache(2, 2, 0, 42)
	if err != nil {
		t.Fatalf("BuildCache: %v", err)
	}
	if cache.Heads() != 2 {
		t.Fatalf("cache.Heads = %d want 2", cache.Heads())
	}

	result, err := EvaluateTransformerLayerCapture(capture, TransformerLayerEvalConfig{
		Method:    TransformerEvalMethodUniform,
		KeyBits:   2,
		ValueBits: 2,
		TopK:      2,
	})
	if err != nil {
		t.Fatalf("EvaluateTransformerLayerCapture(uniform): %v", err)
	}
	if result.Heads != 4 {
		t.Fatalf("result.Heads = %d want 4", result.Heads)
	}
	if result.KVHeads != 2 {
		t.Fatalf("result.KVHeads = %d want 2", result.KVHeads)
	}
	if len(result.HeadsMetrics) != 4 {
		t.Fatalf("len(result.HeadsMetrics) = %d want 4", len(result.HeadsMetrics))
	}
	wantBytes := uniformTransformerLayerBytes(2, 2, 2, 2, capture.TokenCount())
	if result.CacheLiveBytes != wantBytes {
		t.Fatalf("CacheLiveBytes = %d want %d", result.CacheLiveBytes, wantBytes)
	}
}

func TestTransformerLayerCaptureValidateRejectsWrongTokenShape(t *testing.T) {
	capture := TransformerLayerCapture{
		Heads:   2,
		HeadDim: 4,
		Query:   make([]float32, 8),
		Keys:    make([]float32, 9),
		Values:  make([]float32, 9),
	}
	if err := capture.Validate(); err == nil {
		t.Fatal("Validate() = nil want error")
	}
}
