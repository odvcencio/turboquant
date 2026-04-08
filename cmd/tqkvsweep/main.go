package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	turboquant "github.com/odvcencio/turboquant"
)

type sampleReport struct {
	Model          string                                  `json:"model,omitempty"`
	Name           string                                  `json:"name,omitempty"`
	PromptIndex    int                                     `json:"prompt_index,omitempty"`
	Layer          int                                     `json:"layer,omitempty"`
	TokenIndex     int                                     `json:"token_index,omitempty"`
	TokenPosition  string                                  `json:"token_position,omitempty"`
	SequenceLength int                                     `json:"sequence_length,omitempty"`
	Tokens         int                                     `json:"tokens"`
	Heads          int                                     `json:"heads"`
	KVHeads        int                                     `json:"kv_heads,omitempty"`
	HeadDim        int                                     `json:"head_dim"`
	Cases          []turboquant.TransformerLayerEvalResult `json:"cases"`
	BestByMSE      turboquant.TransformerLayerEvalResult   `json:"best_by_mse"`
	BestByCosine   turboquant.TransformerLayerEvalResult   `json:"best_by_cosine"`
	SmallestCache  turboquant.TransformerLayerEvalResult   `json:"smallest_cache"`
}

type configSummary struct {
	Method               string  `json:"method"`
	KeyBits              int     `json:"key_bits"`
	ValueBits            int     `json:"value_bits"`
	TopK                 int     `json:"top_k"`
	Samples              int     `json:"samples"`
	MeanOutputMSE        float64 `json:"mean_output_mse"`
	P50OutputMSE         float64 `json:"p50_output_mse"`
	P95OutputMSE         float64 `json:"p95_output_mse"`
	MaxOutputMSE         float64 `json:"max_output_mse"`
	MeanOutputCosine     float64 `json:"mean_output_cosine"`
	P05OutputCosine      float64 `json:"p05_output_cosine"`
	P50OutputCosine      float64 `json:"p50_output_cosine"`
	MinOutputCosine      float64 `json:"min_output_cosine"`
	MeanRawKVBytes       float64 `json:"mean_raw_kv_bytes"`
	MeanCacheLiveBytes   float64 `json:"mean_cache_live_bytes"`
	MeanStorageBytes     float64 `json:"mean_cache_storage_bytes"`
	MeanCompressionRatio float64 `json:"mean_compression_ratio"`
	BestMSEWins          int     `json:"best_mse_wins"`
	BestCosineWins       int     `json:"best_cosine_wins"`
	SmallestCacheWins    int     `json:"smallest_cache_wins"`
}

type report struct {
	GeneratedAt    string          `json:"generated_at"`
	Samples        []sampleReport  `json:"samples"`
	Configurations []configSummary `json:"configurations"`
	ParetoFrontier []configSummary `json:"pareto_frontier"`
}

type configKey struct {
	Method    string
	KeyBits   int
	ValueBits int
	TopK      int
}

type configAccumulator struct {
	mse               []float64
	cosine            []float64
	sumRawKVBytes     float64
	sumCacheLiveBytes float64
	sumStorageBytes   float64
	sumCompression    float64
	bestMSEWins       int
	bestCosineWins    int
	smallestCacheWins int
}

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runCLI(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("tqkvsweep", flag.ContinueOnError)
	fs.SetOutput(stderr)

	inputPath := fs.String("input", "", "path to a JSON TransformerLayerCapture or TransformerLayerCaptureFile")
	outputPath := fs.String("out", "", "optional path to write the JSON report")
	methodsCSV := fs.String("methods", turboquant.TransformerEvalMethodTurboQuant, "comma-separated evaluation methods: turboquant,uniform")
	keyBitsCSV := fs.String("key-bits", "2,3,4", "comma-separated key bit widths")
	valueBitsCSV := fs.String("value-bits", "2,3,4", "comma-separated value bit widths")
	topKCSV := fs.String("top-k", "8,16,32", "comma-separated top-k values")
	capacity := fs.Int("capacity", 0, "optional token capacity override for the quantized cache")
	seed := fs.Int64("seed", 42, "seed for deterministic quantizer construction")
	queryScale := fs.Float64("query-scale", 0, "multiplier applied to query dot products before softmax; default 0 uses each capture's query_scale or 1")
	tryGPU := fs.Bool("gpu", false, "try the experimental GPU path when available")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *inputPath == "" {
		return errors.New("tqkvsweep: --input is required")
	}

	methods, err := parseCSVStrings(*methodsCSV)
	if err != nil {
		return fmt.Errorf("tqkvsweep: methods: %w", err)
	}
	keyBits, err := parseCSVInts(*keyBitsCSV)
	if err != nil {
		return fmt.Errorf("tqkvsweep: key bits: %w", err)
	}
	valueBits, err := parseCSVInts(*valueBitsCSV)
	if err != nil {
		return fmt.Errorf("tqkvsweep: value bits: %w", err)
	}
	topKs, err := parseCSVInts(*topKCSV)
	if err != nil {
		return fmt.Errorf("tqkvsweep: top-k: %w", err)
	}
	out := report{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Samples:     make([]sampleReport, 0),
	}
	err = turboquant.WalkTransformerLayerCapturesFile(*inputPath, func(sample turboquant.TransformerLayerCapture) error {
		sampleRep, err := runSample(sample, methods, keyBits, valueBits, topKs, *capacity, *seed, float32(*queryScale), *tryGPU)
		if err != nil {
			return err
		}
		out.Samples = append(out.Samples, sampleRep)
		return nil
	})
	if err != nil {
		return fmt.Errorf("tqkvsweep: decode input: %w", err)
	}
	out.Configurations = summarizeConfigurations(out.Samples)
	out.ParetoFrontier = paretoFrontier(out.Configurations)

	if *outputPath == "" {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(out)
	}
	f, err := os.Create(*outputPath)
	if err != nil {
		return fmt.Errorf("tqkvsweep: write report: %w", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(out); err != nil {
		return fmt.Errorf("tqkvsweep: write report: %w", err)
	}
	_, err = fmt.Fprintf(stdout, "wrote %s\n", *outputPath)
	return err
}

func runSample(sample turboquant.TransformerLayerCapture, methods []string, keyBits, valueBits, topKs []int, capacity int, seed int64, queryScale float32, tryGPU bool) (sampleReport, error) {
	cases := make([]turboquant.TransformerLayerEvalResult, 0, len(methods)*len(keyBits)*len(valueBits)*len(topKs))
	for _, method := range methods {
		for _, kb := range keyBits {
			for _, vb := range valueBits {
				for _, topK := range topKs {
					result, err := turboquant.EvaluateTransformerLayerCapture(sample, turboquant.TransformerLayerEvalConfig{
						Method:     method,
						KeyBits:    kb,
						ValueBits:  vb,
						TopK:       topK,
						Capacity:   capacity,
						Seed:       seed,
						QueryScale: queryScale,
						TryGPU:     tryGPU,
					})
					if err != nil {
						return sampleReport{}, err
					}
					cases = append(cases, result)
				}
			}
		}
	}
	sort.SliceStable(cases, func(i, j int) bool {
		return lessByConfig(cases[i], cases[j])
	})
	kvHeads := sample.KVHeads
	if kvHeads == 0 {
		kvHeads = sample.Heads
	}
	rep := sampleReport{
		Model:          sample.Model,
		Name:           sample.Name,
		PromptIndex:    sample.PromptIndex,
		Layer:          sample.Layer,
		TokenIndex:     sample.TokenIndex,
		TokenPosition:  sample.TokenPosition,
		SequenceLength: sample.SequenceLength,
		Tokens:         sample.TokenCount(),
		Heads:          sample.Heads,
		KVHeads:        kvHeads,
		HeadDim:        sample.HeadDim,
		Cases:          cases,
	}
	if len(cases) > 0 {
		rep.BestByMSE = bestResult(cases, func(left, right turboquant.TransformerLayerEvalResult) bool {
			if left.OutputMSE != right.OutputMSE {
				return left.OutputMSE < right.OutputMSE
			}
			if left.CacheLiveBytes != right.CacheLiveBytes {
				return left.CacheLiveBytes < right.CacheLiveBytes
			}
			return lessByConfig(left, right)
		})
		rep.BestByCosine = bestResult(cases, func(left, right turboquant.TransformerLayerEvalResult) bool {
			if left.OutputCosine != right.OutputCosine {
				return left.OutputCosine > right.OutputCosine
			}
			if left.CacheLiveBytes != right.CacheLiveBytes {
				return left.CacheLiveBytes < right.CacheLiveBytes
			}
			return lessByConfig(left, right)
		})
		rep.SmallestCache = bestResult(cases, func(left, right turboquant.TransformerLayerEvalResult) bool {
			if left.CacheLiveBytes != right.CacheLiveBytes {
				return left.CacheLiveBytes < right.CacheLiveBytes
			}
			if left.OutputMSE != right.OutputMSE {
				return left.OutputMSE < right.OutputMSE
			}
			return lessByConfig(left, right)
		})
	}
	return rep, nil
}

func bestResult(results []turboquant.TransformerLayerEvalResult, less func(left, right turboquant.TransformerLayerEvalResult) bool) turboquant.TransformerLayerEvalResult {
	best := results[0]
	for _, item := range results[1:] {
		if less(item, best) {
			best = item
		}
	}
	return best
}

func lessByConfig(left, right turboquant.TransformerLayerEvalResult) bool {
	if left.Method != right.Method {
		return left.Method < right.Method
	}
	if left.KeyBits != right.KeyBits {
		return left.KeyBits < right.KeyBits
	}
	if left.ValueBits != right.ValueBits {
		return left.ValueBits < right.ValueBits
	}
	return left.TopK < right.TopK
}

func parseCSVInts(raw string) ([]int, error) {
	parts := strings.Split(raw, ",")
	values := make([]int, 0, len(parts))
	seen := make(map[int]struct{}, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		value, err := strconv.Atoi(part)
		if err != nil {
			return nil, err
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		values = append(values, value)
	}
	if len(values) == 0 {
		return nil, errors.New("expected at least one integer")
	}
	return values, nil
}

func parseCSVStrings(raw string) ([]string, error) {
	parts := strings.Split(raw, ",")
	values := make([]string, 0, len(parts))
	seen := make(map[string]struct{}, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		method, err := turboquant.NormalizeTransformerEvalMethodForCLI(part)
		if err != nil {
			return nil, err
		}
		if _, ok := seen[method]; ok {
			continue
		}
		seen[method] = struct{}{}
		values = append(values, method)
	}
	if len(values) == 0 {
		return nil, errors.New("expected at least one method")
	}
	return values, nil
}

func summarizeConfigurations(samples []sampleReport) []configSummary {
	if len(samples) == 0 {
		return nil
	}
	accumulators := make(map[configKey]*configAccumulator)
	for _, sample := range samples {
		for _, item := range sample.Cases {
			key := configKeyOf(item)
			acc := accumulators[key]
			if acc == nil {
				acc = &configAccumulator{}
				accumulators[key] = acc
			}
			acc.mse = append(acc.mse, item.OutputMSE)
			acc.cosine = append(acc.cosine, item.OutputCosine)
			acc.sumRawKVBytes += float64(item.RawKVBytes)
			acc.sumCacheLiveBytes += float64(item.CacheLiveBytes)
			acc.sumStorageBytes += float64(item.CacheStorageBytes)
			acc.sumCompression += item.CompressionRatio
		}
		if len(sample.Cases) == 0 {
			continue
		}
		accumulators[configKeyOf(sample.BestByMSE)].bestMSEWins++
		accumulators[configKeyOf(sample.BestByCosine)].bestCosineWins++
		accumulators[configKeyOf(sample.SmallestCache)].smallestCacheWins++
	}

	keys := make([]configKey, 0, len(accumulators))
	for key := range accumulators {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].Method != keys[j].Method {
			return keys[i].Method < keys[j].Method
		}
		if keys[i].KeyBits != keys[j].KeyBits {
			return keys[i].KeyBits < keys[j].KeyBits
		}
		if keys[i].ValueBits != keys[j].ValueBits {
			return keys[i].ValueBits < keys[j].ValueBits
		}
		return keys[i].TopK < keys[j].TopK
	})

	out := make([]configSummary, 0, len(keys))
	for _, key := range keys {
		acc := accumulators[key]
		samplesCount := len(acc.mse)
		out = append(out, configSummary{
			Method:               key.Method,
			KeyBits:              key.KeyBits,
			ValueBits:            key.ValueBits,
			TopK:                 key.TopK,
			Samples:              samplesCount,
			MeanOutputMSE:        meanFloat64(acc.mse),
			P50OutputMSE:         percentile(acc.mse, 0.50),
			P95OutputMSE:         percentile(acc.mse, 0.95),
			MaxOutputMSE:         maxFloat64(acc.mse),
			MeanOutputCosine:     meanFloat64(acc.cosine),
			P05OutputCosine:      percentile(acc.cosine, 0.05),
			P50OutputCosine:      percentile(acc.cosine, 0.50),
			MinOutputCosine:      minFloat64(acc.cosine),
			MeanRawKVBytes:       safeMeanSum(acc.sumRawKVBytes, samplesCount),
			MeanCacheLiveBytes:   safeMeanSum(acc.sumCacheLiveBytes, samplesCount),
			MeanStorageBytes:     safeMeanSum(acc.sumStorageBytes, samplesCount),
			MeanCompressionRatio: safeMeanSum(acc.sumCompression, samplesCount),
			BestMSEWins:          acc.bestMSEWins,
			BestCosineWins:       acc.bestCosineWins,
			SmallestCacheWins:    acc.smallestCacheWins,
		})
	}
	return out
}

func paretoFrontier(configs []configSummary) []configSummary {
	if len(configs) == 0 {
		return nil
	}
	frontier := make([]configSummary, 0, len(configs))
	for i, candidate := range configs {
		dominated := false
		for j, other := range configs {
			if i == j {
				continue
			}
			if dominates(other, candidate) {
				dominated = true
				break
			}
		}
		if !dominated {
			frontier = append(frontier, candidate)
		}
	}
	sort.Slice(frontier, func(i, j int) bool {
		if frontier[i].MeanStorageBytes != frontier[j].MeanStorageBytes {
			return frontier[i].MeanStorageBytes < frontier[j].MeanStorageBytes
		}
		if frontier[i].MeanOutputMSE != frontier[j].MeanOutputMSE {
			return frontier[i].MeanOutputMSE < frontier[j].MeanOutputMSE
		}
		if frontier[i].MeanOutputCosine != frontier[j].MeanOutputCosine {
			return frontier[i].MeanOutputCosine > frontier[j].MeanOutputCosine
		}
		if frontier[i].Method != frontier[j].Method {
			return frontier[i].Method < frontier[j].Method
		}
		if frontier[i].KeyBits != frontier[j].KeyBits {
			return frontier[i].KeyBits < frontier[j].KeyBits
		}
		if frontier[i].ValueBits != frontier[j].ValueBits {
			return frontier[i].ValueBits < frontier[j].ValueBits
		}
		return frontier[i].TopK < frontier[j].TopK
	})
	return frontier
}

func dominates(left, right configSummary) bool {
	if left.MeanStorageBytes > right.MeanStorageBytes {
		return false
	}
	if left.MeanOutputMSE > right.MeanOutputMSE {
		return false
	}
	if left.MeanStorageBytes < right.MeanStorageBytes || left.MeanOutputMSE < right.MeanOutputMSE {
		return true
	}
	return left.MeanOutputCosine > right.MeanOutputCosine
}

func configKeyOf(result turboquant.TransformerLayerEvalResult) configKey {
	topK := result.RequestedTopK
	if topK == 0 {
		topK = result.TopK
	}
	return configKey{
		Method:    defaultMethod(result.Method),
		KeyBits:   result.KeyBits,
		ValueBits: result.ValueBits,
		TopK:      topK,
	}
}

func defaultMethod(method string) string {
	normalized, err := turboquant.NormalizeTransformerEvalMethodForCLI(method)
	if err != nil {
		return turboquant.TransformerEvalMethodTurboQuant
	}
	return normalized
}

func safeMeanSum(sum float64, count int) float64 {
	if count == 0 {
		return 0
	}
	return sum / float64(count)
}

func meanFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func maxFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	maxValue := values[0]
	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}
	return maxValue
}

func minFloat64(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	minValue := values[0]
	for _, value := range values[1:] {
		if value < minValue {
			minValue = value
		}
	}
	return minValue
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	if p <= 0 {
		return minFloat64(values)
	}
	if p >= 1 {
		return maxFloat64(values)
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	index := int(p * float64(len(sorted)-1))
	return sorted[index]
}
