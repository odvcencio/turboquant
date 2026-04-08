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

type sweepReport struct {
	GeneratedAt    string          `json:"generated_at"`
	Samples        []sweepSample   `json:"samples"`
	ParetoFrontier []summaryConfig `json:"pareto_frontier"`
}

type sweepSample struct {
	Model          string                                  `json:"model,omitempty"`
	Name           string                                  `json:"name,omitempty"`
	PromptIndex    int                                     `json:"prompt_index,omitempty"`
	Layer          int                                     `json:"layer,omitempty"`
	TokenIndex     int                                     `json:"token_index,omitempty"`
	TokenPosition  string                                  `json:"token_position,omitempty"`
	SequenceLength int                                     `json:"sequence_length,omitempty"`
	Tokens         int                                     `json:"tokens"`
	Cases          []turboquant.TransformerLayerEvalResult `json:"cases"`
}

type summaryConfig struct {
	Method                string  `json:"method"`
	KeyBits               int     `json:"key_bits"`
	ValueBits             int     `json:"value_bits"`
	TopK                  int     `json:"top_k"`
	Samples               int     `json:"samples"`
	MeanOutputMSE         float64 `json:"mean_output_mse"`
	MeanOutputCosine      float64 `json:"mean_output_cosine"`
	MeanCacheStorageBytes float64 `json:"mean_cache_storage_bytes"`
	MeanCompressionRatio  float64 `json:"mean_compression_ratio"`
}

type groupSummary struct {
	Samples        int             `json:"samples"`
	Tokens         []int           `json:"tokens,omitempty"`
	BestByMSE      summaryConfig   `json:"best_by_mse"`
	ParetoFrontier []summaryConfig `json:"pareto_frontier"`
}

type summaryReport struct {
	GeneratedAt           string                  `json:"generated_at"`
	InputGeneratedAt      string                  `json:"input_generated_at,omitempty"`
	OverallParetoFrontier []summaryConfig         `json:"overall_pareto_frontier"`
	ByMethod              map[string]groupSummary `json:"by_method,omitempty"`
	ByLayer               map[string]groupSummary `json:"by_layer,omitempty"`
	ByTokenIndex          map[string]groupSummary `json:"by_token_index,omitempty"`
	ByRelativePosition    map[string]groupSummary `json:"by_relative_position,omitempty"`
}

type configKey struct {
	Method    string
	KeyBits   int
	ValueBits int
	TopK      int
}

type configAccumulator struct {
	sumMSE         float64
	sumCosine      float64
	sumStorage     float64
	sumCompression float64
	count          int
}

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runCLI(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("tqkvsummarize", flag.ContinueOnError)
	fs.SetOutput(stderr)

	inputPath := fs.String("input", "", "path to a tqkvsweep JSON report")
	outputPath := fs.String("out", "", "optional path to write the JSON summary")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *inputPath == "" {
		return errors.New("tqkvsummarize: --input is required")
	}

	report, err := loadReport(*inputPath)
	if err != nil {
		return err
	}
	summary := summarizeReport(report)

	payload, err := json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return fmt.Errorf("tqkvsummarize: encode summary: %w", err)
	}
	if *outputPath == "" {
		_, err = stdout.Write(append(payload, '\n'))
		return err
	}
	if err := os.WriteFile(*outputPath, append(payload, '\n'), 0o644); err != nil {
		return fmt.Errorf("tqkvsummarize: write summary: %w", err)
	}
	_, err = fmt.Fprintf(stdout, "wrote %s\n", *outputPath)
	return err
}

func loadReport(path string) (sweepReport, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return sweepReport{}, fmt.Errorf("tqkvsummarize: read input: %w", err)
	}
	var report sweepReport
	if err := json.Unmarshal(data, &report); err != nil {
		return sweepReport{}, fmt.Errorf("tqkvsummarize: decode input: %w", err)
	}
	return report, nil
}

func summarizeReport(report sweepReport) summaryReport {
	out := summaryReport{
		GeneratedAt:           time.Now().UTC().Format(time.RFC3339Nano),
		InputGeneratedAt:      report.GeneratedAt,
		OverallParetoFrontier: report.ParetoFrontier,
		ByMethod:              make(map[string]groupSummary),
		ByLayer:               make(map[string]groupSummary),
		ByTokenIndex:          make(map[string]groupSummary),
		ByRelativePosition:    make(map[string]groupSummary),
	}

	methodSamples := make(map[string][]sweepSample)
	layerSamples := make(map[int][]sweepSample)
	tokenSamples := make(map[int][]sweepSample)
	positionSamples := make(map[string][]sweepSample)
	for _, sample := range report.Samples {
		seenMethods := make(map[string]struct{})
		for _, item := range sample.Cases {
			method := defaultMethod(item.Method)
			if _, ok := seenMethods[method]; ok {
				continue
			}
			seenMethods[method] = struct{}{}
			methodSamples[method] = append(methodSamples[method], sweepSample{
				Model:          sample.Model,
				Name:           sample.Name,
				PromptIndex:    sample.PromptIndex,
				Layer:          sample.Layer,
				TokenIndex:     sample.TokenIndex,
				TokenPosition:  sample.TokenPosition,
				SequenceLength: sample.SequenceLength,
				Tokens:         sample.Tokens,
				Cases:          filterCasesByMethod(sample.Cases, method),
			})
		}
		layerSamples[sample.Layer] = append(layerSamples[sample.Layer], sample)
		tokenSamples[sample.TokenIndex] = append(tokenSamples[sample.TokenIndex], sample)
		if position, ok := relativePositionLabel(sample); ok {
			positionSamples[position] = append(positionSamples[position], sample)
		}
	}

	methodKeys := make([]string, 0, len(methodSamples))
	for method := range methodSamples {
		methodKeys = append(methodKeys, method)
	}
	sort.Strings(methodKeys)
	for _, method := range methodKeys {
		out.ByMethod[method] = summarizeGroup(methodSamples[method], false)
	}

	layerKeys := make([]int, 0, len(layerSamples))
	for layer := range layerSamples {
		layerKeys = append(layerKeys, layer)
	}
	sort.Ints(layerKeys)
	for _, layer := range layerKeys {
		out.ByLayer[strconv.Itoa(layer)] = summarizeGroup(layerSamples[layer], false)
	}

	tokenKeys := make([]int, 0, len(tokenSamples))
	for tokenIndex := range tokenSamples {
		tokenKeys = append(tokenKeys, tokenIndex)
	}
	sort.Ints(tokenKeys)
	for _, tokenIndex := range tokenKeys {
		out.ByTokenIndex[strconv.Itoa(tokenIndex)] = summarizeGroup(tokenSamples[tokenIndex], true)
	}

	positionKeys := make([]string, 0, len(positionSamples))
	for position := range positionSamples {
		positionKeys = append(positionKeys, position)
	}
	sort.Slice(positionKeys, func(i, j int) bool {
		return relativePositionRank(positionKeys[i]) < relativePositionRank(positionKeys[j])
	})
	for _, position := range positionKeys {
		out.ByRelativePosition[position] = summarizeGroup(positionSamples[position], true)
	}

	if len(out.ByLayer) == 0 {
		out.ByLayer = nil
	}
	if len(out.ByTokenIndex) == 0 {
		out.ByTokenIndex = nil
	}
	if len(out.ByRelativePosition) == 0 {
		out.ByRelativePosition = nil
	}
	if len(out.ByMethod) == 0 {
		out.ByMethod = nil
	}
	return out
}

func relativePositionLabel(sample sweepSample) (string, bool) {
	if sample.TokenPosition != "" {
		return normalizeRelativePositionLabel(sample.TokenPosition), true
	}
	if sample.SequenceLength <= 1 {
		return "", false
	}
	denom := sample.SequenceLength - 1
	if denom <= 0 {
		return "", false
	}
	fraction := float64(sample.TokenIndex) / float64(denom)
	if fraction <= 0.125 {
		return "start", true
	}
	if fraction <= 0.375 {
		return "quarter", true
	}
	if fraction <= 0.625 {
		return "middle", true
	}
	if fraction <= 0.875 {
		return "three_quarter", true
	}
	return "last", true
}

func normalizeRelativePositionLabel(label string) string {
	switch strings.ToLower(strings.TrimSpace(label)) {
	case "0%":
		return "start"
	case "first":
		return "start"
	case "mid", "middle", "50%", "center", "centre":
		return "middle"
	case "q1", "25%":
		return "quarter"
	case "q3", "75%", "three-quarter":
		return "three_quarter"
	case "100%":
		return "last"
	default:
		return label
	}
}

func relativePositionRank(label string) int {
	switch label {
	case "start":
		return 0
	case "quarter":
		return 1
	case "middle":
		return 2
	case "three_quarter":
		return 3
	case "last":
		return 4
	default:
		return 5
	}
}

func summarizeGroup(samples []sweepSample, includeTokens bool) groupSummary {
	configs := aggregateSamples(samples)
	group := groupSummary{
		Samples:        len(samples),
		BestByMSE:      configs[0],
		ParetoFrontier: paretoFrontier(configs),
	}
	if includeTokens {
		seen := make(map[int]struct{}, len(samples))
		for _, sample := range samples {
			if _, ok := seen[sample.Tokens]; ok {
				continue
			}
			seen[sample.Tokens] = struct{}{}
			group.Tokens = append(group.Tokens, sample.Tokens)
		}
		sort.Ints(group.Tokens)
	}
	return group
}

func aggregateSamples(samples []sweepSample) []summaryConfig {
	accumulators := make(map[configKey]*configAccumulator)
	for _, sample := range samples {
		for _, item := range sample.Cases {
			key := configKey{
				Method:    defaultMethod(item.Method),
				KeyBits:   item.KeyBits,
				ValueBits: item.ValueBits,
				TopK:      requestedTopK(item),
			}
			acc := accumulators[key]
			if acc == nil {
				acc = &configAccumulator{}
				accumulators[key] = acc
			}
			acc.sumMSE += item.OutputMSE
			acc.sumCosine += item.OutputCosine
			acc.sumStorage += float64(item.CacheStorageBytes)
			acc.sumCompression += item.CompressionRatio
			acc.count++
		}
	}

	keys := make([]configKey, 0, len(accumulators))
	for key := range accumulators {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].KeyBits != keys[j].KeyBits {
			return keys[i].KeyBits < keys[j].KeyBits
		}
		if keys[i].ValueBits != keys[j].ValueBits {
			return keys[i].ValueBits < keys[j].ValueBits
		}
		return keys[i].TopK < keys[j].TopK
	})

	out := make([]summaryConfig, 0, len(keys))
	for _, key := range keys {
		acc := accumulators[key]
		out = append(out, summaryConfig{
			Method:                key.Method,
			KeyBits:               key.KeyBits,
			ValueBits:             key.ValueBits,
			TopK:                  key.TopK,
			Samples:               acc.count,
			MeanOutputMSE:         acc.sumMSE / float64(acc.count),
			MeanOutputCosine:      acc.sumCosine / float64(acc.count),
			MeanCacheStorageBytes: acc.sumStorage / float64(acc.count),
			MeanCompressionRatio:  acc.sumCompression / float64(acc.count),
		})
	}

	sort.SliceStable(out, func(i, j int) bool {
		if out[i].MeanOutputMSE != out[j].MeanOutputMSE {
			return out[i].MeanOutputMSE < out[j].MeanOutputMSE
		}
		if out[i].MeanOutputCosine != out[j].MeanOutputCosine {
			return out[i].MeanOutputCosine > out[j].MeanOutputCosine
		}
		if out[i].MeanCacheStorageBytes != out[j].MeanCacheStorageBytes {
			return out[i].MeanCacheStorageBytes < out[j].MeanCacheStorageBytes
		}
		if out[i].Method != out[j].Method {
			return out[i].Method < out[j].Method
		}
		if out[i].KeyBits != out[j].KeyBits {
			return out[i].KeyBits < out[j].KeyBits
		}
		if out[i].ValueBits != out[j].ValueBits {
			return out[i].ValueBits < out[j].ValueBits
		}
		return out[i].TopK < out[j].TopK
	})
	return out
}

func paretoFrontier(configs []summaryConfig) []summaryConfig {
	frontier := make([]summaryConfig, 0, len(configs))
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
		if frontier[i].MeanCacheStorageBytes != frontier[j].MeanCacheStorageBytes {
			return frontier[i].MeanCacheStorageBytes < frontier[j].MeanCacheStorageBytes
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

func dominates(left, right summaryConfig) bool {
	if left.MeanCacheStorageBytes > right.MeanCacheStorageBytes {
		return false
	}
	if left.MeanOutputMSE > right.MeanOutputMSE {
		return false
	}
	if left.MeanCacheStorageBytes < right.MeanCacheStorageBytes || left.MeanOutputMSE < right.MeanOutputMSE {
		return true
	}
	return left.MeanOutputCosine > right.MeanOutputCosine
}

func requestedTopK(item turboquant.TransformerLayerEvalResult) int {
	if item.RequestedTopK != 0 {
		return item.RequestedTopK
	}
	return item.TopK
}

func defaultMethod(method string) string {
	normalized, err := turboquant.NormalizeTransformerEvalMethodForCLI(method)
	if err != nil {
		return turboquant.TransformerEvalMethodTurboQuant
	}
	return normalized
}

func filterCasesByMethod(cases []turboquant.TransformerLayerEvalResult, method string) []turboquant.TransformerLayerEvalResult {
	out := make([]turboquant.TransformerLayerEvalResult, 0, len(cases))
	for _, item := range cases {
		if defaultMethod(item.Method) == method {
			out = append(out, item)
		}
	}
	return out
}
