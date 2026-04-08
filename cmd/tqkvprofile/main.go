package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	turboquant "github.com/odvcencio/turboquant"
)

type sweepReport struct {
	GeneratedAt string        `json:"generated_at"`
	Samples     []sweepSample `json:"samples"`
}

type sweepSample struct {
	Layer         int                                     `json:"layer,omitempty"`
	TokenPosition string                                  `json:"token_position,omitempty"`
	Tokens        int                                     `json:"tokens"`
	Heads         int                                     `json:"heads"`
	KVHeads       int                                     `json:"kv_heads,omitempty"`
	HeadDim       int                                     `json:"head_dim"`
	Cases         []turboquant.TransformerLayerEvalResult `json:"cases"`
}

type layerSummary struct {
	Layer                 int     `json:"layer"`
	Samples               int     `json:"samples"`
	Heads                 int     `json:"heads"`
	KVHeads               int     `json:"kv_heads,omitempty"`
	HeadDim               int     `json:"head_dim"`
	Capacity              int     `json:"capacity"`
	Method                string  `json:"method"`
	KeyBits               int     `json:"key_bits"`
	ValueBits             int     `json:"value_bits"`
	TopK                  int     `json:"top_k"`
	MeanOutputMSE         float64 `json:"mean_output_mse"`
	MeanOutputCosine      float64 `json:"mean_output_cosine"`
	P05OutputCosine       float64 `json:"p05_output_cosine,omitempty"`
	MinOutputCosine       float64 `json:"min_output_cosine,omitempty"`
	MeanRawKVBytes        float64 `json:"mean_raw_kv_bytes"`
	MeanCacheStorageBytes float64 `json:"mean_cache_storage_bytes"`
	MeanCompressionRatio  float64 `json:"mean_compression_ratio"`
}

type allocationSummary struct {
	Layers                int     `json:"layers"`
	Samples               int     `json:"samples"`
	MeanOutputMSE         float64 `json:"mean_output_mse"`
	MeanOutputCosine      float64 `json:"mean_output_cosine"`
	TotalMeanRawKVBytes   float64 `json:"total_mean_raw_kv_bytes"`
	TotalMeanStorageBytes float64 `json:"total_mean_cache_storage_bytes"`
	AggregateCompression  float64 `json:"aggregate_compression_ratio"`
}

type profileReport struct {
	ProfileSet              string                                 `json:"profile_set,omitempty"`
	GeneratedAt             string                                 `json:"generated_at"`
	InputGeneratedAt        string                                 `json:"input_generated_at,omitempty"`
	Method                  string                                 `json:"method"`
	TokenPositions          []string                               `json:"token_positions,omitempty"`
	StorageGranularityBytes int                                    `json:"storage_granularity_bytes"`
	MaxMeanStorageBytes     float64                                `json:"max_mean_storage_bytes,omitempty"`
	MinMeanCompression      float64                                `json:"min_mean_compression,omitempty"`
	MinMeanCosine           float64                                `json:"min_mean_cosine,omitempty"`
	MinP05Cosine            float64                                `json:"min_p05_cosine,omitempty"`
	SelectionMode           string                                 `json:"selection_mode"`
	Summary                 allocationSummary                      `json:"summary"`
	Layers                  []layerSummary                         `json:"layers"`
	Profiles                []turboquant.TransformerLayerKVProfile `json:"profiles"`
}

type profileBundleReport struct {
	GeneratedAt             string          `json:"generated_at"`
	InputGeneratedAt        string          `json:"input_generated_at,omitempty"`
	Method                  string          `json:"method"`
	StorageGranularityBytes int             `json:"storage_granularity_bytes"`
	MaxMeanStorageBytes     float64         `json:"max_mean_storage_bytes,omitempty"`
	MinMeanCompression      float64         `json:"min_mean_compression,omitempty"`
	MinMeanCosine           float64         `json:"min_mean_cosine,omitempty"`
	MinP05Cosine            float64         `json:"min_p05_cosine,omitempty"`
	SelectionMode           string          `json:"selection_mode"`
	Reports                 []profileReport `json:"reports"`
}

type layerKey struct {
	Layer     int
	KeyBits   int
	ValueBits int
	TopK      int
}

type layerMeta struct {
	heads     int
	kvHeads   int
	headDim   int
	maxTokens int
	samples   int
}

type layerAccumulator struct {
	sumMSE         float64
	sumCosine      float64
	cosine         []float64
	sumRaw         float64
	sumStorage     float64
	sumCompression float64
	count          int
}

type dpState struct {
	storageUnits int
	totalMSE     float64
	prev         int
	choice       int
}

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runCLI(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("tqkvprofile", flag.ContinueOnError)
	fs.SetOutput(stderr)

	inputPath := fs.String("input", "", "path to a tqkvsweep JSON report")
	outputPath := fs.String("out", "", "optional path to write the JSON profile")
	methodFlag := fs.String("method", turboquant.TransformerEvalMethodTurboQuant, "evaluation method to allocate: turboquant or uniform")
	tokenPositionsCSV := fs.String("token-position", "", "optional comma-separated relative token positions to include: start, quarter, middle, three_quarter, last")
	profileSetsCSV := fs.String("profile-set", "", "optional comma-separated profile sets to emit in one file: all,start,quarter,middle,three_quarter,last")
	maxMeanStorage := fs.Float64("max-mean-storage-bytes", 0, "optional total mean storage budget across chosen layers")
	minCompression := fs.Float64("min-mean-compression", 0, "optional minimum aggregate compression ratio across chosen layers")
	minMeanCosine := fs.Float64("min-mean-cosine", 0, "optional per-layer minimum mean output cosine")
	minP05Cosine := fs.Float64("min-p05-cosine", 0, "optional per-layer minimum p05 output cosine")
	storageGranularity := fs.Int("storage-granularity-bytes", 1, "storage rounding granularity for budgeted allocation")
	capacityOverride := fs.Int("capacity", 0, "optional capacity override for emitted runtime profiles")
	seedBase := fs.Int64("seed-base", 42, "base seed used when emitting runtime profiles")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *inputPath == "" {
		return errors.New("tqkvprofile: --input is required")
	}
	if *maxMeanStorage > 0 && *minCompression > 0 {
		return errors.New("tqkvprofile: set at most one of --max-mean-storage-bytes or --min-mean-compression")
	}
	if strings.TrimSpace(*tokenPositionsCSV) != "" && strings.TrimSpace(*profileSetsCSV) != "" {
		return errors.New("tqkvprofile: set at most one of --token-position or --profile-set")
	}
	if *storageGranularity <= 0 {
		return errors.New("tqkvprofile: --storage-granularity-bytes must be > 0")
	}
	if *minMeanCosine < 0 || *minMeanCosine > 1 {
		return errors.New("tqkvprofile: --min-mean-cosine must be within [0,1]")
	}
	if *minP05Cosine < 0 || *minP05Cosine > 1 {
		return errors.New("tqkvprofile: --min-p05-cosine must be within [0,1]")
	}
	method, err := turboquant.NormalizeTransformerEvalMethodForCLI(*methodFlag)
	if err != nil {
		return fmt.Errorf("tqkvprofile: method: %w", err)
	}
	tokenPositions, err := parseTokenPositionsCSV(*tokenPositionsCSV)
	if err != nil {
		return fmt.Errorf("tqkvprofile: token position: %w", err)
	}
	profileSets, err := parseProfileSetsCSV(*profileSetsCSV)
	if err != nil {
		return fmt.Errorf("tqkvprofile: profile set: %w", err)
	}
	report, err := loadReport(*inputPath)
	if err != nil {
		return err
	}
	selectionMode := profileSelectionMode(*maxMeanStorage, *minCompression, *minMeanCosine, *minP05Cosine)
	now := time.Now().UTC().Format(time.RFC3339Nano)

	var payload []byte
	if len(profileSets) == 0 {
		out, err := buildProfileReport(report, now, method, "", tokenPositions, *storageGranularity, *maxMeanStorage, *minCompression, *minMeanCosine, *minP05Cosine, *capacityOverride, *seedBase)
		if err != nil {
			return err
		}
		payload, err = json.MarshalIndent(out, "", "  ")
		if err != nil {
			return fmt.Errorf("tqkvprofile: encode report: %w", err)
		}
	} else {
		out := profileBundleReport{
			GeneratedAt:             now,
			InputGeneratedAt:        report.GeneratedAt,
			Method:                  method,
			StorageGranularityBytes: *storageGranularity,
			MaxMeanStorageBytes:     *maxMeanStorage,
			MinMeanCompression:      *minCompression,
			MinMeanCosine:           *minMeanCosine,
			MinP05Cosine:            *minP05Cosine,
			SelectionMode:           selectionMode,
			Reports:                 make([]profileReport, 0, len(profileSets)),
		}
		for _, profileSet := range profileSets {
			var positions []string
			if profileSet != "all" {
				positions = []string{profileSet}
			}
			item, err := buildProfileReport(report, now, method, profileSet, positions, *storageGranularity, *maxMeanStorage, *minCompression, *minMeanCosine, *minP05Cosine, *capacityOverride, *seedBase)
			if err != nil {
				return err
			}
			out.Reports = append(out.Reports, item)
		}
		payload, err = json.MarshalIndent(out, "", "  ")
		if err != nil {
			return fmt.Errorf("tqkvprofile: encode report: %w", err)
		}
	}
	if *outputPath == "" {
		_, err = stdout.Write(append(payload, '\n'))
		return err
	}
	if err := os.WriteFile(*outputPath, append(payload, '\n'), 0o644); err != nil {
		return fmt.Errorf("tqkvprofile: write report: %w", err)
	}
	_, err = fmt.Fprintf(stdout, "wrote %s\n", *outputPath)
	return err
}

func buildProfileReport(report sweepReport, generatedAt, method, profileSet string, tokenPositions []string, storageGranularity int, maxMeanStorage, minCompression, minMeanCosine, minP05Cosine float64, capacityOverride int, seedBase int64) (profileReport, error) {
	report = filterReportByTokenPosition(report, tokenPositions)
	layers, metas, candidates, err := aggregateCandidates(report, method)
	if err != nil {
		return profileReport{}, err
	}
	if len(layers) == 0 {
		if len(tokenPositions) > 0 {
			return profileReport{}, fmt.Errorf("tqkvprofile: no %q cases found for token positions %s", method, strings.Join(tokenPositions, ","))
		}
		return profileReport{}, fmt.Errorf("tqkvprofile: no %q cases found", method)
	}

	filteredCandidates, err := applyFidelityFloors(layers, candidates, minMeanCosine, minP05Cosine)
	if err != nil {
		return profileReport{}, err
	}
	selectionMode := profileSelectionMode(maxMeanStorage, minCompression, minMeanCosine, minP05Cosine)
	chosen, err := selectLayerAllocation(layers, filteredCandidates, selectionMode, maxMeanStorage, minCompression, storageGranularity)
	if err != nil {
		return profileReport{}, err
	}

	out := profileReport{
		ProfileSet:              profileSet,
		GeneratedAt:             generatedAt,
		InputGeneratedAt:        report.GeneratedAt,
		Method:                  method,
		TokenPositions:          tokenPositions,
		StorageGranularityBytes: storageGranularity,
		MaxMeanStorageBytes:     maxMeanStorage,
		MinMeanCompression:      minCompression,
		MinMeanCosine:           minMeanCosine,
		MinP05Cosine:            minP05Cosine,
		SelectionMode:           selectionMode,
		Layers:                  chosen,
	}
	out.Profiles = make([]turboquant.TransformerLayerKVProfile, len(chosen))
	for i, item := range chosen {
		seed := seedBase ^ (int64(item.Layer+1) * 1315423911)
		out.Profiles[i] = turboquant.TransformerLayerKVProfile{
			Layer:     item.Layer,
			Heads:     item.Heads,
			KVHeads:   item.KVHeads,
			HeadDim:   item.HeadDim,
			KeyBits:   item.KeyBits,
			ValueBits: item.ValueBits,
			Capacity:  capacityForLayer(capacityOverride, metas[item.Layer].maxTokens),
			Seed:      seed,
		}
	}
	out.Summary = summarizeAllocation(chosen)
	return out, nil
}

func loadReport(path string) (sweepReport, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return sweepReport{}, fmt.Errorf("tqkvprofile: read input: %w", err)
	}
	var report sweepReport
	if err := json.Unmarshal(data, &report); err != nil {
		return sweepReport{}, fmt.Errorf("tqkvprofile: decode input: %w", err)
	}
	return report, nil
}

func filterReportByTokenPosition(report sweepReport, positions []string) sweepReport {
	if len(positions) == 0 {
		return report
	}
	allowed := make(map[string]struct{}, len(positions))
	for _, position := range positions {
		allowed[position] = struct{}{}
	}
	filtered := sweepReport{
		GeneratedAt: report.GeneratedAt,
		Samples:     make([]sweepSample, 0, len(report.Samples)),
	}
	for _, sample := range report.Samples {
		position, ok := normalizeRelativePositionLabel(sample.TokenPosition)
		if !ok {
			continue
		}
		if _, keep := allowed[position]; !keep {
			continue
		}
		sample.TokenPosition = position
		filtered.Samples = append(filtered.Samples, sample)
	}
	return filtered
}

func aggregateCandidates(report sweepReport, method string) ([]int, map[int]layerMeta, map[int][]layerSummary, error) {
	metas := make(map[int]layerMeta)
	accumulators := make(map[layerKey]*layerAccumulator)
	for _, sample := range report.Samples {
		meta := metas[sample.Layer]
		kvHeads := sample.KVHeads
		if kvHeads == 0 {
			kvHeads = sample.Heads
		}
		if meta.samples == 0 {
			meta.heads = sample.Heads
			meta.kvHeads = kvHeads
			meta.headDim = sample.HeadDim
		} else {
			if meta.heads != sample.Heads || meta.kvHeads != kvHeads || meta.headDim != sample.HeadDim {
				return nil, nil, nil, fmt.Errorf("tqkvprofile: inconsistent shape for layer %d", sample.Layer)
			}
		}
		if sample.Tokens > meta.maxTokens {
			meta.maxTokens = sample.Tokens
		}
		meta.samples++
		metas[sample.Layer] = meta

		for _, item := range sample.Cases {
			if defaultMethod(item.Method) != method {
				continue
			}
			key := layerKey{
				Layer:     sample.Layer,
				KeyBits:   item.KeyBits,
				ValueBits: item.ValueBits,
				TopK:      requestedTopK(item),
			}
			acc := accumulators[key]
			if acc == nil {
				acc = &layerAccumulator{}
				accumulators[key] = acc
			}
			acc.sumMSE += item.OutputMSE
			acc.sumCosine += item.OutputCosine
			acc.cosine = append(acc.cosine, item.OutputCosine)
			acc.sumRaw += float64(item.RawKVBytes)
			acc.sumStorage += float64(item.CacheStorageBytes)
			acc.sumCompression += item.CompressionRatio
			acc.count++
		}
	}

	layersSet := make(map[int]struct{})
	candidates := make(map[int][]layerSummary)
	for key, acc := range accumulators {
		meta := metas[key.Layer]
		item := layerSummary{
			Layer:                 key.Layer,
			Samples:               acc.count,
			Heads:                 meta.heads,
			KVHeads:               meta.kvHeads,
			HeadDim:               meta.headDim,
			Capacity:              meta.maxTokens,
			Method:                method,
			KeyBits:               key.KeyBits,
			ValueBits:             key.ValueBits,
			TopK:                  key.TopK,
			MeanOutputMSE:         acc.sumMSE / float64(acc.count),
			MeanOutputCosine:      acc.sumCosine / float64(acc.count),
			P05OutputCosine:       percentile(acc.cosine, 0.05),
			MinOutputCosine:       minFloat64(acc.cosine),
			MeanRawKVBytes:        acc.sumRaw / float64(acc.count),
			MeanCacheStorageBytes: acc.sumStorage / float64(acc.count),
			MeanCompressionRatio:  acc.sumCompression / float64(acc.count),
		}
		candidates[key.Layer] = append(candidates[key.Layer], item)
		layersSet[key.Layer] = struct{}{}
	}

	layers := make([]int, 0, len(layersSet))
	for layer := range layersSet {
		layers = append(layers, layer)
	}
	sort.Ints(layers)
	for _, layer := range layers {
		candidates[layer] = paretoCandidates(candidates[layer])
		sort.SliceStable(candidates[layer], func(i, j int) bool {
			left := candidates[layer][i]
			right := candidates[layer][j]
			if left.MeanCacheStorageBytes != right.MeanCacheStorageBytes {
				return left.MeanCacheStorageBytes < right.MeanCacheStorageBytes
			}
			if left.MeanOutputMSE != right.MeanOutputMSE {
				return left.MeanOutputMSE < right.MeanOutputMSE
			}
			if left.KeyBits != right.KeyBits {
				return left.KeyBits < right.KeyBits
			}
			if left.ValueBits != right.ValueBits {
				return left.ValueBits < right.ValueBits
			}
			return left.TopK < right.TopK
		})
	}
	return layers, metas, candidates, nil
}

func parseTokenPositionsCSV(raw string) ([]string, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	seen := make(map[string]struct{}, len(parts))
	for _, part := range parts {
		position, ok := normalizeRelativePositionLabel(part)
		if !ok {
			return nil, fmt.Errorf("unsupported token position %q", strings.TrimSpace(part))
		}
		if _, exists := seen[position]; exists {
			continue
		}
		seen[position] = struct{}{}
		out = append(out, position)
	}
	if len(out) == 0 {
		return nil, nil
	}
	sort.Slice(out, func(i, j int) bool {
		return relativePositionRank(out[i]) < relativePositionRank(out[j])
	})
	return out, nil
}

func parseProfileSetsCSV(raw string) ([]string, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	seen := make(map[string]struct{}, len(parts))
	for _, part := range parts {
		part = strings.ToLower(strings.TrimSpace(part))
		if part == "" {
			continue
		}
		item := part
		if item != "all" {
			var ok bool
			item, ok = normalizeRelativePositionLabel(item)
			if !ok {
				return nil, fmt.Errorf("unsupported profile set %q", strings.TrimSpace(part))
			}
		}
		if _, exists := seen[item]; exists {
			continue
		}
		seen[item] = struct{}{}
		out = append(out, item)
	}
	if len(out) == 0 {
		return nil, nil
	}
	sort.Slice(out, func(i, j int) bool {
		return profileSetRank(out[i]) < profileSetRank(out[j])
	})
	return out, nil
}

func normalizeRelativePositionLabel(raw string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "0%", "first", "start":
		return "start", true
	case "q1", "25%", "quarter":
		return "quarter", true
	case "mid", "middle", "50%", "center", "centre":
		return "middle", true
	case "q3", "75%", "three_quarter", "three-quarter":
		return "three_quarter", true
	case "100%", "last":
		return "last", true
	default:
		return "", false
	}
}

func profileSelectionMode(maxMeanStorage, minCompression, minMeanCosine, minP05Cosine float64) string {
	mode := "best-mse"
	if minMeanCosine > 0 || minP05Cosine > 0 {
		mode = "fidelity-floor"
	}
	if maxMeanStorage > 0 || minCompression > 0 {
		if mode == "fidelity-floor" {
			return "budgeted-fidelity-floor"
		}
		return "budgeted"
	}
	return mode
}

func profileSetRank(label string) int {
	if label == "all" {
		return -1
	}
	return relativePositionRank(label)
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

func paretoCandidates(in []layerSummary) []layerSummary {
	if len(in) <= 1 {
		return append([]layerSummary(nil), in...)
	}
	items := append([]layerSummary(nil), in...)
	sort.SliceStable(items, func(i, j int) bool {
		if items[i].MeanCacheStorageBytes != items[j].MeanCacheStorageBytes {
			return items[i].MeanCacheStorageBytes < items[j].MeanCacheStorageBytes
		}
		if items[i].MeanOutputMSE != items[j].MeanOutputMSE {
			return items[i].MeanOutputMSE < items[j].MeanOutputMSE
		}
		if items[i].MeanOutputCosine != items[j].MeanOutputCosine {
			return items[i].MeanOutputCosine > items[j].MeanOutputCosine
		}
		if items[i].KeyBits != items[j].KeyBits {
			return items[i].KeyBits < items[j].KeyBits
		}
		if items[i].ValueBits != items[j].ValueBits {
			return items[i].ValueBits < items[j].ValueBits
		}
		return items[i].TopK < items[j].TopK
	})
	bestMSE := math.Inf(1)
	out := make([]layerSummary, 0, len(items))
	for _, item := range items {
		if item.MeanOutputMSE < bestMSE {
			out = append(out, item)
			bestMSE = item.MeanOutputMSE
		}
	}
	return out
}

func applyFidelityFloors(layers []int, candidates map[int][]layerSummary, minMeanCosine, minP05Cosine float64) (map[int][]layerSummary, error) {
	if minMeanCosine <= 0 && minP05Cosine <= 0 {
		return candidates, nil
	}
	out := make(map[int][]layerSummary, len(candidates))
	for _, layer := range layers {
		items := candidates[layer]
		filtered := make([]layerSummary, 0, len(items))
		for _, item := range items {
			if minMeanCosine > 0 && item.MeanOutputCosine < minMeanCosine {
				continue
			}
			if minP05Cosine > 0 && item.P05OutputCosine < minP05Cosine {
				continue
			}
			filtered = append(filtered, item)
		}
		if len(filtered) == 0 {
			return nil, fmt.Errorf(
				"tqkvprofile: no candidates for layer %d satisfy min_mean_cosine=%.4f min_p05_cosine=%.4f",
				layer,
				minMeanCosine,
				minP05Cosine,
			)
		}
		out[layer] = filtered
	}
	return out, nil
}

func selectLayerAllocation(layers []int, candidates map[int][]layerSummary, mode string, maxMeanStorage, minCompression float64, granularity int) ([]layerSummary, error) {
	if mode == "best-mse" {
		out := make([]layerSummary, len(layers))
		for i, layer := range layers {
			if len(candidates[layer]) == 0 {
				return nil, fmt.Errorf("tqkvprofile: no candidates for layer %d", layer)
			}
			out[i] = bestLayerCandidate(candidates[layer])
		}
		return out, nil
	}
	if mode == "fidelity-floor" {
		out := make([]layerSummary, len(layers))
		for i, layer := range layers {
			if len(candidates[layer]) == 0 {
				return nil, fmt.Errorf("tqkvprofile: no candidates for layer %d", layer)
			}
			out[i] = smallestLayerCandidate(candidates[layer])
		}
		return out, nil
	}

	totalRaw := 0.0
	minPossibleStorage := 0.0
	for _, layer := range layers {
		if len(candidates[layer]) == 0 {
			return nil, fmt.Errorf("tqkvprofile: no candidates for layer %d", layer)
		}
		totalRaw += candidates[layer][0].MeanRawKVBytes
		minPossibleStorage += smallestStorage(candidates[layer])
	}
	maxStorage := maxMeanStorage
	if minCompression > 0 {
		maxStorage = totalRaw / minCompression
	}
	if maxStorage <= 0 {
		return nil, errors.New("tqkvprofile: budgeted mode requires a positive storage or compression target")
	}
	if minPossibleStorage > maxStorage {
		return nil, fmt.Errorf("tqkvprofile: budget %.2f bytes is smaller than minimum achievable %.2f bytes", maxStorage, minPossibleStorage)
	}

	states := []dpState{{storageUnits: 0, totalMSE: 0, prev: -1, choice: -1}}
	frontier := []int{0}
	for _, layer := range layers {
		nextByStorage := make(map[int]int)
		for _, stateIdx := range frontier {
			state := states[stateIdx]
			for choiceIdx, cand := range candidates[layer] {
				units := state.storageUnits + storageUnits(cand.MeanCacheStorageBytes, granularity)
				mse := state.totalMSE + cand.MeanOutputMSE
				if existingIdx, ok := nextByStorage[units]; ok {
					if mse >= states[existingIdx].totalMSE {
						continue
					}
				}
				states = append(states, dpState{
					storageUnits: units,
					totalMSE:     mse,
					prev:         stateIdx,
					choice:       choiceIdx,
				})
				nextByStorage[units] = len(states) - 1
			}
		}
		frontier = pruneFrontier(nextByStorage, states)
	}

	maxUnits := storageUnits(maxStorage, granularity)
	bestIdx := -1
	for _, idx := range frontier {
		if states[idx].storageUnits > maxUnits {
			continue
		}
		if bestIdx == -1 || states[idx].totalMSE < states[bestIdx].totalMSE || (states[idx].totalMSE == states[bestIdx].totalMSE && states[idx].storageUnits < states[bestIdx].storageUnits) {
			bestIdx = idx
		}
	}
	if bestIdx == -1 {
		return nil, fmt.Errorf("tqkvprofile: no allocation satisfies storage budget %.2f bytes", maxStorage)
	}
	return reconstructAllocation(bestIdx, states, layers, candidates), nil
}

func pruneFrontier(indexByStorage map[int]int, states []dpState) []int {
	storages := make([]int, 0, len(indexByStorage))
	for storage := range indexByStorage {
		storages = append(storages, storage)
	}
	sort.Ints(storages)
	bestMSE := math.Inf(1)
	out := make([]int, 0, len(storages))
	for _, storage := range storages {
		idx := indexByStorage[storage]
		if states[idx].totalMSE < bestMSE {
			out = append(out, idx)
			bestMSE = states[idx].totalMSE
		}
	}
	return out
}

func reconstructAllocation(bestIdx int, states []dpState, layers []int, candidates map[int][]layerSummary) []layerSummary {
	out := make([]layerSummary, len(layers))
	current := bestIdx
	for i := len(layers) - 1; i >= 0; i-- {
		state := states[current]
		out[i] = candidates[layers[i]][state.choice]
		current = state.prev
	}
	return out
}

func bestLayerCandidate(items []layerSummary) layerSummary {
	best := items[0]
	for _, item := range items[1:] {
		if item.MeanOutputMSE < best.MeanOutputMSE ||
			(item.MeanOutputMSE == best.MeanOutputMSE && item.MeanCacheStorageBytes < best.MeanCacheStorageBytes) ||
			(item.MeanOutputMSE == best.MeanOutputMSE && item.MeanCacheStorageBytes == best.MeanCacheStorageBytes && item.MeanOutputCosine > best.MeanOutputCosine) {
			best = item
		}
	}
	return best
}

func smallestLayerCandidate(items []layerSummary) layerSummary {
	best := items[0]
	for _, item := range items[1:] {
		if item.MeanCacheStorageBytes < best.MeanCacheStorageBytes ||
			(item.MeanCacheStorageBytes == best.MeanCacheStorageBytes && item.MeanOutputMSE < best.MeanOutputMSE) ||
			(item.MeanCacheStorageBytes == best.MeanCacheStorageBytes && item.MeanOutputMSE == best.MeanOutputMSE && item.MeanOutputCosine > best.MeanOutputCosine) {
			best = item
		}
	}
	return best
}

func smallestStorage(items []layerSummary) float64 {
	best := items[0].MeanCacheStorageBytes
	for _, item := range items[1:] {
		if item.MeanCacheStorageBytes < best {
			best = item.MeanCacheStorageBytes
		}
	}
	return best
}

func summarizeAllocation(items []layerSummary) allocationSummary {
	var out allocationSummary
	out.Layers = len(items)
	for _, item := range items {
		out.Samples += item.Samples
		out.MeanOutputMSE += item.MeanOutputMSE
		out.MeanOutputCosine += item.MeanOutputCosine
		out.TotalMeanRawKVBytes += item.MeanRawKVBytes
		out.TotalMeanStorageBytes += item.MeanCacheStorageBytes
	}
	if out.Layers > 0 {
		out.MeanOutputMSE /= float64(out.Layers)
		out.MeanOutputCosine /= float64(out.Layers)
	}
	if out.TotalMeanStorageBytes > 0 {
		out.AggregateCompression = out.TotalMeanRawKVBytes / out.TotalMeanStorageBytes
	}
	return out
}

func storageUnits(bytes float64, granularity int) int {
	return int(math.Ceil(bytes / float64(granularity)))
}

func capacityForLayer(override, observed int) int {
	if override > 0 {
		return override
	}
	return observed
}

func defaultMethod(method string) string {
	normalized, err := turboquant.NormalizeTransformerEvalMethodForCLI(method)
	if err != nil {
		return turboquant.TransformerEvalMethodTurboQuant
	}
	return normalized
}

func requestedTopK(item turboquant.TransformerLayerEvalResult) int {
	if item.RequestedTopK > 0 {
		return item.RequestedTopK
	}
	return item.TopK
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
	if len(values) == 1 {
		return values[0]
	}
	items := append([]float64(nil), values...)
	sort.Float64s(items)
	if p <= 0 {
		return items[0]
	}
	if p >= 1 {
		return items[len(items)-1]
	}
	pos := p * float64(len(items)-1)
	low := int(math.Floor(pos))
	high := int(math.Ceil(pos))
	if low == high {
		return items[low]
	}
	frac := pos - float64(low)
	return items[low]*(1-frac) + items[high]*frac
}
