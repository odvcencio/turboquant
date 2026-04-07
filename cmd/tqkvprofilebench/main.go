package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	turboquant "github.com/odvcencio/turboquant"
)

type profileLayer struct {
	Layer     int `json:"layer"`
	Heads     int `json:"heads"`
	KVHeads   int `json:"kv_heads,omitempty"`
	HeadDim   int `json:"head_dim"`
	KeyBits   int `json:"key_bits"`
	ValueBits int `json:"value_bits"`
	TopK      int `json:"top_k"`
	Capacity  int `json:"capacity"`
}

type profileReportInput struct {
	ProfileSet     string                                 `json:"profile_set,omitempty"`
	TokenPositions []string                               `json:"token_positions,omitempty"`
	Layers         []profileLayer                         `json:"layers"`
	Profiles       []turboquant.TransformerLayerKVProfile `json:"profiles"`
}

type profileBundleInput struct {
	Reports []profileReportInput `json:"reports"`
}

type captureGroupKey struct {
	Model          string
	PromptIndex    int
	TokenPosition  string
	TokenIndex     int
	SequenceLength int
}

type captureGroup struct {
	Key      captureGroupKey
	Captures map[int]turboquant.TransformerLayerCapture
}

type layerRuntimeSummary struct {
	Layer                       int     `json:"layer"`
	Heads                       int     `json:"heads"`
	KVHeads                     int     `json:"kv_heads,omitempty"`
	HeadDim                     int     `json:"head_dim"`
	KeyBits                     int     `json:"key_bits"`
	ValueBits                   int     `json:"value_bits"`
	TopK                        int     `json:"top_k"`
	Capacity                    int     `json:"capacity"`
	Runs                        int     `json:"runs"`
	MeanTokens                  float64 `json:"mean_tokens"`
	MeanRawFP32Bytes            float64 `json:"mean_raw_fp32_bytes"`
	MeanRawFP16Bytes            float64 `json:"mean_raw_fp16_bytes"`
	MeanQuantLiveBytes          float64 `json:"mean_quant_live_bytes"`
	MeanQuantStorageBytes       float64 `json:"mean_quant_storage_bytes"`
	CompressionVsFP32           float64 `json:"compression_vs_fp32"`
	CompressionVsFP16           float64 `json:"compression_vs_fp16"`
	MeanAppendDurationMS        float64 `json:"mean_append_duration_ms"`
	MeanGPUUploadDurationMS     float64 `json:"mean_gpu_upload_duration_ms,omitempty"`
	MeanAttentionDurationMS     float64 `json:"mean_attention_duration_ms"`
	MeanAppendLayerTokensPerSec float64 `json:"mean_append_layer_tokens_per_second"`
	MeanAttentionQueriesPerSec  float64 `json:"mean_attention_queries_per_second"`
	GPUEnabledRuns              int     `json:"gpu_enabled_runs,omitempty"`
}

type benchSummary struct {
	Runs                        int     `json:"runs"`
	MeanRawFP32Bytes            float64 `json:"mean_raw_fp32_bytes"`
	MeanRawFP16Bytes            float64 `json:"mean_raw_fp16_bytes"`
	MeanQuantLiveBytes          float64 `json:"mean_quant_live_bytes"`
	MeanQuantStorageBytes       float64 `json:"mean_quant_storage_bytes"`
	CompressionVsFP32           float64 `json:"compression_vs_fp32"`
	CompressionVsFP16           float64 `json:"compression_vs_fp16"`
	MeanAppendDurationMS        float64 `json:"mean_append_duration_ms"`
	MeanGPUUploadDurationMS     float64 `json:"mean_gpu_upload_duration_ms,omitempty"`
	MeanAttentionDurationMS     float64 `json:"mean_attention_duration_ms"`
	MeanTotalDurationMS         float64 `json:"mean_total_duration_ms"`
	MeanAppendLayerTokensPerSec float64 `json:"mean_append_layer_tokens_per_second"`
	MeanAttentionQueriesPerSec  float64 `json:"mean_attention_queries_per_second"`
	MeanGroupsPerSec            float64 `json:"mean_groups_per_second"`
	TotalLayerRuns              int     `json:"total_layer_runs"`
	GPUEnabledRuns              int     `json:"gpu_enabled_runs,omitempty"`
	GPUEnabledLayerRuns         int     `json:"gpu_enabled_layer_runs,omitempty"`
}

type profileSetBenchReport struct {
	ProfileSet        string                `json:"profile_set"`
	TokenPositions    []string              `json:"token_positions,omitempty"`
	RequestedGroups   int                   `json:"requested_groups"`
	BenchmarkedGroups int                   `json:"benchmarked_groups"`
	Layers            int                   `json:"layers"`
	Summary           benchSummary          `json:"summary"`
	LayerSummaries    []layerRuntimeSummary `json:"layer_summaries"`
}

type report struct {
	GeneratedAt string                  `json:"generated_at"`
	CapturePath string                  `json:"capture_path"`
	ProfilePath string                  `json:"profile_path"`
	Warmup      int                     `json:"warmup"`
	Iterations  int                     `json:"iterations"`
	TryGPU      bool                    `json:"try_gpu"`
	ProfileSets []profileSetBenchReport `json:"profile_sets"`
}

type layerBenchContext struct {
	profile turboquant.TransformerLayerKVProfile
	layer   profileLayer
	capture turboquant.TransformerLayerCapture
	output  []float32
}

type layerAccumulator struct {
	layer           profileLayer
	count           int
	sumTokens       float64
	sumRawFP32      float64
	sumRawFP16      float64
	sumLive         float64
	sumStorage      float64
	sumAppendNS     float64
	sumGPUUploadNS  float64
	sumAttentionNS  float64
	sumAppendRate   float64
	sumAttentionQPS float64
	gpuRuns         int
}

type summaryAccumulator struct {
	count           int
	sumRawFP32      float64
	sumRawFP16      float64
	sumLive         float64
	sumStorage      float64
	sumAppendNS     float64
	sumGPUUploadNS  float64
	sumAttentionNS  float64
	sumTotalNS      float64
	sumAppendRate   float64
	sumAttentionQPS float64
	sumGroupRate    float64
	totalLayerRuns  int
	gpuRuns         int
	gpuLayerRuns    int
}

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runCLI(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("tqkvprofilebench", flag.ContinueOnError)
	fs.SetOutput(stderr)
	capturePath := fs.String("capture", "", "path to a transformer capture JSON file")
	profilePath := fs.String("profile", "", "path to a tqkvprofile JSON file")
	outputPath := fs.String("out", "", "optional path to write the JSON report")
	profileSetsCSV := fs.String("profile-set", "", "optional comma-separated profile sets to benchmark: all,start,quarter,middle,three_quarter,last")
	warmup := fs.Int("warmup", 1, "number of warmup runs per capture group")
	iterations := fs.Int("iterations", 3, "number of timed runs per capture group")
	maxGroups := fs.Int("max-groups", 0, "optional cap on complete capture groups per profile set")
	tryGPU := fs.Bool("gpu", false, "try the experimental GPU path when available")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*capturePath) == "" {
		return errors.New("tqkvprofilebench: --capture is required")
	}
	if strings.TrimSpace(*profilePath) == "" {
		return errors.New("tqkvprofilebench: --profile is required")
	}
	if *warmup < 0 {
		return errors.New("tqkvprofilebench: --warmup must be >= 0")
	}
	if *iterations <= 0 {
		return errors.New("tqkvprofilebench: --iterations must be > 0")
	}
	if *maxGroups < 0 {
		return errors.New("tqkvprofilebench: --max-groups must be >= 0")
	}

	profileSets, err := parseProfileSetsCSV(*profileSetsCSV)
	if err != nil {
		return fmt.Errorf("tqkvprofilebench: profile set: %w", err)
	}
	profiles, err := loadProfileReports(*profilePath, profileSets)
	if err != nil {
		return err
	}
	groups, err := loadCaptureGroups(*capturePath)
	if err != nil {
		return err
	}

	out := report{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano),
		CapturePath: *capturePath,
		ProfilePath: *profilePath,
		Warmup:      *warmup,
		Iterations:  *iterations,
		TryGPU:      *tryGPU,
		ProfileSets: make([]profileSetBenchReport, 0, len(profiles)),
	}
	for _, profile := range profiles {
		item, err := benchmarkProfileSet(profile, groups, *warmup, *iterations, *maxGroups, *tryGPU)
		if err != nil {
			return err
		}
		out.ProfileSets = append(out.ProfileSets, item)
	}

	payload, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return fmt.Errorf("tqkvprofilebench: encode report: %w", err)
	}
	if strings.TrimSpace(*outputPath) == "" {
		_, err = stdout.Write(append(payload, '\n'))
		return err
	}
	if err := os.WriteFile(*outputPath, append(payload, '\n'), 0o644); err != nil {
		return fmt.Errorf("tqkvprofilebench: write report: %w", err)
	}
	_, err = fmt.Fprintf(stdout, "wrote %s\n", *outputPath)
	return err
}

func loadProfileReports(path string, wanted []string) ([]profileReportInput, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tqkvprofilebench: read profile: %w", err)
	}
	var bundle profileBundleInput
	if err := json.Unmarshal(data, &bundle); err == nil && len(bundle.Reports) != 0 {
		reports := normalizeProfileReports(bundle.Reports)
		return filterProfileReports(reports, wanted)
	}
	var single profileReportInput
	if err := json.Unmarshal(data, &single); err != nil {
		return nil, fmt.Errorf("tqkvprofilebench: decode profile: %w", err)
	}
	if len(single.Profiles) == 0 {
		return nil, fmt.Errorf("tqkvprofilebench: profile file %s did not contain any runtime profiles", path)
	}
	reports, err := filterProfileReports(normalizeProfileReports([]profileReportInput{single}), wanted)
	if err != nil {
		return nil, err
	}
	return reports, nil
}

func normalizeProfileReports(in []profileReportInput) []profileReportInput {
	out := make([]profileReportInput, len(in))
	copy(out, in)
	for i := range out {
		label := strings.TrimSpace(out[i].ProfileSet)
		if label == "" {
			label = "all"
		}
		out[i].ProfileSet = label
		if len(out[i].TokenPositions) == 0 && label != "all" {
			out[i].TokenPositions = []string{label}
		}
		layerByID := make(map[int]profileLayer, len(out[i].Layers))
		for _, layer := range out[i].Layers {
			layerByID[layer.Layer] = layer
		}
		for j := range out[i].Profiles {
			if out[i].Profiles[j].KVHeads != 0 {
				continue
			}
			layer, ok := layerByID[out[i].Profiles[j].Layer]
			if !ok {
				continue
			}
			if effectiveKVHeadsForLayer(layer) != layer.Heads {
				out[i].Profiles[j].KVHeads = effectiveKVHeadsForLayer(layer)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return profileSetRank(out[i].ProfileSet) < profileSetRank(out[j].ProfileSet)
	})
	return out
}

func filterProfileReports(in []profileReportInput, wanted []string) ([]profileReportInput, error) {
	if len(wanted) == 0 {
		return in, nil
	}
	allowed := make(map[string]struct{}, len(wanted))
	for _, item := range wanted {
		allowed[item] = struct{}{}
	}
	out := make([]profileReportInput, 0, len(in))
	for _, item := range in {
		if _, ok := allowed[item.ProfileSet]; ok {
			out = append(out, item)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("tqkvprofilebench: no requested profile sets matched %s", strings.Join(wanted, ","))
	}
	return out, nil
}

func loadCaptureGroups(path string) ([]captureGroup, error) {
	groupMap := make(map[captureGroupKey]map[int]turboquant.TransformerLayerCapture)
	err := turboquant.WalkTransformerLayerCapturesFile(path, func(sample turboquant.TransformerLayerCapture) error {
		if err := sample.Validate(); err != nil {
			return err
		}
		position, ok := normalizeRelativePositionLabel(sample.TokenPosition)
		if ok {
			sample.TokenPosition = position
		}
		key := captureGroupKey{
			Model:          sample.Model,
			PromptIndex:    sample.PromptIndex,
			TokenPosition:  sample.TokenPosition,
			TokenIndex:     sample.TokenIndex,
			SequenceLength: sample.SequenceLength,
		}
		layers := groupMap[key]
		if layers == nil {
			layers = make(map[int]turboquant.TransformerLayerCapture)
			groupMap[key] = layers
		}
		if _, exists := layers[sample.Layer]; exists {
			return fmt.Errorf("tqkvprofilebench: duplicate capture for layer %d in prompt %d position %q token %d", sample.Layer, sample.PromptIndex, sample.TokenPosition, sample.TokenIndex)
		}
		layers[sample.Layer] = sample
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("tqkvprofilebench: decode capture: %w", err)
	}
	groups := make([]captureGroup, 0, len(groupMap))
	for key, captures := range groupMap {
		groups = append(groups, captureGroup{Key: key, Captures: captures})
	}
	sort.Slice(groups, func(i, j int) bool {
		left := groups[i].Key
		right := groups[j].Key
		if left.Model != right.Model {
			return left.Model < right.Model
		}
		if left.PromptIndex != right.PromptIndex {
			return left.PromptIndex < right.PromptIndex
		}
		if relativePositionRank(left.TokenPosition) != relativePositionRank(right.TokenPosition) {
			return relativePositionRank(left.TokenPosition) < relativePositionRank(right.TokenPosition)
		}
		if left.TokenIndex != right.TokenIndex {
			return left.TokenIndex < right.TokenIndex
		}
		return left.SequenceLength < right.SequenceLength
	})
	return groups, nil
}

func benchmarkProfileSet(profile profileReportInput, groups []captureGroup, warmup, iterations, maxGroups int, tryGPU bool) (profileSetBenchReport, error) {
	contextsByGroup, err := selectBenchGroups(profile, groups, maxGroups)
	if err != nil {
		return profileSetBenchReport{}, err
	}

	result := profileSetBenchReport{
		ProfileSet:        profile.ProfileSet,
		TokenPositions:    append([]string(nil), profile.TokenPositions...),
		RequestedGroups:   len(contextsByGroup),
		BenchmarkedGroups: len(contextsByGroup),
		Layers:            len(profile.Profiles),
	}
	layerAcc := make(map[int]*layerAccumulator, len(profile.Profiles))
	summaryAcc := summaryAccumulator{}
	for _, group := range contextsByGroup {
		for run := 0; run < warmup+iterations; run++ {
			timed := run >= warmup
			runMetrics, err := benchmarkGroupRun(group, tryGPU)
			if err != nil {
				return profileSetBenchReport{}, err
			}
			if !timed {
				continue
			}
			summaryAcc.add(runMetrics)
			for _, item := range runMetrics.layers {
				acc := layerAcc[item.layer.Layer]
				if acc == nil {
					acc = &layerAccumulator{layer: item.layer}
					layerAcc[item.layer.Layer] = acc
				}
				acc.add(item)
			}
		}
	}
	result.Summary = summaryAcc.summary()
	layerIDs := make([]int, 0, len(layerAcc))
	for layer := range layerAcc {
		layerIDs = append(layerIDs, layer)
	}
	sort.Ints(layerIDs)
	result.LayerSummaries = make([]layerRuntimeSummary, 0, len(layerIDs))
	for _, layer := range layerIDs {
		result.LayerSummaries = append(result.LayerSummaries, layerAcc[layer].summary())
	}
	return result, nil
}

type groupBenchContext struct {
	key    captureGroupKey
	layers []layerBenchContext
}

func selectBenchGroups(profile profileReportInput, groups []captureGroup, maxGroups int) ([]groupBenchContext, error) {
	if len(profile.Profiles) == 0 || len(profile.Layers) == 0 {
		return nil, fmt.Errorf("tqkvprofilebench: profile set %q does not contain any layer selections", profile.ProfileSet)
	}
	layerByID := make(map[int]profileLayer, len(profile.Layers))
	for _, item := range profile.Layers {
		layerByID[item.Layer] = item
	}
	selected := make([]groupBenchContext, 0, len(groups))
	for _, group := range groups {
		if len(profile.TokenPositions) != 0 {
			if _, ok := containsTokenPosition(profile.TokenPositions, group.Key.TokenPosition); !ok {
				continue
			}
		}
		contexts := make([]layerBenchContext, 0, len(profile.Profiles))
		complete := true
		for _, runtimeProfile := range profile.Profiles {
			capture, ok := group.Captures[runtimeProfile.Layer]
			if !ok {
				complete = false
				break
			}
			layer, ok := layerByID[runtimeProfile.Layer]
			if !ok {
				return nil, fmt.Errorf("tqkvprofilebench: profile set %q is missing layer summary for layer %d", profile.ProfileSet, runtimeProfile.Layer)
			}
			if runtimeProfile.KVHeads == 0 && kvHeadsForCapture(capture) != capture.Heads {
				runtimeProfile.KVHeads = kvHeadsForCapture(capture)
			}
			if layer.KVHeads == 0 && kvHeadsForCapture(capture) != capture.Heads {
				layer.KVHeads = kvHeadsForCapture(capture)
			}
			if err := validateLayerCompatibility(runtimeProfile, layer, capture); err != nil {
				return nil, err
			}
			contexts = append(contexts, layerBenchContext{
				profile: runtimeProfile,
				layer:   layer,
				capture: capture,
				output:  make([]float32, len(capture.Query)),
			})
		}
		if !complete {
			continue
		}
		selected = append(selected, groupBenchContext{key: group.Key, layers: contexts})
	}
	if len(selected) == 0 {
		return nil, fmt.Errorf("tqkvprofilebench: no complete capture groups matched profile set %q", profile.ProfileSet)
	}
	if maxGroups > 0 && len(selected) > maxGroups {
		selected = selected[:maxGroups]
	}
	return selected, nil
}

func containsTokenPosition(positions []string, target string) (string, bool) {
	target, ok := normalizeRelativePositionLabel(target)
	if !ok {
		target = strings.ToLower(strings.TrimSpace(target))
	}
	for _, position := range positions {
		if position == target {
			return target, true
		}
	}
	return "", false
}

func validateLayerCompatibility(profile turboquant.TransformerLayerKVProfile, layer profileLayer, capture turboquant.TransformerLayerCapture) error {
	if profile.Heads != capture.Heads {
		return fmt.Errorf("tqkvprofilebench: layer %d profile heads = %d want %d from capture", profile.Layer, profile.Heads, capture.Heads)
	}
	if profile.KVHeadCount() != kvHeadsForCapture(capture) {
		return fmt.Errorf("tqkvprofilebench: layer %d profile kv_heads = %d want %d from capture", profile.Layer, profile.KVHeadCount(), kvHeadsForCapture(capture))
	}
	if profile.HeadDim != capture.HeadDim {
		return fmt.Errorf("tqkvprofilebench: layer %d profile head_dim = %d want %d from capture", profile.Layer, profile.HeadDim, capture.HeadDim)
	}
	if layer.Layer != profile.Layer || layer.Layer != capture.Layer {
		return fmt.Errorf("tqkvprofilebench: layer selection mismatch for layer %d", profile.Layer)
	}
	if layer.Heads != profile.Heads || effectiveKVHeadsForLayer(layer) != profile.KVHeadCount() || layer.HeadDim != profile.HeadDim {
		return fmt.Errorf("tqkvprofilebench: layer summary mismatch for layer %d", profile.Layer)
	}
	return nil
}

type layerRunMetrics struct {
	layer        profileLayer
	tokens       int
	rawFP32      uint64
	rawFP16      uint64
	liveBytes    uint64
	storageBytes uint64
	appendDur    time.Duration
	gpuUploadDur time.Duration
	attentionDur time.Duration
	gpuEnabled   bool
}

type groupRunMetrics struct {
	rawFP32      uint64
	rawFP16      uint64
	liveBytes    uint64
	storageBytes uint64
	appendDur    time.Duration
	gpuUploadDur time.Duration
	attentionDur time.Duration
	totalDur     time.Duration
	appendTokens int
	attentionOps int
	gpuLayerRuns int
	layers       []layerRunMetrics
}

func benchmarkGroupRun(group groupBenchContext, tryGPU bool) (groupRunMetrics, error) {
	start := time.Now()
	profiles := make([]turboquant.TransformerLayerKVProfile, len(group.layers))
	for i := range group.layers {
		profiles[i] = group.layers[i].profile
	}
	stack := turboquant.NewTransformerModelKVCache(profiles)

	metrics := groupRunMetrics{
		layers: make([]layerRunMetrics, 0, len(group.layers)),
	}

	appendStart := time.Now()
	for _, item := range group.layers {
		layerCache := stack.Layer(item.profile.Layer)
		if layerCache == nil {
			return groupRunMetrics{}, fmt.Errorf("tqkvprofilebench: missing runtime cache for layer %d", item.profile.Layer)
		}
		layerStart := time.Now()
		tokens := item.capture.TokenCount()
		tokenDim := kvHeadsForCapture(item.capture) * item.capture.HeadDim
		for token := 0; token < tokens; token++ {
			base := token * tokenDim
			layerCache.Append(
				item.capture.Keys[base:base+tokenDim],
				item.capture.Values[base:base+tokenDim],
			)
		}
		layerDur := time.Since(layerStart)
		rawBytes := rawKVBytes(item.capture)
		metrics.rawFP32 += rawBytes
		metrics.rawFP16 += rawBytes / 2
		metrics.appendTokens += tokens
		metrics.layers = append(metrics.layers, layerRunMetrics{
			layer:        item.layer,
			tokens:       tokens,
			rawFP32:      rawBytes,
			rawFP16:      rawBytes / 2,
			appendDur:    layerDur,
			liveBytes:    layerCache.LiveBytes(),
			storageBytes: layerCache.StorageBytes(),
		})
	}
	metrics.appendDur = time.Since(appendStart)
	metrics.liveBytes = stack.LiveBytes()
	metrics.storageBytes = stack.StorageBytes()

	if tryGPU {
		gpuStart := time.Now()
		for i, item := range group.layers {
			if !supportsGPUPreparedQueryKeyBits(item.profile.KeyBits) {
				continue
			}
			layerCache := stack.Layer(item.profile.Layer)
			err := layerCache.EnableGPUKeys()
			if err == nil {
				metrics.layers[i].gpuEnabled = true
				metrics.gpuLayerRuns++
			} else if !errors.Is(err, turboquant.ErrGPUBackendUnavailable) {
				return groupRunMetrics{}, fmt.Errorf("tqkvprofilebench: enable GPU for layer %d: %w", item.profile.Layer, err)
			}
		}
		metrics.gpuUploadDur = time.Since(gpuStart)
	}

	attentionStart := time.Now()
	for i, item := range group.layers {
		layerCache := stack.Layer(item.profile.Layer)
		layerStart := time.Now()
		if err := item.capture.AttentionOutputFromCacheInto(item.output, layerCache, item.capture.Query, item.layer.TopK); err != nil {
			return groupRunMetrics{}, err
		}
		metrics.layers[i].attentionDur = time.Since(layerStart)
		metrics.layers[i].gpuUploadDur = 0
		if metrics.layers[i].gpuEnabled {
			metrics.layers[i].gpuUploadDur = metrics.gpuUploadDur / time.Duration(maxInt(1, metrics.gpuLayerRuns))
		}
		metrics.layers[i].liveBytes = layerCache.LiveBytes()
		metrics.layers[i].storageBytes = layerCache.StorageBytes()
		metrics.attentionOps++
	}
	metrics.attentionDur = time.Since(attentionStart)
	metrics.totalDur = time.Since(start)
	return metrics, nil
}

func (a *layerAccumulator) add(run layerRunMetrics) {
	a.count++
	a.sumTokens += float64(run.tokens)
	a.sumRawFP32 += float64(run.rawFP32)
	a.sumRawFP16 += float64(run.rawFP16)
	a.sumLive += float64(run.liveBytes)
	a.sumStorage += float64(run.storageBytes)
	a.sumAppendNS += float64(run.appendDur)
	a.sumGPUUploadNS += float64(run.gpuUploadDur)
	a.sumAttentionNS += float64(run.attentionDur)
	if run.appendDur > 0 && run.tokens > 0 {
		a.sumAppendRate += float64(run.tokens) / run.appendDur.Seconds()
	}
	if run.attentionDur > 0 {
		a.sumAttentionQPS += 1 / run.attentionDur.Seconds()
	}
	if run.gpuEnabled {
		a.gpuRuns++
	}
}

func (a *layerAccumulator) summary() layerRuntimeSummary {
	out := layerRuntimeSummary{
		Layer:          a.layer.Layer,
		Heads:          a.layer.Heads,
		KVHeads:        effectiveKVHeadsForLayer(a.layer),
		HeadDim:        a.layer.HeadDim,
		KeyBits:        a.layer.KeyBits,
		ValueBits:      a.layer.ValueBits,
		TopK:           a.layer.TopK,
		Capacity:       a.layer.Capacity,
		Runs:           a.count,
		GPUEnabledRuns: a.gpuRuns,
	}
	if a.count == 0 {
		return out
	}
	out.MeanTokens = a.sumTokens / float64(a.count)
	out.MeanRawFP32Bytes = a.sumRawFP32 / float64(a.count)
	out.MeanRawFP16Bytes = a.sumRawFP16 / float64(a.count)
	out.MeanQuantLiveBytes = a.sumLive / float64(a.count)
	out.MeanQuantStorageBytes = a.sumStorage / float64(a.count)
	out.MeanAppendDurationMS = nsToMS(a.sumAppendNS / float64(a.count))
	out.MeanGPUUploadDurationMS = nsToMS(a.sumGPUUploadNS / float64(a.count))
	out.MeanAttentionDurationMS = nsToMS(a.sumAttentionNS / float64(a.count))
	out.MeanAppendLayerTokensPerSec = a.sumAppendRate / float64(a.count)
	out.MeanAttentionQueriesPerSec = a.sumAttentionQPS / float64(a.count)
	if out.MeanQuantStorageBytes > 0 {
		out.CompressionVsFP32 = out.MeanRawFP32Bytes / out.MeanQuantStorageBytes
		out.CompressionVsFP16 = out.MeanRawFP16Bytes / out.MeanQuantStorageBytes
	}
	return out
}

func (a *summaryAccumulator) add(run groupRunMetrics) {
	a.count++
	a.sumRawFP32 += float64(run.rawFP32)
	a.sumRawFP16 += float64(run.rawFP16)
	a.sumLive += float64(run.liveBytes)
	a.sumStorage += float64(run.storageBytes)
	a.sumAppendNS += float64(run.appendDur)
	a.sumGPUUploadNS += float64(run.gpuUploadDur)
	a.sumAttentionNS += float64(run.attentionDur)
	a.sumTotalNS += float64(run.totalDur)
	if run.appendDur > 0 && run.appendTokens > 0 {
		a.sumAppendRate += float64(run.appendTokens) / run.appendDur.Seconds()
	}
	if run.attentionDur > 0 && run.attentionOps > 0 {
		a.sumAttentionQPS += float64(run.attentionOps) / run.attentionDur.Seconds()
	}
	if run.totalDur > 0 {
		a.sumGroupRate += 1 / run.totalDur.Seconds()
	}
	if run.gpuLayerRuns > 0 {
		a.gpuRuns++
		a.gpuLayerRuns += run.gpuLayerRuns
	}
	a.totalLayerRuns += len(run.layers)
}

func (a *summaryAccumulator) summary() benchSummary {
	out := benchSummary{
		Runs:                a.count,
		TotalLayerRuns:      a.totalLayerRuns,
		GPUEnabledRuns:      a.gpuRuns,
		GPUEnabledLayerRuns: a.gpuLayerRuns,
	}
	if a.count == 0 {
		return out
	}
	out.MeanRawFP32Bytes = a.sumRawFP32 / float64(a.count)
	out.MeanRawFP16Bytes = a.sumRawFP16 / float64(a.count)
	out.MeanQuantLiveBytes = a.sumLive / float64(a.count)
	out.MeanQuantStorageBytes = a.sumStorage / float64(a.count)
	out.MeanAppendDurationMS = nsToMS(a.sumAppendNS / float64(a.count))
	out.MeanGPUUploadDurationMS = nsToMS(a.sumGPUUploadNS / float64(a.count))
	out.MeanAttentionDurationMS = nsToMS(a.sumAttentionNS / float64(a.count))
	out.MeanTotalDurationMS = nsToMS(a.sumTotalNS / float64(a.count))
	out.MeanAppendLayerTokensPerSec = a.sumAppendRate / float64(a.count)
	out.MeanAttentionQueriesPerSec = a.sumAttentionQPS / float64(a.count)
	out.MeanGroupsPerSec = a.sumGroupRate / float64(a.count)
	if out.MeanQuantStorageBytes > 0 {
		out.CompressionVsFP32 = out.MeanRawFP32Bytes / out.MeanQuantStorageBytes
		out.CompressionVsFP16 = out.MeanRawFP16Bytes / out.MeanQuantStorageBytes
	}
	return out
}

func rawKVBytes(sample turboquant.TransformerLayerCapture) uint64 {
	return uint64(len(sample.Keys)+len(sample.Values)) * 4
}

func kvHeadsForCapture(sample turboquant.TransformerLayerCapture) int {
	if sample.KVHeads > 0 {
		return sample.KVHeads
	}
	return sample.Heads
}

func effectiveKVHeadsForLayer(layer profileLayer) int {
	if layer.KVHeads > 0 {
		return layer.KVHeads
	}
	return layer.Heads
}

// expandProfileToCapture returns a copy of the profile with entries for every
// layer in captureLayers. Layers already present in the profile are kept as-is.
// Missing layers clone both the profileLayer and TransformerLayerKVProfile from
// the nearest existing profile layer (ties broken by preferring the lower index).
func expandProfileToCapture(profile profileReportInput, captureLayers []int) profileReportInput {
	existingLayers := make(map[int]struct{}, len(profile.Layers))
	for _, layer := range profile.Layers {
		existingLayers[layer.Layer] = struct{}{}
	}

	// Build sorted list of profiled layer IDs for nearest-neighbor lookup.
	profiledIDs := make([]int, 0, len(profile.Layers))
	for _, layer := range profile.Layers {
		profiledIDs = append(profiledIDs, layer.Layer)
	}
	sort.Ints(profiledIDs)

	// Index donors by layer ID.
	layerByID := make(map[int]profileLayer, len(profile.Layers))
	for _, layer := range profile.Layers {
		layerByID[layer.Layer] = layer
	}
	profileByID := make(map[int]turboquant.TransformerLayerKVProfile, len(profile.Profiles))
	for _, p := range profile.Profiles {
		profileByID[p.Layer] = p
	}

	out := profileReportInput{
		ProfileSet:     profile.ProfileSet,
		TokenPositions: append([]string(nil), profile.TokenPositions...),
		Layers:         make([]profileLayer, 0, len(captureLayers)),
		Profiles:       make([]turboquant.TransformerLayerKVProfile, 0, len(captureLayers)),
	}

	sorted := make([]int, len(captureLayers))
	copy(sorted, captureLayers)
	sort.Ints(sorted)

	for _, targetLayer := range sorted {
		if _, exists := existingLayers[targetLayer]; exists {
			out.Layers = append(out.Layers, layerByID[targetLayer])
			out.Profiles = append(out.Profiles, profileByID[targetLayer])
			continue
		}
		donorID := nearestProfiledLayer(targetLayer, profiledIDs)
		donorLayer := layerByID[donorID]
		donorProfile := profileByID[donorID]

		clonedLayer := donorLayer
		clonedLayer.Layer = targetLayer
		clonedProfile := donorProfile
		clonedProfile.Layer = targetLayer

		out.Layers = append(out.Layers, clonedLayer)
		out.Profiles = append(out.Profiles, clonedProfile)
	}
	return out
}

// nearestProfiledLayer returns the profiled layer ID closest to target.
// On ties, the lower index wins. Panics if profiledIDs is empty.
func nearestProfiledLayer(target int, profiledIDs []int) int {
	if len(profiledIDs) == 0 {
		panic("turboquant: nearestProfiledLayer called with empty profiledIDs")
	}
	best := profiledIDs[0]
	bestDist := abs(target - best)
	for _, id := range profiledIDs[1:] {
		dist := abs(target - id)
		if dist < bestDist {
			best = id
			bestDist = dist
		}
		// ties: keep the first (lower) id since profiledIDs is sorted ascending
	}
	return best
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func nsToMS(ns float64) float64 {
	return ns / float64(time.Millisecond)
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

func profileSetRank(label string) int {
	if label == "all" {
		return -1
	}
	return relativePositionRank(label)
}

func maxInt(left, right int) int {
	if left > right {
		return left
	}
	return right
}

func supportsGPUPreparedQueryKeyBits(keyBits int) bool {
	switch keyBits - 1 {
	case 1, 2, 4, 8:
		return true
	default:
		return false
	}
}
