package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	defaultPerplexityTimeout = 30 * time.Minute
	defaultBenchTimeout      = 30 * time.Minute
	outputTailLimit          = 4096
)

var (
	finalPPLPattern    = regexp.MustCompile(`Final estimate:\s*PPL\s*=\s*([0-9eE+.\-]+)\s*\+/-\s*([0-9eE+.\-]+)`)
	legacyPPLPattern   = regexp.MustCompile(`perplexity:\s*([0-9eE+.\-]+)\s*(?:\[[^\]]+\])?`)
	kvTotalLinePattern = regexp.MustCompile(`size\s*=\s*([0-9eE+.\-]+)\s+MiB.*K\s+\(([^)]+)\):\s*([0-9eE+.\-]+)\s+MiB,\s*V\s+\(([^)]+)\):\s*([0-9eE+.\-]+)\s+MiB`)
	kvBufferPattern    = regexp.MustCompile(`([A-Za-z0-9_.:+-]+)\s+KV buffer size\s*=\s*([0-9eE+.\-]+)\s+MiB`)
)

type config struct {
	Runs []runConfig `json:"runs"`
}

type runConfig struct {
	Name          string            `json:"name"`
	Workdir       string            `json:"workdir,omitempty"`
	Env           map[string]string `json:"env,omitempty"`
	Model         string            `json:"model"`
	CommonArgs    []string          `json:"common_args,omitempty"`
	PerplexityBin string            `json:"perplexity_bin,omitempty"`
	BenchBin      string            `json:"bench_bin,omitempty"`
	Baseline      string            `json:"baseline,omitempty"`
	KVTypes       []kvTypeConfig    `json:"kv_types"`
	Perplexity    *perplexityConfig `json:"perplexity,omitempty"`
	Bench         *benchConfig      `json:"bench,omitempty"`
}

type kvTypeConfig struct {
	Name  string   `json:"name"`
	TypeK string   `json:"type_k,omitempty"`
	TypeV string   `json:"type_v,omitempty"`
	Args  []string `json:"args,omitempty"`
}

type perplexityConfig struct {
	Dataset   string   `json:"dataset"`
	Contexts  []int    `json:"ctx_sizes"`
	BatchSize int      `json:"batch_size,omitempty"`
	Chunks    int      `json:"chunks,omitempty"`
	Timeout   string   `json:"timeout,omitempty"`
	ExtraArgs []string `json:"extra_args,omitempty"`
}

type benchConfig struct {
	PromptTokens []int    `json:"n_prompt,omitempty"`
	GenTokens    []int    `json:"n_gen,omitempty"`
	Depths       []int    `json:"n_depth,omitempty"`
	Repetitions  int      `json:"repetitions,omitempty"`
	BatchSize    int      `json:"batch_size,omitempty"`
	UBatchSize   int      `json:"ubatch_size,omitempty"`
	Threads      []int    `json:"threads,omitempty"`
	NGPULayers   []int    `json:"n_gpu_layers,omitempty"`
	Timeout      string   `json:"timeout,omitempty"`
	ExtraArgs    []string `json:"extra_args,omitempty"`
}

type report struct {
	GeneratedAt string      `json:"generated_at"`
	Runs        []runReport `json:"runs"`
}

type runReport struct {
	Name        string              `json:"name"`
	Model       string              `json:"model"`
	Baseline    string              `json:"baseline,omitempty"`
	KVVariants  []kvVariantReport   `json:"kv_variants"`
	Comparisons []variantComparison `json:"comparisons,omitempty"`
}

type kvVariantReport struct {
	Name       string                `json:"name"`
	TypeK      string                `json:"type_k,omitempty"`
	TypeV      string                `json:"type_v,omitempty"`
	Perplexity []perplexityRunReport `json:"perplexity,omitempty"`
	Bench      *benchRunReport       `json:"bench,omitempty"`
}

type perplexityRunReport struct {
	ContextSize int            `json:"context_size"`
	Command     commandSummary `json:"command"`
	DurationMS  int64          `json:"duration_ms"`
	PPL         *pplMetric     `json:"ppl,omitempty"`
	KV          *kvMemoryStats `json:"kv,omitempty"`
	Error       string         `json:"error,omitempty"`
	StdoutTail  string         `json:"stdout_tail,omitempty"`
	StderrTail  string         `json:"stderr_tail,omitempty"`
}

type benchRunReport struct {
	Command    commandSummary `json:"command"`
	DurationMS int64          `json:"duration_ms"`
	Samples    []benchSample  `json:"samples,omitempty"`
	Error      string         `json:"error,omitempty"`
	StdoutTail string         `json:"stdout_tail,omitempty"`
	StderrTail string         `json:"stderr_tail,omitempty"`
}

type commandSummary struct {
	Executable string   `json:"executable"`
	Args       []string `json:"args"`
	Workdir    string   `json:"workdir,omitempty"`
}

type pplMetric struct {
	Mean   float64 `json:"mean"`
	Stddev float64 `json:"stddev,omitempty"`
}

type kvMemoryStats struct {
	TotalMiB  float64        `json:"total_mib,omitempty"`
	KeyType   string         `json:"key_type,omitempty"`
	KeyMiB    float64        `json:"key_mib,omitempty"`
	ValueType string         `json:"value_type,omitempty"`
	ValueMiB  float64        `json:"value_mib,omitempty"`
	BufferMiB float64        `json:"buffer_mib,omitempty"`
	Buffers   []kvBufferStat `json:"buffers,omitempty"`
}

type kvBufferStat struct {
	Name string  `json:"name"`
	MiB  float64 `json:"mib"`
}

type benchSample struct {
	BuildCommit   string    `json:"build_commit,omitempty"`
	BuildNumber   int       `json:"build_number,omitempty"`
	CPUInfo       string    `json:"cpu_info,omitempty"`
	GPUInfo       string    `json:"gpu_info,omitempty"`
	Backends      string    `json:"backends,omitempty"`
	ModelFilename string    `json:"model_filename,omitempty"`
	ModelType     string    `json:"model_type,omitempty"`
	ModelSize     int64     `json:"model_size,omitempty"`
	ModelParams   int64     `json:"model_n_params,omitempty"`
	BatchSize     int       `json:"n_batch,omitempty"`
	UBatchSize    int       `json:"n_ubatch,omitempty"`
	Threads       int       `json:"n_threads,omitempty"`
	TypeK         string    `json:"type_k,omitempty"`
	TypeV         string    `json:"type_v,omitempty"`
	NGPULayers    int       `json:"n_gpu_layers,omitempty"`
	SplitMode     string    `json:"split_mode,omitempty"`
	MainGPU       int       `json:"main_gpu,omitempty"`
	NoKVOffload   bool      `json:"no_kv_offload,omitempty"`
	FlashAttn     bool      `json:"flash_attn,omitempty"`
	UseMmap       bool      `json:"use_mmap,omitempty"`
	Embeddings    bool      `json:"embeddings,omitempty"`
	NPrompt       int       `json:"n_prompt,omitempty"`
	NGen          int       `json:"n_gen,omitempty"`
	NDepth        int       `json:"n_depth,omitempty"`
	TestTime      string    `json:"test_time,omitempty"`
	AvgNS         int64     `json:"avg_ns,omitempty"`
	StddevNS      int64     `json:"stddev_ns,omitempty"`
	AvgTS         float64   `json:"avg_ts,omitempty"`
	StddevTS      float64   `json:"stddev_ts,omitempty"`
	SamplesTS     []float64 `json:"samples_ts,omitempty"`
	Kind          string    `json:"kind,omitempty"`
}

type variantComparison struct {
	Name       string                 `json:"name"`
	Against    string                 `json:"against"`
	Perplexity []perplexityComparison `json:"perplexity,omitempty"`
	Bench      []benchComparison      `json:"bench,omitempty"`
}

type perplexityComparison struct {
	ContextSize     int     `json:"context_size"`
	BaselinePPL     float64 `json:"baseline_ppl"`
	CandidatePPL    float64 `json:"candidate_ppl"`
	DeltaPPL        float64 `json:"delta_ppl"`
	RatioPPL        float64 `json:"ratio_ppl"`
	DeltaTotalKVMiB float64 `json:"delta_total_kv_mib,omitempty"`
}

type benchComparison struct {
	Kind          string  `json:"kind"`
	NPrompt       int     `json:"n_prompt"`
	NGen          int     `json:"n_gen"`
	NDepth        int     `json:"n_depth"`
	BaselineTS    float64 `json:"baseline_ts"`
	CandidateTS   float64 `json:"candidate_ts"`
	DeltaTS       float64 `json:"delta_ts"`
	RelativeDelta float64 `json:"relative_delta"`
}

type commandOutput struct {
	stdout   string
	stderr   string
	duration time.Duration
}

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runCLI(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("tqkvbench", flag.ContinueOnError)
	fs.SetOutput(stderr)

	configPath := fs.String("config", "", "path to tqkvbench JSON config")
	outputPath := fs.String("out", "", "path to write benchmark report JSON")
	failFast := fs.Bool("fail-fast", false, "stop after the first command failure")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if strings.TrimSpace(*configPath) == "" {
		return errors.New("tqkvbench: --config is required")
	}

	cfg, err := loadConfig(*configPath)
	if err != nil {
		return err
	}

	rep, err := runBenchmarks(cfg, *failFast)
	if err != nil {
		return err
	}

	payload, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return fmt.Errorf("tqkvbench: encode report: %w", err)
	}

	if strings.TrimSpace(*outputPath) == "" {
		_, err = stdout.Write(append(payload, '\n'))
		return err
	}
	if err := os.WriteFile(*outputPath, append(payload, '\n'), 0o644); err != nil {
		return fmt.Errorf("tqkvbench: write report: %w", err)
	}
	fmt.Fprintf(stdout, "wrote %s\n", *outputPath)
	return nil
}

func loadConfig(path string) (config, error) {
	file, err := os.Open(path)
	if err != nil {
		return config{}, fmt.Errorf("tqkvbench: open config: %w", err)
	}
	defer file.Close()

	var cfg config
	if err := json.NewDecoder(file).Decode(&cfg); err != nil {
		return config{}, fmt.Errorf("tqkvbench: decode config: %w", err)
	}

	baseDir := filepath.Dir(path)
	for i := range cfg.Runs {
		run := &cfg.Runs[i]
		run.Workdir = resolvePath(baseDir, run.Workdir)
		run.Model = resolvePath(baseDir, run.Model)
		run.PerplexityBin = resolvePath(baseDir, run.PerplexityBin)
		run.BenchBin = resolvePath(baseDir, run.BenchBin)
		if run.Perplexity != nil {
			run.Perplexity.Dataset = resolvePath(baseDir, run.Perplexity.Dataset)
		}
		for j := range run.KVTypes {
			if strings.TrimSpace(run.KVTypes[j].Name) == "" {
				run.KVTypes[j].Name = defaultKVName(run.KVTypes[j])
			}
		}
	}

	if err := validateConfig(cfg); err != nil {
		return config{}, err
	}
	return cfg, nil
}

func validateConfig(cfg config) error {
	if len(cfg.Runs) == 0 {
		return errors.New("tqkvbench: config must include at least one run")
	}
	for _, run := range cfg.Runs {
		if strings.TrimSpace(run.Name) == "" {
			return errors.New("tqkvbench: each run requires a name")
		}
		if strings.TrimSpace(run.Model) == "" {
			return fmt.Errorf("tqkvbench: run %q requires a model path", run.Name)
		}
		if len(run.KVTypes) == 0 {
			return fmt.Errorf("tqkvbench: run %q requires at least one kv_types entry", run.Name)
		}
		seen := make(map[string]struct{}, len(run.KVTypes))
		for _, kv := range run.KVTypes {
			if _, ok := seen[kv.Name]; ok {
				return fmt.Errorf("tqkvbench: run %q has duplicate kv variant %q", run.Name, kv.Name)
			}
			seen[kv.Name] = struct{}{}
		}
		if run.Baseline != "" {
			if _, ok := seen[run.Baseline]; !ok {
				return fmt.Errorf("tqkvbench: run %q baseline %q does not match any kv variant", run.Name, run.Baseline)
			}
		}
		if run.Perplexity == nil && run.Bench == nil {
			return fmt.Errorf("tqkvbench: run %q must configure perplexity and/or bench", run.Name)
		}
		if run.Perplexity != nil {
			if strings.TrimSpace(run.PerplexityBin) == "" {
				return fmt.Errorf("tqkvbench: run %q requires perplexity_bin when perplexity is configured", run.Name)
			}
			if strings.TrimSpace(run.Perplexity.Dataset) == "" {
				return fmt.Errorf("tqkvbench: run %q requires perplexity.dataset", run.Name)
			}
			if len(run.Perplexity.Contexts) == 0 {
				return fmt.Errorf("tqkvbench: run %q requires at least one perplexity ctx_sizes entry", run.Name)
			}
		}
		if run.Bench != nil {
			if strings.TrimSpace(run.BenchBin) == "" {
				return fmt.Errorf("tqkvbench: run %q requires bench_bin when bench is configured", run.Name)
			}
			if len(run.Bench.PromptTokens) == 0 && len(run.Bench.GenTokens) == 0 {
				return fmt.Errorf("tqkvbench: run %q requires bench n_prompt and/or n_gen values", run.Name)
			}
		}
	}
	return nil
}

func runBenchmarks(cfg config, failFast bool) (report, error) {
	rep := report{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Runs:        make([]runReport, 0, len(cfg.Runs)),
	}
	for _, run := range cfg.Runs {
		runRep, err := runBenchmark(run, failFast)
		if err != nil {
			return report{}, err
		}
		rep.Runs = append(rep.Runs, runRep)
	}
	return rep, nil
}

func runBenchmark(run runConfig, failFast bool) (runReport, error) {
	runRep := runReport{
		Name:       run.Name,
		Model:      run.Model,
		Baseline:   run.Baseline,
		KVVariants: make([]kvVariantReport, 0, len(run.KVTypes)),
	}
	for _, kv := range run.KVTypes {
		kvRep, err := runKVVariant(run, kv)
		if err != nil && failFast {
			return runReport{}, err
		}
		runRep.KVVariants = append(runRep.KVVariants, kvRep)
	}
	runRep.Comparisons = compareAgainstBaseline(runRep)
	return runRep, nil
}

func runKVVariant(run runConfig, kv kvTypeConfig) (kvVariantReport, error) {
	kvRep := kvVariantReport{
		Name:  kv.Name,
		TypeK: kv.TypeK,
		TypeV: kv.TypeV,
	}
	var firstErr error

	if run.Perplexity != nil {
		kvRep.Perplexity = make([]perplexityRunReport, 0, len(run.Perplexity.Contexts))
		for _, ctxSize := range run.Perplexity.Contexts {
			item, err := runPerplexity(run, kv, ctxSize)
			if err != nil && firstErr == nil {
				firstErr = err
			}
			kvRep.Perplexity = append(kvRep.Perplexity, item)
		}
	}
	if run.Bench != nil {
		item, err := runLlamaBench(run, kv)
		if err != nil && firstErr == nil {
			firstErr = err
		}
		kvRep.Bench = &item
	}

	return kvRep, firstErr
}

func runPerplexity(run runConfig, kv kvTypeConfig, ctxSize int) (perplexityRunReport, error) {
	timeout, err := parseDurationDefault(run.Perplexity.Timeout, defaultPerplexityTimeout)
	if err != nil {
		return perplexityRunReport{}, fmt.Errorf("tqkvbench: run %q perplexity timeout: %w", run.Name, err)
	}

	args := []string{"-m", run.Model, "-f", run.Perplexity.Dataset, "-c", strconv.Itoa(ctxSize)}
	if run.Perplexity.BatchSize > 0 {
		args = append(args, "-b", strconv.Itoa(run.Perplexity.BatchSize))
	}
	if run.Perplexity.Chunks > 0 {
		args = append(args, "--chunks", strconv.Itoa(run.Perplexity.Chunks))
	}
	args = append(args, kvArgs(kv)...)
	args = append(args, run.CommonArgs...)
	args = append(args, run.Perplexity.ExtraArgs...)

	cmd := commandSummary{Executable: run.PerplexityBin, Args: args, Workdir: run.Workdir}
	out, execErr := runCommand(timeout, cmd, run.Env)
	rep := perplexityRunReport{
		ContextSize: ctxSize,
		Command:     cmd,
		DurationMS:  out.duration.Milliseconds(),
		StdoutTail:  tail(out.stdout, outputTailLimit),
		StderrTail:  tail(out.stderr, outputTailLimit),
	}

	combined := strings.TrimSpace(out.stdout + "\n" + out.stderr)
	if ppl, parseErr := parsePerplexity(combined); parseErr == nil {
		rep.PPL = &ppl
	} else if execErr == nil {
		execErr = parseErr
	}
	if kvStats := parseKVStats(combined); kvStats != nil {
		rep.KV = kvStats
	}

	if execErr != nil {
		rep.Error = execErr.Error()
		return rep, execErr
	}
	return rep, nil
}

func runLlamaBench(run runConfig, kv kvTypeConfig) (benchRunReport, error) {
	timeout, err := parseDurationDefault(run.Bench.Timeout, defaultBenchTimeout)
	if err != nil {
		return benchRunReport{}, fmt.Errorf("tqkvbench: run %q bench timeout: %w", run.Name, err)
	}

	args := []string{"-o", "json", "-m", run.Model}
	if len(run.Bench.PromptTokens) > 0 {
		args = append(args, "-p", intsCSV(run.Bench.PromptTokens))
	}
	if len(run.Bench.GenTokens) > 0 {
		args = append(args, "-n", intsCSV(run.Bench.GenTokens))
	}
	if len(run.Bench.Depths) > 0 {
		args = append(args, "-d", intsCSV(run.Bench.Depths))
	}
	if run.Bench.Repetitions > 0 {
		args = append(args, "-r", strconv.Itoa(run.Bench.Repetitions))
	}
	if run.Bench.BatchSize > 0 {
		args = append(args, "-b", strconv.Itoa(run.Bench.BatchSize))
	}
	if run.Bench.UBatchSize > 0 {
		args = append(args, "-ub", strconv.Itoa(run.Bench.UBatchSize))
	}
	if len(run.Bench.Threads) > 0 {
		args = append(args, "-t", intsCSV(run.Bench.Threads))
	}
	if len(run.Bench.NGPULayers) > 0 {
		args = append(args, "-ngl", intsCSV(run.Bench.NGPULayers))
	}
	args = append(args, kvArgs(kv)...)
	args = append(args, run.CommonArgs...)
	args = append(args, run.Bench.ExtraArgs...)

	cmd := commandSummary{Executable: run.BenchBin, Args: args, Workdir: run.Workdir}
	out, execErr := runCommand(timeout, cmd, run.Env)
	rep := benchRunReport{
		Command:    cmd,
		DurationMS: out.duration.Milliseconds(),
		StdoutTail: tail(out.stdout, outputTailLimit),
		StderrTail: tail(out.stderr, outputTailLimit),
	}

	if samples, parseErr := parseBenchSamples(out.stdout); parseErr == nil {
		rep.Samples = samples
	} else if execErr == nil {
		execErr = parseErr
	}

	if execErr != nil {
		rep.Error = execErr.Error()
		return rep, execErr
	}
	return rep, nil
}

func runCommand(timeout time.Duration, cmd commandSummary, env map[string]string) (commandOutput, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	command := exec.CommandContext(ctx, cmd.Executable, cmd.Args...)
	command.Dir = cmd.Workdir
	command.Env = mergeEnv(env)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	command.Stdout = &stdout
	command.Stderr = &stderr

	start := time.Now()
	err := command.Run()
	out := commandOutput{
		stdout:   stdout.String(),
		stderr:   stderr.String(),
		duration: time.Since(start),
	}
	if ctx.Err() == context.DeadlineExceeded {
		return out, fmt.Errorf("command timed out after %s", timeout)
	}
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return out, fmt.Errorf("command exited with code %d", exitErr.ExitCode())
		}
		return out, err
	}
	return out, nil
}

func parsePerplexity(text string) (pplMetric, error) {
	if match := finalPPLPattern.FindStringSubmatch(text); len(match) == 3 {
		mean, err := strconv.ParseFloat(match[1], 64)
		if err != nil {
			return pplMetric{}, fmt.Errorf("parse final ppl mean: %w", err)
		}
		stddev, err := strconv.ParseFloat(match[2], 64)
		if err != nil {
			return pplMetric{}, fmt.Errorf("parse final ppl stddev: %w", err)
		}
		return pplMetric{Mean: mean, Stddev: stddev}, nil
	}
	matches := legacyPPLPattern.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return pplMetric{}, errors.New("unable to parse perplexity output")
	}
	last := matches[len(matches)-1]
	mean, err := strconv.ParseFloat(last[1], 64)
	if err != nil {
		return pplMetric{}, fmt.Errorf("parse legacy ppl mean: %w", err)
	}
	return pplMetric{Mean: mean}, nil
}

func parseKVStats(text string) *kvMemoryStats {
	var stats kvMemoryStats
	if matches := kvTotalLinePattern.FindAllStringSubmatch(text, -1); len(matches) > 0 {
		last := matches[len(matches)-1]
		stats.TotalMiB, _ = strconv.ParseFloat(last[1], 64)
		stats.KeyType = strings.TrimSpace(last[2])
		stats.KeyMiB, _ = strconv.ParseFloat(last[3], 64)
		stats.ValueType = strings.TrimSpace(last[4])
		stats.ValueMiB, _ = strconv.ParseFloat(last[5], 64)
	}
	bufferMatches := kvBufferPattern.FindAllStringSubmatch(text, -1)
	if len(bufferMatches) > 0 {
		stats.Buffers = make([]kvBufferStat, 0, len(bufferMatches))
		for _, match := range bufferMatches {
			mib, _ := strconv.ParseFloat(match[2], 64)
			stats.BufferMiB += mib
			stats.Buffers = append(stats.Buffers, kvBufferStat{Name: strings.TrimSpace(match[1]), MiB: mib})
		}
	}
	if stats.TotalMiB == 0 && stats.BufferMiB == 0 {
		return nil
	}
	return &stats
}

func parseBenchSamples(stdout string) ([]benchSample, error) {
	var samples []benchSample
	if err := json.Unmarshal([]byte(stdout), &samples); err != nil {
		return nil, fmt.Errorf("parse llama-bench json: %w", err)
	}
	for i := range samples {
		samples[i].Kind = benchKind(samples[i])
	}
	sort.Slice(samples, func(i, j int) bool {
		if samples[i].Kind != samples[j].Kind {
			return samples[i].Kind < samples[j].Kind
		}
		if samples[i].NDepth != samples[j].NDepth {
			return samples[i].NDepth < samples[j].NDepth
		}
		if samples[i].NPrompt != samples[j].NPrompt {
			return samples[i].NPrompt < samples[j].NPrompt
		}
		return samples[i].NGen < samples[j].NGen
	})
	return samples, nil
}

func compareAgainstBaseline(run runReport) []variantComparison {
	if run.Baseline == "" {
		return nil
	}
	var baseline *kvVariantReport
	for i := range run.KVVariants {
		if run.KVVariants[i].Name == run.Baseline {
			baseline = &run.KVVariants[i]
			break
		}
	}
	if baseline == nil {
		return nil
	}

	perplexByCtx := make(map[int]perplexityRunReport)
	for _, item := range baseline.Perplexity {
		if item.Error == "" && item.PPL != nil {
			perplexByCtx[item.ContextSize] = item
		}
	}
	benchByKey := make(map[string]benchSample)
	if baseline.Bench != nil && baseline.Bench.Error == "" {
		for _, sample := range baseline.Bench.Samples {
			benchByKey[benchKey(sample)] = sample
		}
	}

	comparisons := make([]variantComparison, 0, len(run.KVVariants))
	for _, candidate := range run.KVVariants {
		if candidate.Name == run.Baseline {
			continue
		}
		comp := variantComparison{Name: candidate.Name, Against: run.Baseline}
		for _, item := range candidate.Perplexity {
			base, ok := perplexByCtx[item.ContextSize]
			if !ok || item.Error != "" || item.PPL == nil {
				continue
			}
			deltaKV := 0.0
			if base.KV != nil && item.KV != nil {
				deltaKV = item.KV.TotalMiB - base.KV.TotalMiB
			}
			comp.Perplexity = append(comp.Perplexity, perplexityComparison{
				ContextSize:     item.ContextSize,
				BaselinePPL:     base.PPL.Mean,
				CandidatePPL:    item.PPL.Mean,
				DeltaPPL:        item.PPL.Mean - base.PPL.Mean,
				RatioPPL:        item.PPL.Mean / base.PPL.Mean,
				DeltaTotalKVMiB: deltaKV,
			})
		}
		if candidate.Bench != nil && candidate.Bench.Error == "" {
			for _, sample := range candidate.Bench.Samples {
				base, ok := benchByKey[benchKey(sample)]
				if !ok || base.AvgTS == 0 {
					continue
				}
				comp.Bench = append(comp.Bench, benchComparison{
					Kind:          sample.Kind,
					NPrompt:       sample.NPrompt,
					NGen:          sample.NGen,
					NDepth:        sample.NDepth,
					BaselineTS:    base.AvgTS,
					CandidateTS:   sample.AvgTS,
					DeltaTS:       sample.AvgTS - base.AvgTS,
					RelativeDelta: (sample.AvgTS - base.AvgTS) / base.AvgTS,
				})
			}
		}
		if len(comp.Perplexity) > 0 || len(comp.Bench) > 0 {
			comparisons = append(comparisons, comp)
		}
	}
	return comparisons
}

func benchKey(sample benchSample) string {
	return fmt.Sprintf("%s:%d:%d:%d", sample.Kind, sample.NPrompt, sample.NGen, sample.NDepth)
}

func benchKind(sample benchSample) string {
	switch {
	case sample.NPrompt > 0 && sample.NGen == 0:
		return "pp"
	case sample.NPrompt == 0 && sample.NGen > 0:
		return "tg"
	case sample.NPrompt > 0 && sample.NGen > 0:
		return "pg"
	default:
		return "unknown"
	}
}

func kvArgs(kv kvTypeConfig) []string {
	args := make([]string, 0, 6)
	if strings.TrimSpace(kv.TypeK) != "" {
		args = append(args, "-ctk", kv.TypeK)
	}
	if strings.TrimSpace(kv.TypeV) != "" {
		args = append(args, "-ctv", kv.TypeV)
	}
	args = append(args, kv.Args...)
	return args
}

func parseDurationDefault(raw string, fallback time.Duration) (time.Duration, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fallback, nil
	}
	return time.ParseDuration(raw)
}

func intsCSV(values []int) string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		parts = append(parts, strconv.Itoa(value))
	}
	return strings.Join(parts, ",")
}

func mergeEnv(extra map[string]string) []string {
	env := os.Environ()
	if len(extra) == 0 {
		return env
	}
	env = append([]string(nil), env...)
	for key, value := range extra {
		prefix := key + "="
		replaced := false
		for i := range env {
			if strings.HasPrefix(env[i], prefix) {
				env[i] = prefix + value
				replaced = true
				break
			}
		}
		if !replaced {
			env = append(env, prefix+value)
		}
	}
	return env
}

func resolvePath(baseDir, value string) string {
	value = strings.TrimSpace(value)
	if value == "" || filepath.IsAbs(value) {
		return value
	}
	return filepath.Join(baseDir, value)
}

func tail(text string, limit int) string {
	text = strings.TrimSpace(text)
	if len(text) <= limit {
		return text
	}
	return text[len(text)-limit:]
}

func defaultKVName(kv kvTypeConfig) string {
	switch {
	case kv.TypeK != "" && kv.TypeV != "":
		return kv.TypeK + "/" + kv.TypeV
	case kv.TypeV != "":
		return kv.TypeV
	case kv.TypeK != "":
		return kv.TypeK
	default:
		return "default"
	}
}
