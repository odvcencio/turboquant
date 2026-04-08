package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestParseCSVInts(t *testing.T) {
	got, err := parseCSVInts("4, 2, 4, 8")
	if err != nil {
		t.Fatalf("parseCSVInts: %v", err)
	}
	want := []int{4, 2, 8}
	if len(got) != len(want) {
		t.Fatalf("len = %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %d want %d", i, got[i], want[i])
		}
	}
}

func TestRunCLIWritesSweepReport(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "capture.json")
	outputPath := filepath.Join(dir, "report.json")
	input := `{
  "name": "layer0-step4",
  "model": "tiny",
  "prompt_index": 2,
  "layer": 7,
  "token_index": 4,
  "heads": 1,
  "head_dim": 2,
  "query": [1, 0],
  "keys": [1, 0, 0, 1],
  "values": [2, 0, 0, 4]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}

	if err := runCLI([]string{
		"--input", inputPath,
		"--out", outputPath,
		"--key-bits", "2,3",
		"--value-bits", "2",
		"--top-k", "1,2",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got report
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if len(got.Samples) != 1 {
		t.Fatalf("samples = %d want 1", len(got.Samples))
	}
	if got.Samples[0].Model != "tiny" || got.Samples[0].PromptIndex != 2 || got.Samples[0].Layer != 7 || got.Samples[0].TokenIndex != 4 {
		t.Fatalf("sample metadata = (%q,%d,%d,%d) want (%q,%d,%d,%d)", got.Samples[0].Model, got.Samples[0].PromptIndex, got.Samples[0].Layer, got.Samples[0].TokenIndex, "tiny", 2, 7, 4)
	}
	if len(got.Samples[0].Cases) != 4 {
		t.Fatalf("cases = %d want 4", len(got.Samples[0].Cases))
	}
	if got.Samples[0].Cases[0].Method != "turboquant" {
		t.Fatalf("cases[0].method = %q want %q", got.Samples[0].Cases[0].Method, "turboquant")
	}
	if got.Samples[0].BestByMSE.Name != "layer0-step4" {
		t.Fatalf("best_by_mse.name = %q want %q", got.Samples[0].BestByMSE.Name, "layer0-step4")
	}
	if len(got.Configurations) != 4 {
		t.Fatalf("configurations = %d want 4", len(got.Configurations))
	}
	if got.Configurations[0].Samples != 1 {
		t.Fatalf("configurations[0].samples = %d want 1", got.Configurations[0].Samples)
	}
	if got.Configurations[0].Method != "turboquant" {
		t.Fatalf("configurations[0].method = %q want %q", got.Configurations[0].Method, "turboquant")
	}
	if got.Configurations[0].MeanRawKVBytes <= 0 {
		t.Fatalf("configurations[0].mean_raw_kv_bytes = %v want > 0", got.Configurations[0].MeanRawKVBytes)
	}
	if len(got.ParetoFrontier) == 0 {
		t.Fatal("pareto_frontier = 0 want > 0")
	}
	if got.ParetoFrontier[0].MeanStorageBytes <= 0 {
		t.Fatalf("pareto_frontier[0].mean_cache_storage_bytes = %v want > 0", got.ParetoFrontier[0].MeanStorageBytes)
	}
}

func TestRunCLISupportsWrappedGroupedKVCaptures(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "captures.json")
	outputPath := filepath.Join(dir, "report.json")
	input := `{
  "model": "tiny",
  "samples": [
    {
      "name": "gqa",
      "heads": 2,
      "kv_heads": 1,
      "head_dim": 2,
      "query": [1, 0, 0, 1],
      "keys": [1, 0, 0, 1],
      "values": [2, 0, 0, 4]
    }
  ]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}

	if err := runCLI([]string{
		"--input", inputPath,
		"--out", outputPath,
		"--methods", "uniform",
		"--key-bits", "2",
		"--value-bits", "2",
		"--top-k", "1",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got report
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if len(got.Samples) != 1 {
		t.Fatalf("len(got.Samples) = %d want 1", len(got.Samples))
	}
	if got.Samples[0].KVHeads != 1 {
		t.Fatalf("got.Samples[0].KVHeads = %d want 1", got.Samples[0].KVHeads)
	}
	if len(got.Samples[0].Cases) != 1 {
		t.Fatalf("len(got.Samples[0].Cases) = %d want 1", len(got.Samples[0].Cases))
	}
	if got.Samples[0].Cases[0].KVHeads != 1 {
		t.Fatalf("got.Samples[0].Cases[0].KVHeads = %d want 1", got.Samples[0].Cases[0].KVHeads)
	}
}

func TestParseCSVStrings(t *testing.T) {
	got, err := parseCSVStrings("uniform, turboquant, uniform")
	if err != nil {
		t.Fatalf("parseCSVStrings: %v", err)
	}
	want := []string{"uniform", "turboquant"}
	if len(got) != len(want) {
		t.Fatalf("len = %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %q want %q", i, got[i], want[i])
		}
	}
}

func TestParetoFrontier(t *testing.T) {
	frontier := paretoFrontier([]configSummary{
		{KeyBits: 2, ValueBits: 2, TopK: 4, MeanStorageBytes: 100, MeanOutputMSE: 10, MeanOutputCosine: 0.8},
		{KeyBits: 3, ValueBits: 3, TopK: 4, MeanStorageBytes: 120, MeanOutputMSE: 9, MeanOutputCosine: 0.85},
		{KeyBits: 4, ValueBits: 4, TopK: 4, MeanStorageBytes: 100, MeanOutputMSE: 12, MeanOutputCosine: 0.9},
		{KeyBits: 2, ValueBits: 3, TopK: 8, MeanStorageBytes: 110, MeanOutputMSE: 11, MeanOutputCosine: 0.82},
		{KeyBits: 4, ValueBits: 3, TopK: 8, MeanStorageBytes: 120, MeanOutputMSE: 9, MeanOutputCosine: 0.9},
	})
	if len(frontier) != 2 {
		t.Fatalf("len(frontier) = %d want 2", len(frontier))
	}
	if frontier[0].KeyBits != 2 || frontier[0].ValueBits != 2 {
		t.Fatalf("frontier[0] = (%d,%d) want (2,2)", frontier[0].KeyBits, frontier[0].ValueBits)
	}
	if frontier[1].KeyBits != 4 || frontier[1].ValueBits != 3 {
		t.Fatalf("frontier[1] = (%d,%d) want (4,3)", frontier[1].KeyBits, frontier[1].ValueBits)
	}
}
