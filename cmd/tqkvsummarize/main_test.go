package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRunCLIWritesSummary(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "report.json")
	outputPath := filepath.Join(dir, "summary.json")
	input := `{
  "generated_at": "2026-04-07T00:00:00Z",
  "pareto_frontier": [
    {
      "method": "turboquant",
      "key_bits": 2,
      "value_bits": 2,
      "top_k": 8,
      "samples": 2,
      "mean_output_mse": 10,
      "mean_output_cosine": 0.8,
      "mean_cache_storage_bytes": 100,
      "mean_compression_ratio": 12.8
    }
  ],
  "samples": [
    {
      "model": "tiny",
      "name": "s0",
      "prompt_index": 0,
      "layer": 1,
      "token_index": 127,
      "token_position": "last",
      "sequence_length": 128,
      "tokens": 128,
      "cases": [
        {
          "method": "uniform",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "cache_storage_bytes": 100,
          "compression_ratio": 12.8,
          "output_mse": 10,
          "output_cosine": 0.8
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "cache_storage_bytes": 200,
          "compression_ratio": 7.1,
          "output_mse": 5,
          "output_cosine": 0.9
        }
      ]
    },
    {
      "model": "tiny",
      "name": "s1",
      "prompt_index": 0,
      "layer": 1,
      "token_index": 511,
      "token_position": "last",
      "sequence_length": 512,
      "tokens": 512,
      "cases": [
        {
          "method": "uniform",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "cache_storage_bytes": 120,
          "compression_ratio": 12.8,
          "output_mse": 20,
          "output_cosine": 0.7
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "cache_storage_bytes": 220,
          "compression_ratio": 7.1,
          "output_mse": 8,
          "output_cosine": 0.85
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}

	if err := runCLI([]string{"--input", inputPath, "--out", outputPath}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got summaryReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(summary): %v", err)
	}
	if len(got.OverallParetoFrontier) != 1 {
		t.Fatalf("overall_pareto_frontier = %d want 1", len(got.OverallParetoFrontier))
	}
	if got.OverallParetoFrontier[0].Method != "turboquant" {
		t.Fatalf("overall_pareto_frontier[0].method = %q want %q", got.OverallParetoFrontier[0].Method, "turboquant")
	}
	if got.ByMethod["uniform"].BestByMSE.Method != "uniform" {
		t.Fatalf("by_method[uniform].best_by_mse.method = %q want %q", got.ByMethod["uniform"].BestByMSE.Method, "uniform")
	}
	layer1 := got.ByLayer["1"]
	if layer1.Samples != 2 {
		t.Fatalf("by_layer[1].samples = %d want 2", layer1.Samples)
	}
	if layer1.BestByMSE.KeyBits != 4 || layer1.BestByMSE.ValueBits != 4 {
		t.Fatalf("by_layer[1].best_by_mse = (%d,%d) want (4,4)", layer1.BestByMSE.KeyBits, layer1.BestByMSE.ValueBits)
	}
	token127 := got.ByTokenIndex["127"]
	if len(token127.Tokens) != 1 || token127.Tokens[0] != 128 {
		t.Fatalf("by_token_index[127].tokens = %v want [128]", token127.Tokens)
	}
	last := got.ByRelativePosition["last"]
	if last.Samples != 2 {
		t.Fatalf("by_relative_position[last].samples = %d want 2", last.Samples)
	}
	if last.BestByMSE.KeyBits != 4 || last.BestByMSE.ValueBits != 4 {
		t.Fatalf("by_relative_position[last].best_by_mse = (%d,%d) want (4,4)", last.BestByMSE.KeyBits, last.BestByMSE.ValueBits)
	}
}
