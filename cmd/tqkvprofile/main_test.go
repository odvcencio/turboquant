package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type bundledProfileReportForTest struct {
	Reports []profileReport `json:"reports"`
}

func TestRunCLIBestMSE(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "report.json")
	outputPath := filepath.Join(dir, "profile.json")
	input := `{
  "generated_at": "2026-04-07T00:00:00Z",
  "samples": [
    {
      "layer": 0,
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 10,
          "output_cosine": 0.7
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 150,
          "compression_ratio": 13.33,
          "output_mse": 5,
          "output_cosine": 0.9
        }
      ]
    },
    {
      "layer": 1,
      "tokens": 256,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 80,
          "compression_ratio": 50,
          "output_mse": 20,
          "output_cosine": 0.65
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 150,
          "compression_ratio": 26.66,
          "output_mse": 6,
          "output_cosine": 0.92
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
	var got profileReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if got.SelectionMode != "best-mse" {
		t.Fatalf("SelectionMode = %q want %q", got.SelectionMode, "best-mse")
	}
	if len(got.Layers) != 2 || len(got.Profiles) != 2 {
		t.Fatalf("lens = (%d,%d) want (2,2)", len(got.Layers), len(got.Profiles))
	}
	if got.Layers[0].KeyBits != 4 || got.Layers[0].ValueBits != 4 {
		t.Fatalf("layer0 bits = (%d,%d) want (4,4)", got.Layers[0].KeyBits, got.Layers[0].ValueBits)
	}
	if got.Layers[1].KeyBits != 4 || got.Layers[1].ValueBits != 4 {
		t.Fatalf("layer1 bits = (%d,%d) want (4,4)", got.Layers[1].KeyBits, got.Layers[1].ValueBits)
	}
	if got.Profiles[0].Capacity != 128 || got.Profiles[1].Capacity != 256 {
		t.Fatalf("profile capacities = (%d,%d) want (128,256)", got.Profiles[0].Capacity, got.Profiles[1].Capacity)
	}
	if got.Summary.AggregateCompression <= 0 {
		t.Fatalf("AggregateCompression = %v want > 0", got.Summary.AggregateCompression)
	}
}

func TestRunCLIPreservesGroupedKVHeadsInProfiles(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "report.json")
	outputPath := filepath.Join(dir, "profile.json")
	input := `{
  "generated_at": "2026-04-07T00:00:00Z",
  "samples": [
    {
      "layer": 9,
      "tokens": 128,
      "heads": 4,
      "kv_heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 3,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 10,
          "output_cosine": 0.7
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
	var got profileReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if len(got.Layers) != 1 || len(got.Profiles) != 1 {
		t.Fatalf("lens = (%d,%d) want (1,1)", len(got.Layers), len(got.Profiles))
	}
	if got.Layers[0].KVHeads != 2 {
		t.Fatalf("Layers[0].KVHeads = %d want 2", got.Layers[0].KVHeads)
	}
	if got.Profiles[0].KVHeads != 2 {
		t.Fatalf("Profiles[0].KVHeads = %d want 2", got.Profiles[0].KVHeads)
	}
}

func TestRunCLIBudgetedAllocation(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "report.json")
	outputPath := filepath.Join(dir, "profile.json")
	input := `{
  "generated_at": "2026-04-07T00:00:00Z",
  "samples": [
    {
      "layer": 0,
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 10,
          "output_cosine": 0.7
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 150,
          "compression_ratio": 13.33,
          "output_mse": 5,
          "output_cosine": 0.9
        }
      ]
    },
    {
      "layer": 1,
      "tokens": 256,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 80,
          "compression_ratio": 50,
          "output_mse": 20,
          "output_cosine": 0.65
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 150,
          "compression_ratio": 26.66,
          "output_mse": 6,
          "output_cosine": 0.92
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}

	if err := runCLI([]string{
		"--input", inputPath,
		"--out", outputPath,
		"--max-mean-storage-bytes", "250",
		"--capacity", "512",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got profileReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if got.SelectionMode != "budgeted" {
		t.Fatalf("SelectionMode = %q want %q", got.SelectionMode, "budgeted")
	}
	if len(got.Layers) != 2 {
		t.Fatalf("len(layers) = %d want 2", len(got.Layers))
	}
	if got.Layers[0].KeyBits != 2 || got.Layers[0].ValueBits != 2 {
		t.Fatalf("budget layer0 bits = (%d,%d) want (2,2)", got.Layers[0].KeyBits, got.Layers[0].ValueBits)
	}
	if got.Layers[1].KeyBits != 4 || got.Layers[1].ValueBits != 4 {
		t.Fatalf("budget layer1 bits = (%d,%d) want (4,4)", got.Layers[1].KeyBits, got.Layers[1].ValueBits)
	}
	if got.Summary.TotalMeanStorageBytes > 250 {
		t.Fatalf("TotalMeanStorageBytes = %v want <= 250", got.Summary.TotalMeanStorageBytes)
	}
	if got.Profiles[0].Capacity != 512 || got.Profiles[1].Capacity != 512 {
		t.Fatalf("profile capacities = (%d,%d) want (512,512)", got.Profiles[0].Capacity, got.Profiles[1].Capacity)
	}
}

func TestRunCLIFidelityFloorMinMeanCosine(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "report.json")
	outputPath := filepath.Join(dir, "profile.json")
	input := `{
  "generated_at": "2026-04-07T00:00:00Z",
  "samples": [
    {
      "layer": 0,
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 10,
          "output_cosine": 0.70
        },
        {
          "method": "turboquant",
          "key_bits": 3,
          "value_bits": 3,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 130,
          "compression_ratio": 15.38,
          "output_mse": 7,
          "output_cosine": 0.82
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 160,
          "compression_ratio": 12.5,
          "output_mse": 5,
          "output_cosine": 0.90
        }
      ]
    },
    {
      "layer": 1,
      "tokens": 256,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 80,
          "compression_ratio": 50,
          "output_mse": 20,
          "output_cosine": 0.65
        },
        {
          "method": "turboquant",
          "key_bits": 3,
          "value_bits": 3,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 120,
          "compression_ratio": 33.33,
          "output_mse": 10,
          "output_cosine": 0.84
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 150,
          "compression_ratio": 26.66,
          "output_mse": 6,
          "output_cosine": 0.92
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}

	if err := runCLI([]string{
		"--input", inputPath,
		"--out", outputPath,
		"--min-mean-cosine", "0.80",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got profileReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if got.SelectionMode != "fidelity-floor" {
		t.Fatalf("SelectionMode = %q want %q", got.SelectionMode, "fidelity-floor")
	}
	if got.Layers[0].KeyBits != 3 || got.Layers[0].ValueBits != 3 {
		t.Fatalf("layer0 bits = (%d,%d) want (3,3)", got.Layers[0].KeyBits, got.Layers[0].ValueBits)
	}
	if got.Layers[1].KeyBits != 3 || got.Layers[1].ValueBits != 3 {
		t.Fatalf("layer1 bits = (%d,%d) want (3,3)", got.Layers[1].KeyBits, got.Layers[1].ValueBits)
	}
	if got.MinMeanCosine != 0.80 {
		t.Fatalf("MinMeanCosine = %v want 0.80", got.MinMeanCosine)
	}
}

func TestRunCLIFidelityFloorMinP05CosineBudgeted(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "report.json")
	outputPath := filepath.Join(dir, "profile.json")
	input := `{
  "generated_at": "2026-04-07T00:00:00Z",
  "samples": [
    {
      "layer": 0,
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 10,
          "output_cosine": 0.55
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 150,
          "compression_ratio": 13.33,
          "output_mse": 5,
          "output_cosine": 0.80
        }
      ]
    },
    {
      "layer": 0,
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 11,
          "output_cosine": 0.56
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 150,
          "compression_ratio": 13.33,
          "output_mse": 6,
          "output_cosine": 0.81
        }
      ]
    },
    {
      "layer": 1,
      "tokens": 256,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 3,
          "value_bits": 3,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 120,
          "compression_ratio": 33.33,
          "output_mse": 9,
          "output_cosine": 0.69
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 150,
          "compression_ratio": 26.66,
          "output_mse": 6,
          "output_cosine": 0.75
        }
      ]
    },
    {
      "layer": 1,
      "tokens": 256,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 3,
          "value_bits": 3,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 120,
          "compression_ratio": 33.33,
          "output_mse": 10,
          "output_cosine": 0.71
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 4000,
          "cache_storage_bytes": 150,
          "compression_ratio": 26.66,
          "output_mse": 7,
          "output_cosine": 0.77
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}

	if err := runCLI([]string{
		"--input", inputPath,
		"--out", outputPath,
		"--min-p05-cosine", "0.74",
		"--max-mean-storage-bytes", "310",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got profileReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if got.SelectionMode != "budgeted-fidelity-floor" {
		t.Fatalf("SelectionMode = %q want %q", got.SelectionMode, "budgeted-fidelity-floor")
	}
	if got.Layers[0].KeyBits != 4 || got.Layers[0].ValueBits != 4 {
		t.Fatalf("layer0 bits = (%d,%d) want (4,4)", got.Layers[0].KeyBits, got.Layers[0].ValueBits)
	}
	if got.Layers[1].KeyBits != 4 || got.Layers[1].ValueBits != 4 {
		t.Fatalf("layer1 bits = (%d,%d) want (4,4)", got.Layers[1].KeyBits, got.Layers[1].ValueBits)
	}
	if got.Layers[0].P05OutputCosine < 0.74 || got.Layers[1].P05OutputCosine < 0.74 {
		t.Fatalf("P05OutputCosine = (%v,%v) want >= 0.74", got.Layers[0].P05OutputCosine, got.Layers[1].P05OutputCosine)
	}
}

func TestRunCLITokenPositionFilter(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "report.json")
	outputPath := filepath.Join(dir, "profile.json")
	input := `{
  "generated_at": "2026-04-07T00:00:00Z",
  "samples": [
    {
      "layer": 0,
      "token_position": "25%",
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 2,
          "output_cosine": 0.95
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 150,
          "compression_ratio": 13.33,
          "output_mse": 1,
          "output_cosine": 0.96
        }
      ]
    },
    {
      "layer": 0,
      "token_position": "last",
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 20,
          "output_cosine": 0.55
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 150,
          "compression_ratio": 13.33,
          "output_mse": 5,
          "output_cosine": 0.80
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}

	if err := runCLI([]string{
		"--input", inputPath,
		"--out", outputPath,
		"--token-position", "last",
		"--min-mean-cosine", "0.75",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got profileReport
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(report): %v", err)
	}
	if len(got.TokenPositions) != 1 || got.TokenPositions[0] != "last" {
		t.Fatalf("TokenPositions = %v want [last]", got.TokenPositions)
	}
	if got.Layers[0].KeyBits != 4 || got.Layers[0].ValueBits != 4 {
		t.Fatalf("layer0 bits = (%d,%d) want (4,4)", got.Layers[0].KeyBits, got.Layers[0].ValueBits)
	}
	if got.Layers[0].Samples != 1 {
		t.Fatalf("layer0 samples = %d want 1", got.Layers[0].Samples)
	}
}

func TestRunCLIProfileSetBundle(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "report.json")
	outputPath := filepath.Join(dir, "bundle.json")
	input := `{
  "generated_at": "2026-04-07T00:00:00Z",
  "samples": [
    {
      "layer": 0,
      "token_position": "25%",
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 2,
          "output_cosine": 0.95
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 150,
          "compression_ratio": 13.33,
          "output_mse": 1,
          "output_cosine": 0.96
        }
      ]
    },
    {
      "layer": 0,
      "token_position": "last",
      "tokens": 128,
      "heads": 2,
      "head_dim": 16,
      "cases": [
        {
          "method": "turboquant",
          "key_bits": 2,
          "value_bits": 2,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 100,
          "compression_ratio": 20,
          "output_mse": 20,
          "output_cosine": 0.55
        },
        {
          "method": "turboquant",
          "key_bits": 4,
          "value_bits": 4,
          "requested_top_k": 8,
          "top_k": 8,
          "raw_kv_bytes": 2000,
          "cache_storage_bytes": 150,
          "compression_ratio": 13.33,
          "output_mse": 5,
          "output_cosine": 0.80
        }
      ]
    }
  ]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}

	if err := runCLI([]string{
		"--input", inputPath,
		"--out", outputPath,
		"--profile-set", "all,last",
		"--min-mean-cosine", "0.75",
	}, os.Stdout, os.Stderr); err != nil {
		t.Fatalf("runCLI: %v", err)
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("ReadFile(output): %v", err)
	}
	var got bundledProfileReportForTest
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("json.Unmarshal(bundle): %v", err)
	}
	if len(got.Reports) != 2 {
		t.Fatalf("len(reports) = %d want 2", len(got.Reports))
	}
	if got.Reports[0].ProfileSet != "all" || got.Reports[1].ProfileSet != "last" {
		t.Fatalf("profile sets = (%q,%q) want (%q,%q)", got.Reports[0].ProfileSet, got.Reports[1].ProfileSet, "all", "last")
	}
	if got.Reports[0].Layers[0].KeyBits != 2 || got.Reports[0].Layers[0].ValueBits != 2 {
		t.Fatalf("all layer0 bits = (%d,%d) want (2,2)", got.Reports[0].Layers[0].KeyBits, got.Reports[0].Layers[0].ValueBits)
	}
	if got.Reports[1].Layers[0].KeyBits != 4 || got.Reports[1].Layers[0].ValueBits != 4 {
		t.Fatalf("last layer0 bits = (%d,%d) want (4,4)", got.Reports[1].Layers[0].KeyBits, got.Reports[1].Layers[0].ValueBits)
	}
}
