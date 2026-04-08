package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRunCLISingleCaptureToFile(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "capture.json")
	outputPath := filepath.Join(dir, "report.json")

	input := `{
  "name": "layer0-step3",
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
		"--key-bits", "3",
		"--value-bits", "2",
		"--top-k", "2",
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
	if len(got.Results) != 1 {
		t.Fatalf("results = %d want 1", len(got.Results))
	}
	if got.Results[0].Name != "layer0-step3" {
		t.Fatalf("name = %q want %q", got.Results[0].Name, "layer0-step3")
	}
	if got.Results[0].TopK != 2 {
		t.Fatalf("top_k = %d want 2", got.Results[0].TopK)
	}
}

func TestRunCLISupportsWrappedSamples(t *testing.T) {
	dir := t.TempDir()
	inputPath := filepath.Join(dir, "captures.json")
	outputPath := filepath.Join(dir, "report.json")
	input := `{
  "model": "tiny",
  "samples": [
    {
      "name": "layer0-step1",
      "heads": 1,
      "head_dim": 2,
      "query": [1, 0],
      "keys": [1, 0],
      "values": [2, 0]
    },
    {
      "name": "layer1-step1",
      "heads": 1,
      "head_dim": 2,
      "query": [0, 1],
      "keys": [0, 1],
      "values": [0, 2]
    }
  ]
}`
	if err := os.WriteFile(inputPath, []byte(input), 0o644); err != nil {
		t.Fatalf("WriteFile(input): %v", err)
	}
	if err := runCLI([]string{
		"--input", inputPath,
		"--out", outputPath,
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
	if len(got.Results) != 2 {
		t.Fatalf("results = %d want 2", len(got.Results))
	}
}
