package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"m31labs.dev/turboquant/internal/recall"
)

func TestRunSyntheticWritesReport(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "report.json")

	code := run([]string{
		"-dataset", "synthetic",
		"-method", "turboquant-ip",
		"-bits", "4",
		"-k", "1,10",
		"-corpus", "512",
		"-queries", "64",
		"-seed", "7",
		"-out", outPath,
	})
	if code != 0 {
		t.Fatalf("run() = %d want 0", code)
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	var reports []*recall.Report
	if err := json.Unmarshal(data, &reports); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if len(reports) != 1 {
		t.Fatalf("len(reports) = %d want 1", len(reports))
	}
	if reports[0].CorpusCount != 512 || reports[0].QueryCount != 64 {
		t.Fatalf("unexpected report shape: %+v", reports[0])
	}
}

func TestRunBaselineToleranceGate(t *testing.T) {
	dir := t.TempDir()
	baselinePath := filepath.Join(dir, "baseline.json")

	baseline := []*recall.Report{
		{
			Config:    recall.Config{Method: "turboquant-ip", BitWidth: 4},
			Dataset:   "synthetic",
			RecallAtK: map[int]float64{10: 0.99},
		},
	}
	data, err := json.Marshal(baseline)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if err := os.WriteFile(baselinePath, data, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	current := []*recall.Report{
		{
			Config:    recall.Config{Method: "turboquant-ip", BitWidth: 4},
			Dataset:   "synthetic",
			RecallAtK: map[int]float64{10: 0.50},
		},
	}
	if compareToBaseline(current, baseline, 0.005) {
		t.Fatal("expected a large recall regression to fail the gate")
	}

	current[0].RecallAtK[10] = 0.989
	if !compareToBaseline(current, baseline, 0.005) {
		t.Fatal("expected a small recall drop within tolerance to pass the gate")
	}
}
