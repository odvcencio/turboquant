package turboquant_test

import (
	"os"
	"testing"

	"m31labs.dev/turboquant/internal/recall"
)

// TestRecallSynthetic runs on every CI job. It measures recall 1@10 for the
// 4-bit inner-product quantizer against exact search on a synthetic
// clustered dataset and fails when recall drops below the recorded floor.
func TestRecallSynthetic(t *testing.T) {
	d := recall.Synthetic(64, 4096, 256, 8, 42)
	gt := recall.ComputeGroundTruth(d, 10)
	cfg := recall.Config{
		Method:   "turboquant-ip",
		BitWidth: 4,
		Seed:     42,
		K:        []int{10},
	}
	report, err := recall.Run(d, gt, cfg)
	if err != nil {
		t.Fatalf("recall.Run: %v", err)
	}
	got := report.RecallAtK[10]
	if got <= 0.90 {
		t.Fatalf("recall 1@10 = %.4f, want > 0.90", got)
	}
}

// TestRecallDataDirSkipsWhenUnset documents the TQ_RECALL_DATA contract: a
// real dataset run is opt-in and skips cleanly when the variable is absent.
func TestRecallDataDirSkipsWhenUnset(t *testing.T) {
	dir := os.Getenv("TQ_RECALL_DATA")
	if dir == "" {
		t.Skip("TQ_RECALL_DATA not set, skipping real dataset recall run")
	}
	if _, err := recall.LoadDataset(dir, "glove-200"); err != nil {
		t.Fatalf("LoadDataset: %v", err)
	}
}
