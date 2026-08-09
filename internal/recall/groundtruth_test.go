package recall

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestRecall1AtKKnownCase(t *testing.T) {
	gt := &GroundTruth{
		K: 3,
		Neighbors: []uint32{
			10, 20, 30, // query 0: true NN = 10
			11, 21, 31, // query 1: true NN = 11
		},
	}
	approx := []uint32{
		5, 10, // query 0: contains 10 at position 1 -> hit
		99, 98, // query 1: no hit
	}
	got := Recall1AtK(gt, approx, 2)
	want := 0.5
	if got != want {
		t.Fatalf("Recall1AtK = %v want %v", got, want)
	}
}

func TestGroundTruthCacheRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gt.json")
	d := Synthetic(16, 200, 20, 4, 1)

	gt1, err := LoadOrComputeGroundTruth(path, d, 5)
	if err != nil {
		t.Fatalf("LoadOrComputeGroundTruth: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected cache file to exist: %v", err)
	}

	gt2, err := LoadOrComputeGroundTruth(path, d, 5)
	if err != nil {
		t.Fatalf("LoadOrComputeGroundTruth (cached): %v", err)
	}
	if !reflect.DeepEqual(gt1.Neighbors, gt2.Neighbors) {
		t.Fatalf("cached neighbours differ from freshly computed neighbours")
	}

	// A different dataset produces a different checksum, forcing a
	// recompute rather than returning the stale cache.
	d2 := Synthetic(16, 200, 20, 4, 2)
	gt3, err := LoadOrComputeGroundTruth(path, d2, 5)
	if err != nil {
		t.Fatalf("LoadOrComputeGroundTruth (checksum mismatch): %v", err)
	}
	if reflect.DeepEqual(gt3.Neighbors, gt1.Neighbors) {
		t.Fatalf("expected different neighbours after checksum mismatch")
	}
}

func TestComputeGroundTruthCosine(t *testing.T) {
	d := &Dataset{
		Dim:     2,
		Corpus:  []float32{1, 0, 0, 1, -1, 0},
		Queries: []float32{2, 0},
		Metric:  MetricCosine,
	}
	gt := ComputeGroundTruth(d, 3)
	if gt.K != 3 {
		t.Fatalf("K = %d want 3", gt.K)
	}
	if gt.Neighbors[0] != 0 {
		t.Fatalf("nearest neighbour = %d want 0", gt.Neighbors[0])
	}
}
