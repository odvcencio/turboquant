package turboquant

import (
	"math"
	"testing"
	"time"
)

func TestCodebookMSEMatchesPaper(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	dim := 384
	// Paper bounds exist only for bit widths 1-4; bit widths 5-8 gate on a
	// looser bound since the reference table does not cover them, but the
	// prefix-sum solver (Phase 3) makes exercising them cheap enough to add
	// as a monotonicity/sanity check.
	expected := map[int]float64{
		1: 0.36,
		2: 0.117,
		3: 0.03,
		4: 0.009,
	}
	var prevMSE float64 = math.Inf(1)
	for bw := 1; bw <= 8; bw++ {
		cb := computeCodebook(dim, bw)
		got := cb.expectedMSE(dim)
		if want, ok := expected[bw]; ok {
			ratio := got / want
			if ratio < 0.8 || ratio > 1.2 {
				t.Errorf("bw=%d: MSE=%.4f, want ≈ %.4f (ratio %.2f)", bw, got, want, ratio)
			}
		}
		if got > prevMSE {
			t.Errorf("bw=%d: MSE=%.6f rose above bw=%d's MSE=%.6f, want monotonically decreasing", bw, got, bw-1, prevMSE)
		}
		if got < 0 {
			t.Errorf("bw=%d: MSE=%.6f is negative", bw, got)
		}
		prevMSE = got
	}
}

// TestCodebookSolverMatchesReference compares the prefix-sum-quadrature
// solver against the original per-interval Simpson solver for dimensions
// 200, 384, and 1536 at bit widths 1-6, gating the Phase 3 rewrite.
func TestCodebookSolverMatchesReference(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping slow reference comparison under race detector")
	}
	if testing.Short() {
		t.Skip("skipping slow reference comparison in -short mode")
	}
	const tolerance = 1e-6
	for _, dim := range []int{200, 384, 1536} {
		for bw := 1; bw <= 6; bw++ {
			ref := computeCodebookReference(dim, bw)
			got := computeCodebook(dim, bw)
			if len(ref.centroids) != len(got.centroids) {
				t.Fatalf("dim=%d bw=%d: centroid count %d != reference %d", dim, bw, len(got.centroids), len(ref.centroids))
			}
			for i := range ref.centroids {
				diff := math.Abs(float64(ref.centroids[i] - got.centroids[i]))
				if diff > tolerance {
					t.Errorf("dim=%d bw=%d: centroid[%d] = %v, reference = %v, diff %.3e exceeds %.0e",
						dim, bw, i, got.centroids[i], ref.centroids[i], diff, tolerance)
				}
			}
			for i := range ref.boundaries {
				diff := math.Abs(float64(ref.boundaries[i] - got.boundaries[i]))
				if diff > tolerance {
					t.Errorf("dim=%d bw=%d: boundary[%d] = %v, reference = %v, diff %.3e exceeds %.0e",
						dim, bw, i, got.boundaries[i], ref.boundaries[i], diff, tolerance)
				}
			}
		}
	}
}

// TestCodebookTableMatchesSolver compares every embedded table entry
// against a fresh solver run.
func TestCodebookTableMatchesSolver(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping slow comparison under race detector")
	}
	const tolerance = 1e-6
	for _, dim := range TabulatedDims() {
		for bw := 1; bw <= 8; bw++ {
			tabled, ok := codebookFromTable(dim, bw)
			if !ok {
				t.Fatalf("dim=%d bw=%d: expected a table entry", dim, bw)
			}
			solved := computeCodebook(dim, bw)
			if len(tabled.centroids) != len(solved.centroids) {
				t.Fatalf("dim=%d bw=%d: centroid count %d != solver %d", dim, bw, len(tabled.centroids), len(solved.centroids))
			}
			for i := range tabled.centroids {
				diff := math.Abs(float64(tabled.centroids[i] - solved.centroids[i]))
				if diff > tolerance {
					t.Errorf("dim=%d bw=%d: table centroid[%d] = %v, solver = %v, diff %.3e exceeds %.0e",
						dim, bw, i, tabled.centroids[i], solved.centroids[i], diff, tolerance)
				}
			}
			for i := range tabled.boundaries {
				diff := math.Abs(float64(tabled.boundaries[i] - solved.boundaries[i]))
				if diff > tolerance {
					t.Errorf("dim=%d bw=%d: table boundary[%d] = %v, solver = %v, diff %.3e exceeds %.0e",
						dim, bw, i, tabled.boundaries[i], solved.boundaries[i], diff, tolerance)
				}
			}
		}
	}
}

// TestCodebookTableRoundTrip parses the embedded blob and asserts the entry
// count, the centroid count, and the boundary count.
func TestCodebookTableRoundTrip(t *testing.T) {
	table, err := parseCodebookTable(codebookTableBytes)
	if err != nil {
		t.Fatalf("parseCodebookTable: %v", err)
	}
	wantDims := 13
	wantBitWidths := 8
	if len(table) != wantDims*wantBitWidths {
		t.Fatalf("table entry count = %d, want %d", len(table), wantDims*wantBitWidths)
	}
	for key, cb := range table {
		wantLevels := 1 << uint(key.bitWidth)
		if len(cb.centroids) != wantLevels {
			t.Errorf("dim=%d bw=%d: centroid count %d want %d", key.dim, key.bitWidth, len(cb.centroids), wantLevels)
		}
		if len(cb.boundaries) != wantLevels-1 {
			t.Errorf("dim=%d bw=%d: boundary count %d want %d", key.dim, key.bitWidth, len(cb.boundaries), wantLevels-1)
		}
	}
}

// TestCodebookSourceReportsTable asserts "table" for a curated dimension and
// "computed" for one outside the table.
func TestCodebookSourceReportsTable(t *testing.T) {
	if got := CodebookSource(1536, 8); got != "table" {
		t.Errorf("CodebookSource(1536, 8) = %q, want %q", got, "table")
	}
	if got := CodebookSource(1537, 8); got != "computed" {
		t.Errorf("CodebookSource(1537, 8) = %q, want %q", got, "computed")
	}
}

// TestTabulatedDims asserts the curated dimension list, in ascending order.
func TestTabulatedDims(t *testing.T) {
	want := []int{64, 128, 200, 256, 384, 512, 768, 1024, 1152, 1536, 2048, 3072, 4096}
	got := TabulatedDims()
	if len(got) != len(want) {
		t.Fatalf("TabulatedDims() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("TabulatedDims()[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestQuantizerOutputUnchangedByTable quantizes vectors with a table-loaded
// codebook and with a solver-built codebook and requires identical packed
// bytes for a tabulated dimension.
func TestQuantizerOutputUnchangedByTable(t *testing.T) {
	dim := 1536
	bitWidth := 4
	seed := int64(7)

	tabled, ok := codebookFromTable(dim, bitWidth)
	if !ok {
		t.Fatalf("expected dim=%d bitWidth=%d to be tabulated", dim, bitWidth)
	}
	solved := computeCodebook(dim, bitWidth)

	rng := newTestRNG()
	vecs := make([][]float32, 256)
	for i := range vecs {
		vecs[i] = randomUnitVector(dim, rng)
	}

	qTable := newQuantizerWithRotation(dim, bitWidth, seed, newHadamardRotation(dim, newTestRNG()), tabled)
	qSolved := newQuantizerWithRotation(dim, bitWidth, seed, newHadamardRotation(dim, newTestRNG()), solved)

	for i, v := range vecs {
		pTable, _ := qTable.Quantize(v)
		pSolved, _ := qSolved.Quantize(v)
		if len(pTable) != len(pSolved) {
			t.Fatalf("vector %d: packed length mismatch", i)
		}
		for j := range pTable {
			if pTable[j] != pSolved[j] {
				t.Fatalf("vector %d: packed byte %d differs: table=%d solver=%d", i, j, pTable[j], pSolved[j])
			}
		}
	}
}

// TestPrecomputeCodebookWarmsCache asserts that PrecomputeCodebook populates
// the runtime cache and rejects invalid parameters like the other
// validated constructors.
func TestPrecomputeCodebookWarmsCache(t *testing.T) {
	if err := PrecomputeCodebook(1537, 3); err != nil {
		t.Fatalf("PrecomputeCodebook: %v", err)
	}
	key := codebookKey{dim: 1537, bitWidth: 3}
	codebookCacheMu.RLock()
	_, ok := codebookCache[key]
	codebookCacheMu.RUnlock()
	if !ok {
		t.Fatal("PrecomputeCodebook did not warm the cache")
	}

	if err := PrecomputeCodebook(1, 3); err == nil {
		t.Fatal("expected error for invalid dim")
	}
	if err := PrecomputeCodebook(64, 0); err == nil {
		t.Fatal("expected error for invalid bitWidth")
	}
}

func TestNewQuantizer1536_8bitTableIsFast(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping timing assertion under race detector")
	}
	NewWithSeed(1536, 8, 1) // warm the codebook cache
	start := time.Now()
	NewWithSeed(1536, 8, 2)
	elapsed := time.Since(start)
	if elapsed > time.Millisecond {
		t.Errorf("NewWithSeed(1536, 8, ...) took %s from a warm table cache, want < 1ms", elapsed)
	}
}

func TestCodebookCentroidsAreSorted(t *testing.T) {
	for bw := 1; bw <= 4; bw++ {
		cb := computeCodebook(384, bw)
		for i := 1; i < len(cb.centroids); i++ {
			if cb.centroids[i] <= cb.centroids[i-1] {
				t.Errorf("bw=%d: centroids not sorted at %d: %.4f <= %.4f",
					bw, i, cb.centroids[i], cb.centroids[i-1])
			}
		}
	}
}

func TestCodebookCentroidCount(t *testing.T) {
	for bw := 1; bw <= 4; bw++ {
		cb := computeCodebook(384, bw)
		want := 1 << uint(bw)
		if len(cb.centroids) != want {
			t.Errorf("bw=%d: got %d centroids want %d", bw, len(cb.centroids), want)
		}
		if len(cb.boundaries) != want-1 {
			t.Errorf("bw=%d: got %d boundaries want %d", bw, len(cb.boundaries), want-1)
		}
	}
}

func TestNearestCentroidBinarySearch(t *testing.T) {
	cb := computeCodebook(384, 2)
	for i, c := range cb.centroids {
		got := cb.nearestCentroid(c)
		if got != i {
			t.Errorf("nearestCentroid(%.4f) = %d, want %d", c, got, i)
		}
	}
}

func TestCodebookSymmetry(t *testing.T) {
	// Beta distribution is symmetric around 0, so centroids should be symmetric
	cb := computeCodebook(384, 2)
	n := len(cb.centroids)
	for i := 0; i < n/2; i++ {
		sum := float64(cb.centroids[i]) + float64(cb.centroids[n-1-i])
		if math.Abs(sum) > 0.01 {
			t.Errorf("centroids[%d]+centroids[%d] = %.4f, want ≈ 0", i, n-1-i, sum)
		}
	}
}
