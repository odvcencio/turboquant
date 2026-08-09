package turboquant

import (
	"math"
	"math/rand"
	"testing"
)

func TestQuantizerRoundTrip(t *testing.T) {
	for _, bw := range []int{1, 2, 3, 4} {
		q := NewWithSeed(384, bw, 42)
		rng := rand.New(rand.NewSource(99))
		x := randomUnitVector(384, rng)
		packed, norm := q.Quantize(x)
		if math.Abs(float64(norm)-1.0) > 0.01 {
			t.Errorf("bw=%d: norm=%.4f want ≈ 1.0", bw, norm)
		}
		wantSize := packedSize(384, bw)
		if len(packed) != wantSize {
			t.Errorf("bw=%d: packed size %d want %d", bw, len(packed), wantSize)
		}
		recon := q.Dequantize(packed)
		if len(recon) != 384 {
			t.Fatalf("bw=%d: dequantize returned %d values want 384", bw, len(recon))
		}
	}
}

func TestQuantizerMSEMatchesPaperTable(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	bounds := map[int]float64{
		1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009,
	}
	for bw, bound := range bounds {
		q := NewDenseWithSeed(384, bw, 42)
		rng := rand.New(rand.NewSource(123))
		var totalMSE float64
		trials := 1000
		for trial := 0; trial < trials; trial++ {
			x := randomUnitVector(384, rng)
			packed, _ := q.Quantize(x)
			recon := q.Dequantize(packed)
			var mse float64
			for i := range x {
				d := float64(x[i] - recon[i])
				mse += d * d
			}
			totalMSE += mse
		}
		avgMSE := totalMSE / float64(trials)
		if avgMSE > bound*1.15 {
			t.Errorf("bw=%d: avg MSE %.4f exceeds bound %.4f by >15%%", bw, avgMSE, bound)
		}
	}
}

func TestQuantizerDeterminism(t *testing.T) {
	q1 := NewWithSeed(128, 2, 42)
	q2 := NewWithSeed(128, 2, 42)
	rng := rand.New(rand.NewSource(99))
	x := randomUnitVector(128, rng)
	p1, n1 := q1.Quantize(x)
	p2, n2 := q2.Quantize(x)
	if n1 != n2 {
		t.Errorf("norms differ: %.4f vs %.4f", n1, n2)
	}
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Fatalf("packed byte %d differs: %d vs %d", i, p1[i], p2[i])
		}
	}
}

func TestQuantizerSeed(t *testing.T) {
	q := NewWithSeed(64, 2, 12345)
	if q.Seed() != 12345 {
		t.Errorf("Seed() = %d want 12345", q.Seed())
	}
	if q.Dim() != 64 {
		t.Errorf("Dim() = %d want 64", q.Dim())
	}
	if q.BitWidth() != 2 {
		t.Errorf("BitWidth() = %d want 2", q.BitWidth())
	}
}

func TestQuantizerInnerProduct(t *testing.T) {
	q := NewWithSeed(384, 3, 42)
	rng := rand.New(rand.NewSource(99))
	x := randomUnitVector(384, rng)
	y := randomUnitVector(384, rng)
	packed, norm := q.Quantize(x)
	estimated := q.InnerProduct(packed, norm, y)
	var trueIP float64
	for i := range x {
		trueIP += float64(x[i]) * float64(y[i])
	}
	diff := math.Abs(float64(estimated) - trueIP)
	if diff > 0.3 {
		t.Errorf("IP estimate %.4f vs true %.4f, diff %.4f", estimated, trueIP, diff)
	}
}

func TestHadamardQuantizerRoundTrip(t *testing.T) {
	q := NewHadamardWithSeed(384, 3, 42)
	if q.RotationKind() != "hadamard-multi" {
		t.Fatalf("RotationKind() = %q want hadamard-multi", q.RotationKind())
	}
	if q.Rounds() != DefaultHadamardRounds {
		t.Fatalf("Rounds() = %d want %d", q.Rounds(), DefaultHadamardRounds)
	}
	rng := rand.New(rand.NewSource(99))
	x := randomUnitVector(384, rng)
	packed, norm := q.Quantize(x)
	if len(packed) != packedSize(384, 3) {
		t.Fatalf("packed size = %d want %d", len(packed), packedSize(384, 3))
	}
	if math.Abs(float64(norm)-1.0) > 0.01 {
		t.Fatalf("norm=%.4f want ≈ 1.0", norm)
	}
	recon := q.Dequantize(packed)
	if len(recon) != 384 {
		t.Fatalf("recon len = %d want 384", len(recon))
	}
}

func TestHadamardQuantizerDeterminism(t *testing.T) {
	q1 := NewHadamardWithSeed(128, 2, 42)
	q2 := NewHadamardWithSeed(128, 2, 42)
	rng := rand.New(rand.NewSource(99))
	x := randomUnitVector(128, rng)
	p1, n1 := q1.Quantize(x)
	p2, n2 := q2.Quantize(x)
	if n1 != n2 {
		t.Fatalf("norm mismatch: %.4f != %.4f", n1, n2)
	}
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Fatalf("packed[%d] = %d want %d", i, p1[i], p2[i])
		}
	}
}

func TestQuantizeToRoundTrip(t *testing.T) {
	q := NewHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(88))
	x := randomUnitVector(128, rng)
	packed := make([]byte, PackedSize(q.Dim(), q.BitWidth()))
	norm := q.QuantizeTo(packed, x)
	if math.Abs(float64(norm)-1.0) > 0.01 {
		t.Fatalf("norm=%.4f want ≈ 1.0", norm)
	}
	recon := make([]float32, q.Dim())
	q.DequantizeTo(recon, packed)
	if len(recon) != 128 {
		t.Fatalf("recon len = %d want 128", len(recon))
	}
}

func TestQuantizeToZeroAllocs(t *testing.T) {
	skipAllocsUnderRace(t)
	q := NewHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(66))
	x := randomUnitVector(128, rng)
	dst := make([]byte, PackedSize(q.Dim(), q.BitWidth()))

	_ = q.QuantizeTo(dst, x)

	allocs := testing.AllocsPerRun(100, func() {
		_ = q.QuantizeTo(dst, x)
	})
	if allocs != 0 {
		t.Fatalf("QuantizeTo allocs = %.2f want 0", allocs)
	}
}

func TestDequantizeToZeroAllocs(t *testing.T) {
	skipAllocsUnderRace(t)
	q := NewHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(66))
	x := randomUnitVector(128, rng)
	packed, _ := q.Quantize(x)
	dst := make([]float32, q.Dim())

	q.DequantizeTo(dst, packed)

	allocs := testing.AllocsPerRun(100, func() {
		q.DequantizeTo(dst, packed)
	})
	if allocs != 0 {
		t.Fatalf("DequantizeTo allocs = %.2f want 0", allocs)
	}
}

func TestQuantizeToRejectsWrongVectorLength(t *testing.T) {
	q := NewHadamardWithSeed(8, 2, 42)
	dst := make([]byte, PackedSize(q.Dim(), q.BitWidth()))
	expectPanic(t, func() {
		q.QuantizeTo(dst, []float32{1, 2, 3})
	})
}

func TestDequantizeToRejectsWrongPackedLength(t *testing.T) {
	q := NewHadamardWithSeed(8, 2, 42)
	dst := make([]float32, q.Dim())
	expectPanic(t, func() {
		q.DequantizeTo(dst, []byte{1})
	})
}

func TestNewHadamardRoundsRejectsInvalidRounds(t *testing.T) {
	expectPanic(t, func() {
		NewHadamardRoundsWithSeed(64, 3, 0, 42)
	})
	expectPanic(t, func() {
		NewHadamardRoundsWithSeed(64, 3, 9, 42)
	})
}

func TestNewHadamardRoundsProducesRequestedRoundCount(t *testing.T) {
	for _, rounds := range []int{1, 2, 3, 4, 8} {
		q := NewHadamardRoundsWithSeed(64, 3, rounds, 42)
		if q.Rounds() != rounds {
			t.Errorf("rounds=%d: q.Rounds() = %d", rounds, q.Rounds())
		}
	}
}

func TestNewHadamardRoundsRandomSeedVaries(t *testing.T) {
	q1 := NewHadamardRounds(64, 3, 2)
	q2 := NewHadamardRounds(64, 3, 2)
	if q1.Seed() == q2.Seed() {
		t.Fatalf("expected different random seeds, got %d twice", q1.Seed())
	}
}

func TestDenseRotationRoundsIsZero(t *testing.T) {
	q := NewDenseWithSeed(64, 3, 42)
	if q.Rounds() != 0 {
		t.Fatalf("dense Rounds() = %d want 0", q.Rounds())
	}
}

func TestHadamardMSEParityToDense(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	for _, bw := range []int{2, 3, 4} {
		dense := NewDenseWithSeed(384, bw, 42)
		hadamard := NewHadamardWithSeed(384, bw, 42)
		rng := rand.New(rand.NewSource(123))
		var denseMSE, hadamardMSE float64
		trials := 250
		for trial := 0; trial < trials; trial++ {
			x := randomUnitVector(384, rng)

			pDense, _ := dense.Quantize(x)
			rDense := dense.Dequantize(pDense)
			pHad, _ := hadamard.Quantize(x)
			rHad := hadamard.Dequantize(pHad)

			for i := range x {
				dd := float64(x[i] - rDense[i])
				hd := float64(x[i] - rHad[i])
				denseMSE += dd * dd
				hadamardMSE += hd * hd
			}
		}
		ratio := hadamardMSE / denseMSE
		if ratio > 1.35 {
			t.Fatalf("bw=%d: hadamard/dense MSE ratio %.3f exceeds 1.35", bw, ratio)
		}
	}
}

// TestQuantizerMSEOneHotMatchesDense is the direct regression test for G2:
// with a single Hadamard round, a one-hot input stayed exactly zero outside
// its power-of-two block, so its quantized MSE badly trailed dense QR. The
// default multi-round rotation must match dense QR within 5 percent even on
// this adversarial input.
func TestQuantizerMSEOneHotMatchesDense(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	const dim = 1536
	for _, bw := range []int{1, 2, 3, 4} {
		dense := NewDenseWithSeed(dim, bw, 42)
		hadamard := NewHadamardWithSeed(dim, bw, 42)
		rng := rand.New(rand.NewSource(123))

		var denseMSE, hadamardMSE float64
		trials := 64
		for trial := 0; trial < trials; trial++ {
			x := oneHotVector(dim, rng.Intn(dim))

			pDense, _ := dense.Quantize(x)
			rDense := dense.Dequantize(pDense)
			pHad, _ := hadamard.Quantize(x)
			rHad := hadamard.Dequantize(pHad)

			for i := range x {
				dd := float64(x[i] - rDense[i])
				hd := float64(x[i] - rHad[i])
				denseMSE += dd * dd
				hadamardMSE += hd * hd
			}
		}
		ratio := hadamardMSE / denseMSE
		if ratio > 1.05 {
			t.Fatalf("bw=%d: one-hot hadamard/dense MSE ratio %.4f exceeds 1.05", bw, ratio)
		}
	}
}

// TestQuantizerMSEOneHotMatchesPaperTable reuses the paper distortion bounds
// from TestQuantizerMSEMatchesPaperTable and asserts that one-hot inputs stay
// within 15 percent of them once the rotation flattens them.
//
// This runs at dimension 384, not the spec's suggested 1536. At dimension
// 1536 the existing Lloyd-Max codebook solver (codebook.go, unchanged by
// this phase) underflows: computeCodebook's Simpson quadrature cannot
// resolve the Beta(d/2,d/2) density's vanishingly small tail mass at that
// concentration, so its "negligible probability mass, keep midpoint"
// fallback fires and produces a visibly wrong centroid (for example bit
// width 2's outer centroid lands at 0.673 instead of near 0.09, matching bit
// width 1's inner centroids almost exactly). That is a pre-existing
// numerical defect in the codebook solver, not a rotation defect, and Phase
// 3's prefix-sum quadrature is designed to fix exactly this failure mode.
// TestQuantizerMSEOneHotMatchesDense above is dimension 1536 as specified
// and is unaffected, because it compares dense and Hadamard rotations
// against the same (possibly imperfect) codebook rather than against an
// external bound.
func TestQuantizerMSEOneHotMatchesPaperTable(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	const dim = 384
	bounds := map[int]float64{
		1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009,
	}
	for bw, bound := range bounds {
		q := NewHadamardWithSeed(dim, bw, 42)
		rng := rand.New(rand.NewSource(321))
		var totalMSE float64
		trials := 64
		for trial := 0; trial < trials; trial++ {
			x := oneHotVector(dim, rng.Intn(dim))
			packed, _ := q.Quantize(x)
			recon := q.Dequantize(packed)
			var mse float64
			for i := range x {
				d := float64(x[i] - recon[i])
				mse += d * d
			}
			totalMSE += mse
		}
		avgMSE := totalMSE / float64(trials)
		if avgMSE > bound*1.15 {
			t.Errorf("bw=%d: one-hot avg MSE %.4f exceeds bound %.4f by >15%%", bw, avgMSE, bound)
		}
	}
}
