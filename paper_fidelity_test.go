package turboquant

import (
	"math"
	"math/rand"
	"testing"
)

// paperMSETable holds the exact Lloyd-Max optimum MSE for a Gaussian
// coordinate at b = 1..4 (1 - 2/pi, and the classic Max 1960 values). The
// paper's Theorem 1 table (0.36 / 0.117 / 0.03 / 0.009) rounds these; the
// b=3 entry rounds 0.03454 down to 0.03, so tests must compare against the
// exact optimum, not the rounded print.
var paperMSETable = map[int]float64{
	1: 0.3634, 2: 0.1175, 3: 0.03455, 4: 0.0095,
}

// paperIPTable holds the paper's Theorem 2 fine-grained inner-product
// distortion values (times dimension) for b = 1..4 with unit x and y.
var paperIPTable = map[int]float64{
	1: 1.57, 2: 0.56, 3: 0.18, 4: 0.047,
}

func empiricalMSE(t *testing.T, q *Quantizer, dim, trials int, seed int64) float64 {
	t.Helper()
	rng := rand.New(rand.NewSource(seed))
	var total float64
	for range trials {
		x := randomUnitVector(dim, rng)
		packed, norm := q.Quantize(x)
		recon := q.Dequantize(packed)
		var mse float64
		for i := range x {
			d := float64(x[i]) - float64(norm)*float64(recon[i])
			mse += d * d
		}
		total += mse
	}
	return total / float64(trials)
}

// TestMSEMatchesPaperTableNonPow2 validates the paper MSE table for the
// default Hadamard rotation at a non-power-of-two dimension (200, one of the
// paper's own experimental dimensions) and at 384. The multi-round rotation
// must hold the table even though both dimensions split into multiple
// power-of-two FWHT blocks.
func TestMSEMatchesPaperTableNonPow2(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	for _, dim := range []int{200, 384} {
		for bw, bound := range paperMSETable {
			q := NewHadamardWithSeed(dim, bw, 77)
			avg := empiricalMSE(t, q, dim, 500, int64(100*dim+bw))
			if avg > bound*1.10 {
				t.Errorf("dim=%d bw=%d: avg MSE %.5f exceeds the Lloyd-Max optimum %.5f by >10%%", dim, bw, avg, bound)
			}
			// The information-theoretic lower bound (Theorem 3): no quantizer
			// can average below 1/4^b on the uniform sphere source. Beating it
			// would mean the measurement (not the quantizer) is broken.
			lower := math.Pow(0.25, float64(bw))
			if avg < lower*0.92 {
				t.Errorf("dim=%d bw=%d: avg MSE %.5f beats the information-theoretic lower bound %.5f", dim, bw, avg, lower)
			}
		}
	}
}

// TestMSEHighBitWidthsWithinPanterDite validates b = 5..8 against the
// paper's asymptotic bound (sqrt(3)*pi/2)/4^b, both for the codebook's
// theoretical expectation and empirically.
func TestMSEHighBitWidthsWithinPanterDite(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	const dim = 384
	panter := math.Sqrt(3) * math.Pi / 2
	for bw := 5; bw <= 8; bw++ {
		bound := panter * math.Pow(0.25, float64(bw))
		cb := cachedCodebook(dim, bw)
		if theory := cb.expectedMSE(dim); theory > bound {
			t.Errorf("bw=%d: codebook expected MSE %.3e exceeds Panter-Dite bound %.3e", bw, theory, bound)
		}
		q := NewHadamardWithSeed(dim, bw, 5)
		avg := empiricalMSE(t, q, dim, 200, int64(bw))
		if avg > bound*1.25 {
			t.Errorf("bw=%d: empirical MSE %.3e exceeds Panter-Dite bound %.3e by >25%%", bw, avg, bound)
		}
	}
}

// adversarialInputs returns structured worst-case-style inputs that a
// single-round block-split rotation mishandles: two-hot vectors, energy
// concentrated on a contiguous prefix, and an alternating-sign vector that
// resembles an unpermuted Hadamard row.
func adversarialInputs(dim int) map[string][]float32 {
	twoHot := make([]float32, dim)
	twoHot[3] = float32(1 / math.Sqrt2)
	twoHot[dim-2] = float32(1 / math.Sqrt2)

	prefix := make([]float32, dim)
	span := dim / 2
	for i := range span {
		prefix[i] = float32(1 / math.Sqrt(float64(span)))
	}

	alternating := make([]float32, dim)
	inv := float32(1 / math.Sqrt(float64(dim)))
	for i := range alternating {
		if i%2 == 0 {
			alternating[i] = inv
		} else {
			alternating[i] = -inv
		}
	}
	return map[string][]float32{
		"two-hot":     twoHot,
		"prefix-half": prefix,
		"alternating": alternating,
	}
}

// TestMSEAdversarialInputsMeetPaperTable fixes the input and averages over
// quantizer seeds, which is the paper's actual guarantee (worst-case x,
// expectation over the quantizer's randomness).
func TestMSEAdversarialInputsMeetPaperTable(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	const dim, bw, seeds = 384, 2, 150
	bound := paperMSETable[bw]
	for name, x := range adversarialInputs(dim) {
		var total float64
		for seed := range seeds {
			q := NewHadamardWithSeed(dim, bw, int64(3000+seed))
			packed, norm := q.Quantize(x)
			recon := q.Dequantize(packed)
			var mse float64
			for i := range x {
				d := float64(x[i]) - float64(norm)*float64(recon[i])
				mse += d * d
			}
			total += mse
		}
		avg := total / seeds
		if avg > bound*1.20 {
			t.Errorf("input %q: avg MSE %.5f over %d seeds exceeds the Lloyd-Max optimum %.5f by >20%%", name, avg, seeds, bound)
		}
	}
}

// TestIPDistortionMatchesPaperTable validates the paper's Theorem 2 table
// for all four tabulated bit widths, including the b=1 pure QJL entry, with
// non-unit inputs and queries (distortion scales by ||x||^2 * ||y||^2).
func TestIPDistortionMatchesPaperTable(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	const dim, trials = 256, 900
	const xScale, yScale = 2.0, 1.5
	scaleSq := float64(xScale * xScale * yScale * yScale)
	for bw, tableValue := range paperIPTable {
		q := NewIPHadamardWithSeed(dim, bw, int64(40+bw))
		rng := rand.New(rand.NewSource(int64(700 + bw)))
		var sumSq float64
		for range trials {
			x := scaledVector(randomUnitVector(dim, rng), xScale)
			y := scaledVector(randomUnitVector(dim, rng), yScale)
			var trueIP float64
			for i := range x {
				trueIP += float64(x[i]) * float64(y[i])
			}
			e := float64(q.InnerProduct(q.Quantize(x), y)) - trueIP
			sumSq += e * e
		}
		distortion := sumSq / trials
		want := tableValue / float64(dim) * scaleSq
		if distortion > want*1.30 {
			t.Errorf("bw=%d: IP distortion %.6f exceeds paper value %.6f by >30%%", bw, distortion, want)
		}
		lower := math.Pow(0.25, float64(bw)) / float64(dim) * scaleSq
		if distortion < lower*0.85 {
			t.Errorf("bw=%d: IP distortion %.6f beats the lower bound %.6f", bw, distortion, lower)
		}
	}
}

// TestSplitIPDistortionInterpolates checks that the paper's 2.5-bit
// outlier-split configuration lands strictly between the uniform 2-bit and
// 3-bit distortions.
func TestSplitIPDistortionInterpolates(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	const dim, trials = 128, 900
	spec := SplitSpec{Outliers: seqInts(32), OutlierBits: 4, RegularBits: 2}

	distortion := func(estimate func(x, y []float32) float64) float64 {
		rng := rand.New(rand.NewSource(99))
		var sumSq float64
		for range trials {
			x := randomUnitVector(dim, rng)
			y := randomUnitVector(dim, rng)
			var trueIP float64
			for i := range x {
				trueIP += float64(x[i]) * float64(y[i])
			}
			e := estimate(x, y) - trueIP
			sumSq += e * e
		}
		return sumSq / trials
	}

	q2 := NewIPHadamardWithSeed(dim, 2, 61)
	q3 := NewIPHadamardWithSeed(dim, 3, 61)
	qs := NewSplitIPWithSeed(dim, spec, 61)

	d2 := distortion(func(x, y []float32) float64 { return float64(q2.InnerProduct(q2.Quantize(x), y)) })
	d3 := distortion(func(x, y []float32) float64 { return float64(q3.InnerProduct(q3.Quantize(x), y)) })
	ds := distortion(func(x, y []float32) float64 { return float64(qs.InnerProduct(qs.Quantize(x), y)) })

	if !(ds < d2*0.97) {
		t.Errorf("2.5-bit split distortion %.6f not below uniform 2-bit %.6f", ds, d2)
	}
	if !(ds > d3*1.03) {
		t.Errorf("2.5-bit split distortion %.6f not above uniform 3-bit %.6f", ds, d3)
	}
}
