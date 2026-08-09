package turboquant

import (
	"math"
	"math/rand"
	"testing"
)

func TestRotationOrthogonality(t *testing.T) {
	dims := []int{4, 16, 64, 384}
	if raceEnabled {
		dims = []int{4, 16, 64}
	}
	for _, dim := range dims {
		rng := rand.New(rand.NewSource(42))
		rot := generateRotation(dim, rng)
		for trial := 0; trial < 10; trial++ {
			x := randomUnitVector(dim, rng)
			y := make([]float32, dim)
			tmp := make([]float32, dim)
			rotate(y, x, rot, dim)
			rotateInverse(tmp, y, rot, dim)
			var errSq float64
			for i := range x {
				d := float64(tmp[i] - x[i])
				errSq += d * d
			}
			if errSq > 1e-4 {
				t.Errorf("dim=%d trial=%d: round-trip error %.6f", dim, trial, errSq)
			}
		}
	}
}

func TestRotationPreservesNorm(t *testing.T) {
	dim := 128
	rng := rand.New(rand.NewSource(99))
	rot := generateRotation(dim, rng)
	x := randomUnitVector(dim, rng)
	y := make([]float32, dim)
	rotate(y, x, rot, dim)
	var normSq float64
	for _, v := range y {
		normSq += float64(v) * float64(v)
	}
	if math.Abs(normSq-1.0) > 1e-4 {
		t.Errorf("rotated norm² = %.6f, want ≈ 1.0", normSq)
	}
}

func TestHadamardRotationOrthogonality(t *testing.T) {
	dims := []int{3, 5, 16, 127, 384}
	if raceEnabled {
		dims = []int{3, 5, 16, 127}
	}
	for _, dim := range dims {
		rng := rand.New(rand.NewSource(7))
		rot := newHadamardRotation(dim, rng)
		for trial := 0; trial < 10; trial++ {
			x := randomUnitVector(dim, rng)
			y := make([]float32, dim)
			tmp := make([]float32, dim)
			work := make([]float32, dim)
			rot.apply(y, x, work)
			rot.applyInverse(tmp, y, work)
			var errSq float64
			for i := range x {
				d := float64(tmp[i] - x[i])
				errSq += d * d
			}
			if errSq > 1e-4 {
				t.Errorf("dim=%d trial=%d: hadamard round-trip error %.6f", dim, trial, errSq)
			}
		}
	}
}

func TestHadamardRotationPreservesNorm(t *testing.T) {
	dim := 384
	rng := rand.New(rand.NewSource(11))
	rot := newHadamardRotation(dim, rng)
	x := randomUnitVector(dim, rng)
	y := make([]float32, dim)
	work := make([]float32, dim)
	rot.apply(y, x, work)
	var normSq float64
	for _, v := range y {
		normSq += float64(v) * float64(v)
	}
	if math.Abs(normSq-1.0) > 1e-4 {
		t.Errorf("hadamard rotated norm² = %.6f, want ≈ 1.0", normSq)
	}
}

// oneHotVector returns a unit vector with a single 1 at idx.
func oneHotVector(dim, idx int) []float32 {
	v := make([]float32, dim)
	v[idx] = 1
	return v
}

// TestRotationMixesAcrossBlocks is the cheapest direct probe of the
// block-isolation bug (G2): a one-hot input rotated with a single Hadamard
// round stays exactly zero over its entire complementary block (hundreds of
// coordinates for the dimensions tested here). The default multi-round
// rotation must leave at most a small handful of coordinates exactly zero
// across every trial combined.
//
// A handful of exact zeros is expected, not a bug: every surviving
// coordinate after one round shares the same magnitude (a Rademacher sum),
// so a later round's FWHT can occasionally cancel an even number of them
// exactly. That is a combinatorial coincidence with small, bounded
// probability, not the systematic block isolation this test guards against.
// See the "pathological case" discussion in the hardening spec's rotation
// design section for the same acknowledgement (residual probability ~1e-6
// for the worst decomposition, higher but still small for others).
func TestRotationMixesAcrossBlocks(t *testing.T) {
	dims := []int{200, 384, 768, 1536, 3072}
	if raceEnabled {
		dims = []int{200, 384}
	}
	const maxTotalZeros = 3
	for _, dim := range dims {
		rng := rand.New(rand.NewSource(1))
		rot := newHadamardRotationRounds(dim, DefaultHadamardRounds, rng)
		work := make([]float32, dim)
		y := make([]float32, dim)
		totalZeros := 0
		for trial := 0; trial < 32; trial++ {
			idx := rng.Intn(dim)
			rot.apply(y, oneHotVector(dim, idx), work)
			for _, v := range y {
				if v == 0 {
					totalZeros++
				}
			}
		}
		if totalZeros > maxTotalZeros {
			t.Fatalf("dim=%d: %d exact-zero coordinates across 32 one-hot trials, want <= %d", dim, totalZeros, maxTotalZeros)
		}
	}
}

// TestSingleRoundLeavesBlockZero documents the regression that
// TestRotationMixesAcrossBlocks guards against: with a single Hadamard
// round, a one-hot input stays exactly zero over its entire complementary
// block. This is the root cause of G2 (rotation_backend.go's block-isolation
// bug), preserved here as a live demonstration rather than a comment.
func TestSingleRoundLeavesBlockZero(t *testing.T) {
	dim := 1536
	rng := rand.New(rand.NewSource(1))
	rot := newHadamardRotation(dim, rng)
	work := make([]float32, dim)
	y := make([]float32, dim)
	rot.apply(y, oneHotVector(dim, rng.Intn(dim)), work)

	zeros := 0
	for _, v := range y {
		if v == 0 {
			zeros++
		}
	}
	// The block decomposition for 1536 is [1024, 512]; a one-hot input
	// zeroes out whichever block it does not land in, so at least the
	// smaller block (512) stays exactly zero.
	if zeros < 512 {
		t.Fatalf("single-round rotation left only %d zero coordinates, want >= 512 (demonstrating the block-isolation bug)", zeros)
	}
}

// TestRotationFlattensOneHot asserts that the default multi-round rotation
// bounds the maximum rotated coordinate for a one-hot input the same way it
// would for an isotropic vector, matching the paper's Theorem 1 worst-case
// distortion bound.
func TestRotationFlattensOneHot(t *testing.T) {
	dims := []int{200, 384, 768, 1536, 3072}
	if raceEnabled {
		dims = []int{200, 384}
	}
	for _, dim := range dims {
		rng := rand.New(rand.NewSource(2))
		rot := newHadamardRotationRounds(dim, DefaultHadamardRounds, rng)
		work := make([]float32, dim)
		y := make([]float32, dim)
		bound := float32(4.0 * math.Sqrt(math.Log(float64(dim))/float64(dim)))
		for trial := 0; trial < 32; trial++ {
			idx := rng.Intn(dim)
			rot.apply(y, oneHotVector(dim, idx), work)
			for i, v := range y {
				av := v
				if av < 0 {
					av = -av
				}
				if av > bound {
					t.Fatalf("dim=%d trial=%d (one-hot at %d): |rotated[%d]|=%.4f exceeds bound %.4f", dim, trial, idx, i, av, bound)
				}
			}
		}
	}
}

// TestRotationRoundsAreOrthogonal extends the round-trip and norm checks to
// round counts 1, 2, 3, and 4.
func TestRotationRoundsAreOrthogonal(t *testing.T) {
	dims := []int{3, 5, 16, 127, 384}
	if raceEnabled {
		dims = []int{3, 5, 16, 127}
	}
	for _, rounds := range []int{1, 2, 3, 4} {
		for _, dim := range dims {
			rng := rand.New(rand.NewSource(7))
			rot := newHadamardRotationRounds(dim, rounds, rng)
			for trial := 0; trial < 5; trial++ {
				x := randomUnitVector(dim, rng)
				y := make([]float32, dim)
				tmp := make([]float32, dim)
				work := make([]float32, dim)
				rot.apply(y, x, work)
				rot.applyInverse(tmp, y, work)

				var errSq float64
				for i := range x {
					d := float64(tmp[i] - x[i])
					errSq += d * d
				}
				if errSq > 1e-4 {
					t.Errorf("rounds=%d dim=%d trial=%d: round-trip error %.6f", rounds, dim, trial, errSq)
				}

				var normSq float64
				for _, v := range y {
					normSq += float64(v) * float64(v)
				}
				if math.Abs(normSq-1.0) > 1e-4 {
					t.Errorf("rounds=%d dim=%d trial=%d: rotated norm² = %.6f want ≈ 1.0", rounds, dim, trial, normSq)
				}
			}
		}
	}
}
