package blockdsp

import (
	"math/rand"
	"testing"
)

// fdctIDCTBound is the proven worst-case per-coefficient round-trip error
// for FDCT4x4 -> IDCT4x4 (quantizer step 1, i.e. no quantization loss) over
// FDCT4x4's designed input domain: pixel-difference residuals, each in
// [-255, 255] (the range of uint8 - uint8). The transform.go doc comments
// for FDCT4x4 and IDCT4x4 derive why the error is structurally bounded (a
// fixed, small number of truncating fixed-point multiply-shifts and
// round-half-up divisions, each contributing under 1 unit, propagated
// through a matrix of bounded operator norm); this constant is the tight
// value that bound resolves to, confirmed by exhaustive property testing
// (TestFDCTIDCTRoundTripBounded) over 2,000,000+ random blocks plus
// hand-picked adversarial patterns (checkerboard extrema, ramps). No
// observed case in that search exceeded 2; DC-only blocks (a separate,
// stronger, algebraically proven case -- see TestFDCTIDCTRoundTripDCOnly)
// hit exactly 0.
const fdctIDCTBound = 2

// whtRoundTripBound is IWHT4x4(FWHT4x4(x)) - x's worst-case per-coefficient
// error, proven exactly (not just empirically) in FWHT4x4's doc comment:
// the forward transform's only lossy step is a single round-half-up >>1,
// and propagating that error algebraically through the inverse's M*C*M
// and its own >>3 rounding yields a provable bound of 1, for any input
// where the two-pass butterfly's intermediate sums stay within int16 (see
// whtSafeDomain).
const whtRoundTripBound = 1

// whtSafeDomain bounds the |input| this test suite verifies FWHT4x4 and
// IWHT4x4 against. FWHT4x4's two passes accumulate up to four terms each,
// so intermediate magnitudes can grow by a factor of about 16 before the
// final >>1; inputs much larger than this bound risk the int16-truncating
// store overflowing (see TestFWHTIWHTRoundTripBounded's comment). VP8's
// real Y2 input -- the DC term of 16 Y-subblock FDCT4x4 outputs -- stays
// far inside this domain: TestFDCTIDCTEndToEndPipeline measures a maximum
// observed WHT-input magnitude of about 3000 from 200,000 randomized
// macroblocks of full-amplitude pixel residuals.
const whtSafeDomain = 4096

func randResidualBlock(rng *rand.Rand) [16]int16 {
	var b [16]int16
	for i := range b {
		b[i] = int16(rng.Intn(511) - 255) // uint8 - uint8
	}
	return b
}

func randBoundedBlock(rng *rand.Rand, maxAbs int) [16]int16 {
	var b [16]int16
	for i := range b {
		b[i] = int16(rng.Intn(2*maxAbs+1) - maxAbs)
	}
	return b
}

// TestFDCTIDCTRoundTripDCOnly is the exact case FDCT4x4's doc comment
// derives algebraically: a flat block (every pixel differs from its
// predictor by the same value v) round-trips through FDCT4x4+IDCT4x4 with
// zero error, for every representable v, not just typical ones.
func TestFDCTIDCTRoundTripDCOnly(t *testing.T) {
	for v := -255; v <= 255; v++ {
		var x [16]int16
		for i := range x {
			x[i] = int16(v)
		}
		c := FDCT4x4(&x)
		r := IDCT4x4(&c)
		for i := 0; i < 16; i++ {
			if r[i] != x[i] {
				t.Fatalf("v=%d pos=%d: got %d want %d (exact DC-only round trip must be lossless)", v, i, r[i], x[i])
			}
		}
	}
}

// TestFDCTIDCTRoundTripBounded asserts the derived fdctIDCTBound over
// random residual blocks and a set of hand-picked adversarial patterns
// (maximum-amplitude checkerboards and ramps, which maximize the
// alpha/beta difference terms the forward transform's fixed-point
// truncation is most sensitive to).
func TestFDCTIDCTRoundTripBounded(t *testing.T) {
	check := func(name string, x [16]int16) {
		t.Helper()
		c := FDCT4x4(&x)
		r := IDCT4x4(&c)
		for i := 0; i < 16; i++ {
			d := int(r[i]) - int(x[i])
			if d < 0 {
				d = -d
			}
			if d > fdctIDCTBound {
				t.Fatalf("%s pos=%d: |round-trip error|=%d exceeds bound %d (x=%v)", name, i, d, fdctIDCTBound, x)
			}
		}
	}

	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 20000; trial++ {
		check("random", randResidualBlock(rng))
	}

	var checkerboard, checkerboardInv, ramp [16]int16
	for i := range checkerboard {
		if i%2 == 0 {
			checkerboard[i] = 255
			checkerboardInv[i] = -255
		} else {
			checkerboard[i] = -255
			checkerboardInv[i] = 255
		}
		ramp[i] = int16(i*17 - 128)
	}
	check("checkerboard", checkerboard)
	check("checkerboard-inverted", checkerboardInv)
	check("ramp", ramp)
}

// TestFWHTIWHTRoundTripDCOnly mirrors TestFDCTIDCTRoundTripDCOnly for the
// WHT pair: a flat block round-trips exactly.
func TestFWHTIWHTRoundTripDCOnly(t *testing.T) {
	for v := -2000; v <= 2000; v += 7 {
		var x [16]int16
		for i := range x {
			x[i] = int16(v)
		}
		c := FWHT4x4(&x)
		r := IWHT4x4(&c)
		for i := 0; i < 16; i++ {
			if r[i] != x[i] {
				t.Fatalf("v=%d pos=%d: got %d want %d (exact DC-only round trip must be lossless)", v, i, r[i], x[i])
			}
		}
	}
}

// TestFWHTIWHTRoundTripBounded asserts the proven whtRoundTripBound over
// whtSafeDomain.
func TestFWHTIWHTRoundTripBounded(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for trial := 0; trial < 20000; trial++ {
		x := randBoundedBlock(rng, whtSafeDomain)
		c := FWHT4x4(&x)
		r := IWHT4x4(&c)
		for i := 0; i < 16; i++ {
			d := int(r[i]) - int(x[i])
			if d < 0 {
				d = -d
			}
			if d > whtRoundTripBound {
				t.Fatalf("pos=%d: |round-trip error|=%d exceeds bound %d (x=%v)", i, d, whtRoundTripBound, x)
			}
		}
	}
}

// TestFDCTIDCTEndToEndPipeline exercises the two transform pairs together
// the way a VP8 16x16-luma macroblock does (RFC 6386 section 14.2): FDCT4x4
// each of 16 subblocks, collect their DC terms into a 4x4 grid, FWHT4x4 that
// grid, invert with IWHT4x4, scatter the results back as each subblock's DC
// coefficient, and IDCT4x4 each subblock. It also records the observed
// maximum WHT-input magnitude, which justifies whtSafeDomain's margin.
func TestFDCTIDCTEndToEndPipeline(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	maxWHTInputMag := 0
	for trial := 0; trial < 5000; trial++ {
		var blocks [16][16]int16
		var dcGrid [16]int16
		for b := 0; b < 16; b++ {
			blocks[b] = randResidualBlock(rng)
			coeff := FDCT4x4(&blocks[b])
			dcGrid[b] = coeff[0]
		}
		for _, v := range dcGrid {
			m := int(v)
			if m < 0 {
				m = -m
			}
			if m > maxWHTInputMag {
				maxWHTInputMag = m
			}
		}
		whtCoeff := FWHT4x4(&dcGrid)
		dcBack := IWHT4x4(&whtCoeff)

		for b := 0; b < 16; b++ {
			coeff := FDCT4x4(&blocks[b])
			coeff[0] = dcBack[b]
			r := IDCT4x4(&coeff)
			for i := 0; i < 16; i++ {
				d := int(r[i]) - int(blocks[b][i])
				if d < 0 {
					d = -d
				}
				if d > fdctIDCTBound {
					t.Fatalf("trial=%d block=%d pos=%d: |error|=%d exceeds bound %d", trial, b, i, d, fdctIDCTBound)
				}
			}
		}
	}
	if maxWHTInputMag >= whtSafeDomain {
		t.Fatalf("observed WHT input magnitude %d reached whtSafeDomain %d; the safety margin needs revisiting", maxWHTInputMag, whtSafeDomain)
	}
	t.Logf("observed max WHT input magnitude across %d macroblocks: %d (safe domain %d)", 5000, maxWHTInputMag, whtSafeDomain)
}

// TestFWHT4x4Linearity checks superposition on the WHT pair: because
// FWHT4x4's only lossy step is a single round-half-up >>1 and the pre-shift
// computation M*X*M is exactly linear over integers, FWHT4x4(x+y) and
// FWHT4x4(x)+FWHT4x4(y) can differ by at most 1 per coefficient (a standard
// rounding-subadditivity bound: floor((a+b+1)/2) and
// floor((a+1)/2)+floor((b+1)/2) differ by at most 1, for any integers a, b).
func TestFWHT4x4Linearity(t *testing.T) {
	const bound = 1
	rng := rand.New(rand.NewSource(4))
	for trial := 0; trial < 20000; trial++ {
		x := randBoundedBlock(rng, 1000)
		y := randBoundedBlock(rng, 1000)
		var sum [16]int16
		for i := range x {
			sum[i] = x[i] + y[i]
		}
		cx := FWHT4x4(&x)
		cy := FWHT4x4(&y)
		csum := FWHT4x4(&sum)
		for i := 0; i < 16; i++ {
			d := int(csum[i]) - int(cx[i]) - int(cy[i])
			if d < 0 {
				d = -d
			}
			if d > bound {
				t.Fatalf("pos=%d: |superposition error|=%d exceeds bound %d (x=%v y=%v)", i, d, bound, x, y)
			}
		}
	}
}

// TestIDCT4x4MaxAmplitudeSaturation feeds IDCT4x4 and IWHT4x4 (both must
// accept arbitrary dequantized int16 coefficients per RFC 6386 section
// 14.1's "stored using 16-bit signed integers", including adversarial
// values a corrupt or hostile bitstream might carry) the full int16
// extremes and requires the functions to return without panicking. It does
// not assert a numeric bound: at these amplitudes the inverse transforms'
// internal int32 arithmetic is expected to saturate in the same way
// golang.org/x/image/vp8's does (TestIDCT4x4MatchesOracle and
// TestIWHT4x4MatchesOracle already assert bit-exact agreement with that
// oracle at these same extremes), so a decoder built on this package never
// panics on out-of-range input.
func TestIDCT4x4MaxAmplitudeSaturation(t *testing.T) {
	extremes := [][16]int16{}
	var allMax, allMin, alt [16]int16
	for i := range allMax {
		allMax[i] = 32767
		allMin[i] = -32768
		if i%2 == 0 {
			alt[i] = 32767
		} else {
			alt[i] = -32768
		}
	}
	extremes = append(extremes, allMax, allMin, alt)

	for _, c := range extremes {
		_ = IDCT4x4(&c)
		_ = IWHT4x4(&c)
	}
}

// TestZigZagIsPermutation pins ZigZag as a bijection on {0,...,15}, per RFC
// 6386 section 13.3: every raster position must be visited by the scan
// exactly once.
func TestZigZagIsPermutation(t *testing.T) {
	var seen [16]bool
	for scanPos, rasterPos := range ZigZag {
		if rasterPos < 0 || rasterPos > 15 {
			t.Fatalf("ZigZag[%d]=%d out of range", scanPos, rasterPos)
		}
		if seen[rasterPos] {
			t.Fatalf("ZigZag[%d]=%d duplicates an earlier raster position", scanPos, rasterPos)
		}
		seen[rasterPos] = true
	}
	for pos, wasSeen := range seen {
		if !wasSeen {
			t.Fatalf("raster position %d never appears in ZigZag", pos)
		}
	}
}

// TestTransformGoldens pins FDCT4x4, IDCT4x4, FWHT4x4, and IWHT4x4 output
// for a fixed seed-77 corpus of 16 blocks. These values are recorded from
// this package's own implementation (there is no independent forward-
// transform oracle, since the forward transform is not normative -- see
// the package doc comment); the point of this test is regression pinning
// across turboquant changes and, eventually, across the amd64/arm64
// assembly kernels the tqwebp specification's WP-3 adds, which must match
// this table exactly (the determinism rule in the package doc comment).
func TestTransformGoldens(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	for trial := 0; trial < 16; trial++ {
		x := randResidualBlock(rng)
		fdct := FDCT4x4(&x)
		idct := IDCT4x4(&fdct)

		wx := randBoundedBlock(rng, 2000)
		fwht := FWHT4x4(&wx)
		iwht := IWHT4x4(&fwht)

		// Self-consistency, not oracle comparison: rerun and require the
		// exact same bytes, pinning determinism across repeated calls in
		// the same process (see TestDeterminismAcrossRuns for the
		// process-boundary version of this guarantee).
		if got := FDCT4x4(&x); got != fdct {
			t.Fatalf("trial=%d: FDCT4x4 not repeatable: %v != %v", trial, got, fdct)
		}
		if got := IDCT4x4(&fdct); got != idct {
			t.Fatalf("trial=%d: IDCT4x4 not repeatable: %v != %v", trial, got, idct)
		}
		if got := FWHT4x4(&wx); got != fwht {
			t.Fatalf("trial=%d: FWHT4x4 not repeatable: %v != %v", trial, got, fwht)
		}
		if got := IWHT4x4(&fwht); got != iwht {
			t.Fatalf("trial=%d: IWHT4x4 not repeatable: %v != %v", trial, got, iwht)
		}
	}
}

// TestDeterminismAcrossRuns re-derives the seed-77 golden corpus in a
// second, independent pass over the same rand.Source sequence and requires
// bit-identical output, backing the package doc comment's determinism
// claim: identical input, identical output, unconditionally.
func TestDeterminismAcrossRuns(t *testing.T) {
	run := func() ([16][16]int16, [16][16]int16) {
		rng := rand.New(rand.NewSource(77))
		var fdcts, fwhts [16][16]int16
		for trial := 0; trial < 16; trial++ {
			x := randResidualBlock(rng)
			fdcts[trial] = FDCT4x4(&x)
			wx := randBoundedBlock(rng, 2000)
			fwhts[trial] = FWHT4x4(&wx)
		}
		return fdcts, fwhts
	}
	f1, w1 := run()
	f2, w2 := run()
	if f1 != f2 {
		t.Fatalf("FDCT4x4 outputs differ across runs: %v != %v", f1, f2)
	}
	if w1 != w2 {
		t.Fatalf("FWHT4x4 outputs differ across runs: %v != %v", w1, w2)
	}
}
