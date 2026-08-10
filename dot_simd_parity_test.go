//go:build goexperiment.simd && amd64

package turboquant

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// tolerance mirrors dot_test.go's TestDotFloat32sMatchesScalar bound: 1e-4
// relative, scaled by the reference magnitude, to absorb float32 summation
// reordering between the scalar, SSE, and simd accumulation orders.
func closeEnough(t *testing.T, tag string, got, want float32) {
	t.Helper()
	diff := math.Abs(float64(got - want))
	limit := 1e-4 * math.Max(1, math.Abs(float64(want)))
	if diff > limit {
		t.Fatalf("%s: got %.8f want %.8f diff %.8f (limit %.8f)", tag, got, want, diff, limit)
	}
}

var parityLengths = []int{0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65,
	127, 128, 129, 255, 256, 257, 383, 384, 385, 511, 512, 1535, 1536, 1537}

// TestDotFloat32sSIMDParity checks that the simd kernel (dispatched as
// DotFloat32s and directly as dotFloat32sSIMD) matches both the plain scalar
// reference and the hand-written SSE asm across a range of lengths including
// non-multiple-of-8 tails.
func TestDotFloat32sSIMDParity(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, n := range parityLengths {
		a := make([]float32, n)
		b := make([]float32, n)
		for i := 0; i < n; i++ {
			a[i] = float32(rng.NormFloat64())
			b[i] = float32(rng.NormFloat64())
		}
		want := dotFloat32sScalar(a, b)

		gotDispatch := DotFloat32s(a, b)
		closeEnough(t, "n=?/dispatch", gotDispatch, want)

		var gotSIMD, gotSSE float32
		if n > 0 {
			gotSIMD = dotFloat32sSIMD(unsafe.SliceData(a), unsafe.SliceData(b), n)
			gotSSE = dotFloat32sSSE(unsafe.SliceData(a), unsafe.SliceData(b), n)
		}
		closeEnough(t, "n=?/simd-direct", gotSIMD, want)
		closeEnough(t, "n=?/sse-direct", gotSSE, want)
	}
}

func rowsScalarRef(dst []float32, rows, vec []float32, numRows, n int) {
	for r := 0; r < numRows; r++ {
		dst[r] = dotFloat32sScalar(rows[r*n:(r+1)*n], vec)
	}
}

// TestDotFloat32Rows4SIMDParity checks dotFloat32Rows4 (dispatched to the
// simd kernel under this build) against a per-row scalar reference.
func TestDotFloat32Rows4SIMDParity(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range parityLengths {
		rows := make([]float32, 4*n)
		vec := make([]float32, n)
		for i := range rows {
			rows[i] = float32(rng.NormFloat64())
		}
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		want := make([]float32, 4)
		rowsScalarRef(want, rows, vec, 4, n)

		var got [4]float32
		dotFloat32Rows4(&got, rows, vec)
		for r := 0; r < 4; r++ {
			closeEnough(t, "rows4", got[r], want[r])
		}

		if n > 0 {
			var gotSIMD [4]float32
			dotFloat32Rows4SIMD(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &gotSIMD[0])
			for r := 0; r < 4; r++ {
				closeEnough(t, "rows4-simd-direct", gotSIMD[r], want[r])
			}
			var gotSSE [4]float32
			dotFloat32Rows4SSE(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &gotSSE[0])
			for r := 0; r < 4; r++ {
				closeEnough(t, "rows4-sse-direct", gotSSE[r], want[r])
			}
		}
	}
}

// TestDotFloat32Rows8SIMDParity checks dotFloat32Rows8 (dispatched to the
// simd kernel under this build) against a per-row scalar reference.
func TestDotFloat32Rows8SIMDParity(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, n := range parityLengths {
		rows := make([]float32, 8*n)
		vec := make([]float32, n)
		for i := range rows {
			rows[i] = float32(rng.NormFloat64())
		}
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		want := make([]float32, 8)
		rowsScalarRef(want, rows, vec, 8, n)

		var got [8]float32
		dotFloat32Rows8(&got, rows, vec)
		for r := 0; r < 8; r++ {
			closeEnough(t, "rows8", got[r], want[r])
		}

		if n > 0 {
			var gotSIMD [8]float32
			dotFloat32Rows8SIMD(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &gotSIMD[0])
			for r := 0; r < 8; r++ {
				closeEnough(t, "rows8-simd-direct", gotSIMD[r], want[r])
			}
			var gotSSE [8]float32
			dotFloat32Rows8SSE(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &gotSSE[0])
			for r := 0; r < 8; r++ {
				closeEnough(t, "rows8-sse-direct", gotSSE[r], want[r])
			}
		}
	}
}

// TestDotFloat32Rows8BlockedSIMDParity checks the blocked kernel using the
// SAME buffer layout that blockProjectionRowGroup8 produces (unchanged by
// this spike), matching how qjlProjectBlocked / PrepareQuery drive it.
// Only multiple-of-8 lengths are exercised: blockProjectionRowGroup8 itself
// is only well-defined for dim % 4 == 0 (see dot_rows_simd_amd64.go's
// doc comment), and its sole caller, blockProjectionRows8, refuses anything
// that isn't a multiple of 8, so that's the kernel's real domain.
func TestDotFloat32Rows8BlockedSIMDParity(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	for _, n := range []int{8, 16, 32, 64, 128, 256, 384, 1536} {
		rows := make([]float32, 8*n)
		vec := make([]float32, n)
		for i := range rows {
			rows[i] = float32(rng.NormFloat64())
		}
		for i := range vec {
			vec[i] = float32(rng.NormFloat64())
		}
		want := make([]float32, 8)
		rowsScalarRef(want, rows, vec, 8, n)

		blocked := make([]float32, len(rows))
		blockProjectionRowGroup8(blocked, rows, n)

		var got [8]float32
		dotFloat32Rows8Blocked(&got, blocked, vec)
		for r := 0; r < 8; r++ {
			closeEnough(t, "rows8blocked", got[r], want[r])
		}

		var gotSIMD [8]float32
		dotFloat32Rows8BlockedSIMD(unsafe.SliceData(blocked), unsafe.SliceData(vec), n, &gotSIMD[0])
		for r := 0; r < 8; r++ {
			closeEnough(t, "rows8blocked-simd-direct", gotSIMD[r], want[r])
		}
		var gotSSE [8]float32
		dotFloat32Rows8BlockedSSE(unsafe.SliceData(blocked), unsafe.SliceData(vec), n, &gotSSE[0])
		for r := 0; r < 8; r++ {
			closeEnough(t, "rows8blocked-sse-direct", gotSSE[r], want[r])
		}
	}
}

// TestFWHTSIMDParity checks the vectorized fwhtNormalizedInPlace against the
// pre-spike scalar reference. Only power-of-two lengths are exercised: the
// transform's own butterfly indexing requires it (see fwhtScalarRef and
// rotation_fwht_generic.go's doc comment), independent of SIMD.
func TestFWHTSIMDParity(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	for _, n := range []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 1536, 2048} {
		// 384 and 1536 are not themselves powers of two but decompose into
		// power-of-two hadamardBlocks chunks (256+128, 1024+512); exercise
		// those chunk sizes directly since that's what callers ever pass.
		sizes := []int{n}
		if n == 384 {
			sizes = []int{256, 128}
		}
		if n == 1536 {
			sizes = []int{1024, 512}
		}
		for _, size := range sizes {
			values := make([]float32, size)
			for i := range values {
				values[i] = float32(rng.NormFloat64())
			}
			want := make([]float32, size)
			copy(want, values)
			fwhtScalarRef(want)

			got := make([]float32, size)
			copy(got, values)
			fwhtNormalizedInPlace(got)

			for i := range got {
				closeEnough(t, "fwht", got[i], want[i])
			}
		}
	}
}

// TestSignedProjectedSum8SIMDParity checks the branchless masked-multiply
// variant against the branchy scalar reference over random sign bytes and
// dot values, including the all-zero and all-one sign patterns.
func TestSignedProjectedSum8SIMDParity(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	cases := []byte{0x00, 0xFF, 0x01, 0x80, 0xAA, 0x55}
	for i := 0; i < 250; i++ {
		cases = append(cases, byte(rng.Intn(256)))
	}
	for _, signs := range cases {
		var dots [8]float32
		for i := range dots {
			dots[i] = float32(rng.NormFloat64())
		}
		want := signedProjectedSum8ScalarRef(signs, dots)
		got := signedProjectedSum8(signs, dots)
		closeEnough(t, "signedProjectedSum8", got, want)
	}
}
