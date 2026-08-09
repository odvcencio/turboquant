package mesh

import (
	"math"
	"strings"
	"testing"
)

// mustPanic runs fn and reports whether it panicked, returning the recovered
// value formatted as a string (empty when fn did not panic).
func mustPanic(t *testing.T, fn func()) (panicked bool, msg string) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			panicked = true
			if s, ok := r.(string); ok {
				msg = s
			} else if err, ok := r.(error); ok {
				msg = err.Error()
			}
		}
	}()
	fn()
	return false, ""
}

func TestDecodePanicsOnShortBuffer(t *testing.T) {
	const n = 4 // claim 4 vertices but supply far too few bytes
	p := PositionParams{Bits: 16, Min: [3]float32{}, Scale: [3]float32{1, 1, 1}}
	short := []byte{0, 0} // 2 bytes, need 3*4*2 = 24

	t.Run("positions", func(t *testing.T) {
		panicked, msg := mustPanic(t, func() { DecodePositions(short, p, n) })
		if !panicked {
			t.Fatal("DecodePositions did not panic on short buffer")
		}
		if !strings.Contains(msg, "positions buffer too short") {
			t.Errorf("panic message %q does not name the positions invariant", msg)
		}
	})
	t.Run("normals", func(t *testing.T) {
		panicked, msg := mustPanic(t, func() { DecodeNormals(short, 16, n) })
		if !panicked {
			t.Fatal("DecodeNormals did not panic on short buffer")
		}
		if !strings.Contains(msg, "normals buffer too short") {
			t.Errorf("panic message %q does not name the normals invariant", msg)
		}
	})
	t.Run("weights", func(t *testing.T) {
		panicked, msg := mustPanic(t, func() { DecodeWeights(short, n) })
		if !panicked {
			t.Fatal("DecodeWeights did not panic on short buffer")
		}
		if !strings.Contains(msg, "weights buffer too short") {
			t.Errorf("panic message %q does not name the weights invariant", msg)
		}
	})
	t.Run("joints", func(t *testing.T) {
		panicked, msg := mustPanic(t, func() { DecodeJoints(short, 8, n) })
		if !panicked {
			t.Fatal("DecodeJoints did not panic on short buffer")
		}
		if !strings.Contains(msg, "joints buffer too short") {
			t.Errorf("panic message %q does not name the joints invariant", msg)
		}
	})
}

func TestEncodePositionsRejectsBadBits(t *testing.T) {
	pos := []float32{0, 0, 0, 1, 1, 1}
	t.Run("encode", func(t *testing.T) {
		panicked, msg := mustPanic(t, func() { EncodePositions(pos, 24) })
		if !panicked {
			t.Fatal("EncodePositions did not panic on bits=24")
		}
		if !strings.Contains(msg, "bits must be 8 or 16") {
			t.Errorf("panic message %q does not explain the bits constraint", msg)
		}
	})
	t.Run("decode", func(t *testing.T) {
		p := PositionParams{Bits: 24, Scale: [3]float32{1, 1, 1}}
		panicked, msg := mustPanic(t, func() { DecodePositions([]byte{0, 0, 0}, p, 1) })
		if !panicked {
			t.Fatal("DecodePositions did not panic on bits=24")
		}
		if !strings.Contains(msg, "bits must be 8 or 16") {
			t.Errorf("panic message %q does not explain the bits constraint", msg)
		}
	})
}

func TestPositionsRoundTripWithinHalfStep(t *testing.T) {
	pos := []float32{0, 0, 0, 1, 2, 3, -5, 0.5, 10}
	const bits = 16
	packed, p := EncodePositions(pos, bits)
	got := DecodePositions(packed, p, 3)

	if len(got) != len(pos) {
		t.Fatalf("len(got)=%d, want %d", len(got), len(pos))
	}
	for i := range pos {
		axis := i % 3
		tol := p.Scale[axis]/2 + 1e-6
		if diff := float32(math.Abs(float64(got[i] - pos[i]))); diff > tol {
			t.Errorf("component %d (axis %d): got %v want %v, diff %v > tol %v",
				i, axis, got[i], pos[i], diff, tol)
		}
	}
}

func TestPositionsNonFiniteProducesFiniteSpec(t *testing.T) {
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	// Non-finite components appear in the first vertex (which seeds the AABB)
	// as well as later vertices, across all three axes.
	pos := []float32{
		nan, inf, 0, // first vertex poisons every axis if unguarded
		1, 2, 3,
		-5, nan, 10,
		inf, 0.5, nan,
	}
	const bits = 16
	packed, p := EncodePositions(pos, bits)

	for a := 0; a < 3; a++ {
		if math.IsNaN(float64(p.Min[a])) || math.IsInf(float64(p.Min[a]), 0) {
			t.Errorf("axis %d: Min %v is not finite", a, p.Min[a])
		}
		if math.IsNaN(float64(p.Scale[a])) || math.IsInf(float64(p.Scale[a]), 0) {
			t.Errorf("axis %d: Scale %v is not finite", a, p.Scale[a])
		}
	}

	got := DecodePositions(packed, p, len(pos)/3)
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("decoded component %d = %v is not finite", i, v)
		}
	}
}

func TestPositionsDegenerateAxisExact(t *testing.T) {
	// All X equal to 4; Y and Z vary. The degenerate X axis must decode
	// back to exactly 4.
	pos := []float32{4, 0, 0, 4, 1, 2, 4, -3, 7}
	const bits = 16
	packed, p := EncodePositions(pos, bits)
	got := DecodePositions(packed, p, 3)

	for v := 0; v < 3; v++ {
		x := got[v*3+0]
		if x != 4 {
			t.Errorf("vertex %d X: got %v, want exactly 4", v, x)
		}
	}
}
