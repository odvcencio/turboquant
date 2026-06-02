package mesh

import (
	"math"
	"testing"
)

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
