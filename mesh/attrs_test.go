package mesh

import (
	"math"
	"testing"
)

func TestWeightsRoundTripNormalized(t *testing.T) {
	w := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.7, 0.3, 0, 0,
	}
	n := len(w) / 4
	packed := EncodeWeights(w)
	got := DecodeWeights(packed, n)

	if len(got) != len(w) {
		t.Fatalf("len(got)=%d, want %d", len(got), len(w))
	}
	const tol = 1.0/255 + 1e-4
	for i := range w {
		if d := math.Abs(float64(got[i] - w[i])); d > tol {
			t.Errorf("component %d: got %v want %v, diff %v > tol %v", i, got[i], w[i], d, tol)
		}
	}
	for v := 0; v < n; v++ {
		var sum float64
		for k := 0; k < 4; k++ {
			sum += float64(got[v*4+k])
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Errorf("vertex %d weights sum to %v, want 1 within 1e-5", v, sum)
		}
	}
}

func TestJointsRoundTripExact(t *testing.T) {
	j := []uint32{
		0, 1, 2, 3,
		4, 250, 12, 9,
	}
	// u8 exact round-trip.
	packed8 := EncodeJoints(j, 8)
	got8 := DecodeJoints(packed8, 8, len(j)/4)
	if len(got8) != len(j) {
		t.Fatalf("u8 len(got)=%d, want %d", len(got8), len(j))
	}
	for i := range j {
		if got8[i] != j[i] {
			t.Errorf("u8 joint %d: got %d want %d", i, got8[i], j[i])
		}
	}

	// u16: include a value (65535) that exceeds u8 range.
	j16 := []uint32{0, 1, 65535, 3, 4, 250, 1000, 9}
	packed16 := EncodeJoints(j16, 16)
	got16 := DecodeJoints(packed16, 16, len(j16)/4)
	for i := range j16 {
		if got16[i] != j16[i] {
			t.Errorf("u16 joint %d: got %d want %d", i, got16[i], j16[i])
		}
	}
}

func TestJointsClampOutOfRange(t *testing.T) {
	// 300 exceeds the u8 max (255) and must clamp to 255.
	j := []uint32{300, 1, 2, 3}
	packed := EncodeJoints(j, 8)
	got := DecodeJoints(packed, 8, 1)
	if got[0] != 255 {
		t.Errorf("u8 clamp: joint 300 -> %d, want 255", got[0])
	}

	// 70000 exceeds the u16 max (65535) and must clamp to 65535.
	j16 := []uint32{70000, 1, 2, 3}
	packed16 := EncodeJoints(j16, 16)
	got16 := DecodeJoints(packed16, 16, 1)
	if got16[0] != 65535 {
		t.Errorf("u16 clamp: joint 70000 -> %d, want 65535", got16[0])
	}
}

func TestWeightsZeroGuard(t *testing.T) {
	// A vertex whose 4 weights sum to zero must not divide by zero and must
	// decode back to all zeros.
	w := []float32{0, 0, 0, 0}
	packed := EncodeWeights(w)
	got := DecodeWeights(packed, 1)
	for i := range got {
		if got[i] != 0 {
			t.Errorf("zero-weight component %d: got %v, want 0", i, got[i])
		}
	}
}
