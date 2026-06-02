package mesh

import (
	"math"
	"testing"
)

func normalize3(v []float32) {
	for i := 0; i+2 < len(v); i += 3 {
		x, y, z := float64(v[i]), float64(v[i+1]), float64(v[i+2])
		l := math.Sqrt(x*x + y*y + z*z)
		if l == 0 {
			continue
		}
		v[i] = float32(x / l)
		v[i+1] = float32(y / l)
		v[i+2] = float32(z / l)
	}
}

func TestNormalsOctRoundTripAngularError(t *testing.T) {
	c := float32(0.57735027) // 1/sqrt(3)
	in := []float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		-1, 0, 0,
		0, 0, -1,
		c, c, c,
	}
	normalize3(in) // defensive: ensure unit inputs

	const bits = 16
	packed := EncodeNormals(in, bits)
	got := DecodeNormals(packed, bits, len(in)/3)

	if len(got) != len(in) {
		t.Fatalf("len(got)=%d, want %d", len(got), len(in))
	}
	for i := 0; i+2 < len(in); i += 3 {
		dot := float64(in[i])*float64(got[i]) +
			float64(in[i+1])*float64(got[i+1]) +
			float64(in[i+2])*float64(got[i+2])
		if dot > 1 {
			dot = 1
		}
		if dot < -1 {
			dot = -1
		}
		deg := math.Acos(dot) * 180 / math.Pi
		if deg >= 1.0 {
			t.Errorf("normal %d: angular error %v deg >= 1.0 (in=%v got=%v)",
				i/3, deg, in[i:i+3], got[i:i+3])
		}
	}
}

func TestSnormRoundTrip(t *testing.T) {
	for _, bits := range []int{8, 16} {
		maxS := int32(1)<<(bits-1) - 1
		// Sweep every representable signed code.
		for s := -maxS - 1; s <= maxS; s++ {
			u := uint32(s) & ((1 << bits) - 1)
			o := snormDecode(u, bits)
			// Re-encode the decoded value and confirm it maps back to the
			// same code (clamped representable range round-trips).
			re := snormEncode(o, bits)
			// snormEncode clamps to [-1,1] then to maxS, so the most-negative
			// code (-maxS-1) collapses onto -maxS on re-encode.
			want := u
			if s == -maxS-1 {
				want = uint32(-maxS) & ((1 << bits) - 1)
			}
			if re != want {
				t.Errorf("bits=%d s=%d: decode=%v re-encode=%#x want %#x",
					bits, s, o, re, want)
			}
		}
		// Most-negative code decodes to approximately -1.
		mostNeg := uint32(-maxS-1) & ((1 << bits) - 1)
		v := snormDecode(mostNeg, bits)
		if math.Abs(float64(v)+1) > 1.0/float64(maxS)+1e-6 {
			t.Errorf("bits=%d most-negative code decoded to %v, want ~ -1", bits, v)
		}
	}
}
