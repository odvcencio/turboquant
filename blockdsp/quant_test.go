package blockdsp

import (
	"math/rand"
	"testing"
)

// TestQuantizeWorkedExample pins the tqwebp specification's own worked
// example (section 12.3): coefficients 63 and -30 at UV quantizer index 20
// (dcq=21, acq=24) quantize to levels +3 and -1, and dequantizing those
// levels reproduces 63 exactly and -24 (a 6-unit quantization loss on the
// second coefficient, matching the spec's own stated error).
func TestQuantizeWorkedExample(t *testing.T) {
	var coeff [16]int16
	coeff[0] = 63
	coeff[1] = -30

	const dcq, acq = 21, 24
	levels := QuantizeBlock(&coeff, dcq, acq)
	if levels[0] != 3 {
		t.Fatalf("level[0] = %d, want 3", levels[0])
	}
	if levels[1] != -1 {
		t.Fatalf("level[1] = %d, want -1", levels[1])
	}

	recon := DequantizeBlock(&levels, dcq, acq)
	if recon[0] != 63 {
		t.Fatalf("recon[0] = %d, want 63 (exact)", recon[0])
	}
	if recon[1] != -24 {
		t.Fatalf("recon[1] = %d, want -24 (6-unit loss from -30)", recon[1])
	}
}

// TestQuantizeDeadZoneBoundary pins the dead zone precisely at each tested
// factor: |coeff| < factor must quantize to level 0, and |coeff| == factor
// must quantize to level +-1 exactly, for both the DC and AC factor slots.
func TestQuantizeDeadZoneBoundary(t *testing.T) {
	factors := []int16{1, 4, 8, 21, 24, 100, 157, 284}
	for _, f := range factors {
		var coeff [16]int16
		coeff[0] = f - 1 // DC, just inside the dead zone
		if f == 1 {
			coeff[0] = 0
		}
		coeff[1] = -(f - 1) // AC, just inside, negative
		if f == 1 {
			coeff[1] = 0
		}
		coeff[2] = f // DC-equivalent boundary on an AC slot: exactly one step out
		coeff[3] = -f

		levels := QuantizeBlock(&coeff, f, f)
		if levels[0] != 0 {
			t.Fatalf("factor=%d: level[0]=%d for coeff %d, want 0 (inside dead zone)", f, levels[0], coeff[0])
		}
		if levels[1] != 0 {
			t.Fatalf("factor=%d: level[1]=%d for coeff %d, want 0 (inside dead zone)", f, levels[1], coeff[1])
		}
		if levels[2] != 1 {
			t.Fatalf("factor=%d: level[2]=%d for coeff %d, want 1 (exactly one step out)", f, levels[2], coeff[2])
		}
		if levels[3] != -1 {
			t.Fatalf("factor=%d: level[3]=%d for coeff %d, want -1 (exactly one step out)", f, levels[3], coeff[3])
		}
	}
}

// TestQuantizeDequantizeRoundTripBound proves and checks the standard
// uniform dead-zone quantizer bound: because level = coeff / factor
// truncates toward zero, the reconstruction error coeff - level*factor is
// exactly coeff's remainder modulo factor (sign-matched to coeff), whose
// magnitude is always strictly less than factor.
func TestQuantizeDequantizeRoundTripBound(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	for trial := 0; trial < 20000; trial++ {
		dcq := int16(1 + rng.Intn(300))
		acq := int16(1 + rng.Intn(300))
		var coeff [16]int16
		for i := range coeff {
			coeff[i] = int16(rng.Intn(4001) - 2000)
		}

		levels := QuantizeBlock(&coeff, dcq, acq)
		recon := DequantizeBlock(&levels, dcq, acq)

		factor := func(i int) int16 {
			if i == 0 {
				return dcq
			}
			return acq
		}
		for i := 0; i < 16; i++ {
			d := int(coeff[i]) - int(recon[i])
			if d < 0 {
				d = -d
			}
			if d >= int(factor(i)) {
				t.Fatalf("trial=%d pos=%d: |coeff-recon|=%d >= factor=%d (coeff=%d recon=%d)",
					trial, i, d, factor(i), coeff[i], recon[i])
			}
		}
	}
}

// TestQuantizeSignSymmetry checks that quantizing c and -c produces
// opposite-signed levels of equal magnitude, for every factor: the
// truncating-toward-zero division is odd-symmetric.
func TestQuantizeSignSymmetry(t *testing.T) {
	rng := rand.New(rand.NewSource(22))
	for trial := 0; trial < 5000; trial++ {
		dcq := int16(1 + rng.Intn(300))
		acq := int16(1 + rng.Intn(300))
		var pos, neg [16]int16
		for i := range pos {
			v := int16(rng.Intn(4001) - 2000)
			pos[i] = v
			neg[i] = -v
		}
		lp := QuantizeBlock(&pos, dcq, acq)
		ln := QuantizeBlock(&neg, dcq, acq)
		for i := 0; i < 16; i++ {
			if lp[i] != -ln[i] {
				t.Fatalf("trial=%d pos=%d: level(c)=%d, level(-c)=%d, want negation", trial, i, lp[i], ln[i])
			}
		}
	}
}

func TestQuantizePanicsOnNonPositiveFactor(t *testing.T) {
	var coeff [16]int16
	cases := []struct {
		name     string
		dcq, acq int16
	}{
		{"zero dc", 0, 10},
		{"zero ac", 10, 0},
		{"negative dc", -1, 10},
		{"negative ac", 10, -1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s: expected panic, got none", c.name)
				}
			}()
			QuantizeBlock(&coeff, c.dcq, c.acq)
		})
	}
}

func TestQuantizeAllZero(t *testing.T) {
	var coeff [16]int16
	levels := QuantizeBlock(&coeff, 21, 24)
	if levels != ([16]int16{}) {
		t.Fatalf("QuantizeBlock of all-zero coeff = %v, want all zero", levels)
	}
	recon := DequantizeBlock(&levels, 21, 24)
	if recon != ([16]int16{}) {
		t.Fatalf("DequantizeBlock of all-zero levels = %v, want all zero", recon)
	}
}
