package blockdsp

import (
	"math/rand"
	"testing"
)

func randPlane(rng *rand.Rand, rows, stride int) []uint8 {
	p := make([]uint8, rows*stride)
	for i := range p {
		p[i] = uint8(rng.Intn(256))
	}
	return p
}

func sadReference(a []uint8, aStride int, b []uint8, bStride int, size int) int32 {
	var sum int32
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			d := int32(a[y*aStride+x]) - int32(b[y*bStride+x])
			if d < 0 {
				d = -d
			}
			sum += d
		}
	}
	return sum
}

func sseReference(a []uint8, aStride int, b []uint8, bStride int, size int) int32 {
	var sum int32
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			d := int32(a[y*aStride+x]) - int32(b[y*bStride+x])
			sum += d * d
		}
	}
	return sum
}

func TestSADMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(10))
	for trial := 0; trial < 500; trial++ {
		aStride := 4 + rng.Intn(20)
		bStride := 4 + rng.Intn(20)
		a := randPlane(rng, 4, aStride)
		b := randPlane(rng, 4, bStride)
		got := SAD4x4(a, aStride, b, bStride)
		want := sadReference(a, aStride, b, bStride, 4)
		if got != want {
			t.Fatalf("trial=%d: SAD4x4=%d want %d", trial, got, want)
		}
	}
	for trial := 0; trial < 500; trial++ {
		aStride := 16 + rng.Intn(20)
		bStride := 16 + rng.Intn(20)
		a := randPlane(rng, 16, aStride)
		b := randPlane(rng, 16, bStride)
		got := SAD16x16(a, aStride, b, bStride)
		want := sadReference(a, aStride, b, bStride, 16)
		if got != want {
			t.Fatalf("trial=%d: SAD16x16=%d want %d", trial, got, want)
		}
	}
}

func TestSSEMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for trial := 0; trial < 500; trial++ {
		aStride := 4 + rng.Intn(20)
		bStride := 4 + rng.Intn(20)
		a := randPlane(rng, 4, aStride)
		b := randPlane(rng, 4, bStride)
		got := SSE4x4(a, aStride, b, bStride)
		want := sseReference(a, aStride, b, bStride, 4)
		if got != want {
			t.Fatalf("trial=%d: SSE4x4=%d want %d", trial, got, want)
		}
	}
	for trial := 0; trial < 500; trial++ {
		aStride := 16 + rng.Intn(20)
		bStride := 16 + rng.Intn(20)
		a := randPlane(rng, 16, aStride)
		b := randPlane(rng, 16, bStride)
		got := SSE16x16(a, aStride, b, bStride)
		want := sseReference(a, aStride, b, bStride, 16)
		if got != want {
			t.Fatalf("trial=%d: SSE16x16=%d want %d", trial, got, want)
		}
	}
}

func TestSADZeroForIdenticalBlocks(t *testing.T) {
	rng := rand.New(rand.NewSource(12))
	a := randPlane(rng, 16, 16)
	if got := SAD16x16(a, 16, a, 16); got != 0 {
		t.Fatalf("SAD16x16 of a block against itself = %d, want 0", got)
	}
	if got := SSE16x16(a, 16, a, 16); got != 0 {
		t.Fatalf("SSE16x16 of a block against itself = %d, want 0", got)
	}
	if got := SATD4x4(a, 16, a, 16); got != 0 {
		t.Fatalf("SATD4x4 of a block against itself = %d, want 0", got)
	}
}

func TestSADMaxAmplitude(t *testing.T) {
	white := make([]uint8, 16*16)
	black := make([]uint8, 16*16)
	for i := range white {
		white[i] = 255
	}
	// black is already all zero.
	if got, want := SAD16x16(white, 16, black, 16), int32(256*255); got != want {
		t.Fatalf("SAD16x16(white, black) = %d, want %d", got, want)
	}
	if got, want := SSE16x16(white, 16, black, 16), int32(256*255*255); got != want {
		t.Fatalf("SSE16x16(white, black) = %d, want %d", got, want)
	}
}

// satdReference is a second, independently coded implementation of a 4x4
// Hadamard-transformed SAD: it multiplies the difference block by the
// explicit 4x4 Hadamard matrix M (see FWHT4x4's doc comment for M's
// derivation) on both sides using ordinary matrix multiplication, instead
// of SATD4x4's add/subtract butterfly. Agreement between the two pins the
// butterfly's algebraic identity to M*D*M, not just self-consistency.
func satdReference(a []uint8, aStride int, b []uint8, bStride int) int32 {
	m := [4][4]int32{
		{1, 1, 1, 1},
		{1, 1, -1, -1},
		{1, -1, -1, 1},
		{1, -1, 1, -1},
	}
	var d [4][4]int32
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			d[y][x] = int32(a[y*aStride+x]) - int32(b[y*bStride+x])
		}
	}
	// t = M * d
	var t [4][4]int32
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var s int32
			for k := 0; k < 4; k++ {
				s += m[i][k] * d[k][j]
			}
			t[i][j] = s
		}
	}
	// h = t * M (M symmetric, so this equals t * M^T too)
	var sum int32
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var s int32
			for k := 0; k < 4; k++ {
				s += t[i][k] * m[k][j]
			}
			if s < 0 {
				s = -s
			}
			sum += s
		}
	}
	return sum >> 1
}

func TestSATD4x4MatchesIndependentImplementation(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	for trial := 0; trial < 2000; trial++ {
		aStride := 4 + rng.Intn(20)
		bStride := 4 + rng.Intn(20)
		a := randPlane(rng, 4, aStride)
		b := randPlane(rng, 4, bStride)
		got := SATD4x4(a, aStride, b, bStride)
		want := satdReference(a, aStride, b, bStride)
		if got != want {
			t.Fatalf("trial=%d: SATD4x4=%d want %d", trial, got, want)
		}
	}
}

func TestMetricsPanicOnShortSlice(t *testing.T) {
	cases := []struct {
		name string
		fn   func()
	}{
		{"SAD4x4 short a", func() { SAD4x4(make([]uint8, 2), 4, make([]uint8, 16), 4) }},
		{"SAD16x16 short b", func() { SAD16x16(make([]uint8, 16*16), 16, make([]uint8, 4), 16) }},
		{"SSE4x4 stride smaller than size", func() { SSE4x4(make([]uint8, 16), 2, make([]uint8, 16), 4) }},
		{"SATD4x4 short a", func() { SATD4x4(make([]uint8, 3), 4, make([]uint8, 16), 4) }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s: expected panic, got none", c.name)
				}
			}()
			c.fn()
		})
	}
}
