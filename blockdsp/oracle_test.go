package blockdsp

import (
	"math/rand"
	"testing"
)

// This file vendors two small functions from golang.org/x/image/vp8's
// idct.go: the unexported (*Decoder).inverseDCT4 and (*Decoder).
// inverseWHT16 methods. They serve as this test file's correctness oracle
// for IDCT4x4 and IWHT4x4: x/image/vp8 is a complete, widely used, pure-Go
// VP8 DECODER, so its inverse-transform arithmetic is a second, independent
// expression of the exact same RFC 6386 section 14 algorithm blockdsp's own
// IDCT4x4/IWHT4x4 implement. Matching it on randomized input is strong
// evidence blockdsp's transforms decode real VP8 bitstreams correctly, not
// just RFC prose correctly.
//
// Why vendor instead of import: the methods are unexported (they hang off
// vp8.Decoder, which has no public constructor path that exposes bare
// 4x4-block transform calls), and turboquant's go.mod must stay
// dependency-free outside of tests (see the package doc comment and the
// blockdsp implementation report's dependency note). Vendoring a ~90-line,
// BSD-licensed, attributed snippet into a _test.go file is the mechanism
// that keeps both constraints: no non-test module dependency, and a real,
// independently written oracle rather than a self-comparison against
// blockdsp's own code.
//
// Provenance: golang.org/x/image/vp8/idct.go, fetched 2026-08-16 from
// https://raw.githubusercontent.com/golang/image/master/vp8/idct.go.
//
// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the golang.org/x/image LICENSE file.
//
// Adaptation from the original: both methods normally hang off a
// *vp8.Decoder and either accumulate their result into a clip8-saturated
// pixel buffer (inverseDCT4, which is macroblock reconstruction: predictor
// plus residual) or scatter their 16 results across a macroblock-sized
// coefficient array at a fixed offset (inverseWHT16, which places each
// result as luma subblock 0-coefficients). Neither shape is a pure 4x4
// transform. oracleInverseDCT4 and oracleInverseWHT4x4 below keep every
// arithmetic operation, constant, and evaluation order from the originals
// unchanged, and replace only the input/output plumbing: coefficients in as
// a plain [16]int16 (raster order, matching the originals' own flat, row-
// major indexing), untouched pre-clip pre-add transform output out as a
// plain [16]int32.

// oracleInverseDCT4 is adapted from (*vp8.Decoder).inverseDCT4.
func oracleInverseDCT4(coeff [16]int16) [16]int32 {
	const (
		c1 = 85627 // 65536 * cos(pi/8) * sqrt(2).
		c2 = 35468 // 65536 * sin(pi/8) * sqrt(2).
	)
	var m [4][4]int32
	coeffBase := 0
	for i := 0; i < 4; i++ {
		a := int32(coeff[coeffBase+0]) + int32(coeff[coeffBase+8])
		b := int32(coeff[coeffBase+0]) - int32(coeff[coeffBase+8])
		c := (int32(coeff[coeffBase+4])*c2)>>16 - (int32(coeff[coeffBase+12])*c1)>>16
		d := (int32(coeff[coeffBase+4])*c1)>>16 + (int32(coeff[coeffBase+12])*c2)>>16
		m[i][0] = a + d
		m[i][1] = b + c
		m[i][2] = b - c
		m[i][3] = a - d
		coeffBase++
	}
	var out [16]int32
	for j := 0; j < 4; j++ {
		dc := m[0][j] + 4
		a := dc + m[2][j]
		b := dc - m[2][j]
		c := (m[1][j]*c2)>>16 - (m[3][j]*c1)>>16
		d := (m[1][j]*c1)>>16 + (m[3][j]*c2)>>16
		out[j*4+0] = (a + d) >> 3
		out[j*4+1] = (b + c) >> 3
		out[j*4+2] = (b - c) >> 3
		out[j*4+3] = (a - d) >> 3
	}
	return out
}

// oracleInverseWHT4x4 is adapted from (*vp8.Decoder).inverseWHT16, reduced
// to the single 4x4 block that method computes 16 scattered copies of the
// plumbing around (see the file-level comment above).
func oracleInverseWHT4x4(coeff [16]int16) [16]int32 {
	var m [16]int32
	for i := 0; i < 4; i++ {
		a0 := int32(coeff[0+i]) + int32(coeff[12+i])
		a1 := int32(coeff[4+i]) + int32(coeff[8+i])
		a2 := int32(coeff[4+i]) - int32(coeff[8+i])
		a3 := int32(coeff[0+i]) - int32(coeff[12+i])
		m[0+i] = a0 + a1
		m[8+i] = a0 - a1
		m[4+i] = a3 + a2
		m[12+i] = a3 - a2
	}
	var out [16]int32
	for i := 0; i < 4; i++ {
		dc := m[0+i*4] + 3
		a0 := dc + m[3+i*4]
		a1 := m[1+i*4] + m[2+i*4]
		a2 := m[1+i*4] - m[2+i*4]
		a3 := dc - m[3+i*4]
		out[i*4+0] = (a0 + a1) >> 3
		out[i*4+1] = (a3 + a2) >> 3
		out[i*4+2] = (a0 - a1) >> 3
		out[i*4+3] = (a3 - a2) >> 3
	}
	return out
}

func randCoeffBlock(rng *rand.Rand, maxAbs int) [16]int16 {
	var b [16]int16
	for i := range b {
		b[i] = int16(rng.Intn(2*maxAbs+1) - maxAbs)
	}
	return b
}

// TestIDCT4x4MatchesOracle drives IDCT4x4 and oracleInverseDCT4 with the
// same randomized coefficient blocks and requires exact agreement. Both
// implement RFC 6386 section 14.4; the oracle is x/image/vp8's own
// unexported decoder method (see the file-level comment), so exact
// agreement here means blockdsp's inverse DCT decodes real VP8 bitstreams
// exactly as golang.org/x/image/webp would.
func TestIDCT4x4MatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	// Dequantized VP8 coefficients are stored in int16; sweep from small
	// magnitudes (typical after quantization) up to the full int16 range
	// (worst case, e.g. a corrupt or adversarial stream a decoder must
	// still process without divergence from a reference).
	for _, maxAbs := range []int{0, 1, 2, 8, 64, 512, 4096, 32767} {
		for trial := 0; trial < 200; trial++ {
			coeff := randCoeffBlock(rng, maxAbs)
			got := IDCT4x4(&coeff)
			want := oracleInverseDCT4(coeff)
			for i := 0; i < 16; i++ {
				if int32(got[i]) != want[i] {
					t.Fatalf("maxAbs=%d trial=%d pos=%d: coeff=%v got=%d want=%d",
						maxAbs, trial, i, coeff, got[i], want[i])
				}
			}
		}
	}
}

// TestIWHT4x4MatchesOracle is IWHT4x4's counterpart to
// TestIDCT4x4MatchesOracle, against x/image/vp8's inverseWHT16.
func TestIWHT4x4MatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, maxAbs := range []int{0, 1, 2, 8, 64, 512, 4096, 32767} {
		for trial := 0; trial < 200; trial++ {
			coeff := randCoeffBlock(rng, maxAbs)
			got := IWHT4x4(&coeff)
			want := oracleInverseWHT4x4(coeff)
			for i := 0; i < 16; i++ {
				if int32(got[i]) != want[i] {
					t.Fatalf("maxAbs=%d trial=%d pos=%d: coeff=%v got=%d want=%d",
						maxAbs, trial, i, coeff, got[i], want[i])
				}
			}
		}
	}
}
