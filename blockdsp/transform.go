package blockdsp

// Fixed-point trigonometric constants shared by the forward and inverse 4x4
// DCT-like transform, RFC 6386 section 14.4:
//
//	cospi8Q16 = round(65536 * sqrt(2) * cos(pi/8)) = 85627
//	sinpi8Q16 = round(65536 * sqrt(2) * sin(pi/8)) = 35468
//
// RFC 6386's reference decoder avoids a 17-bit constant for cos by using the
// identity x*a = x + x*(a-1), storing cospi8sqrt2minus1 = 20091 = 85627 -
// 65536 and computing ip[12] + ((ip[12] * 20091) >> 16). For an arithmetic
// right shift (floor division by a power of two), (a*65536 + a*20091) >> 16
// equals a + ((a*20091) >> 16) exactly, because a*65536 is itself an exact
// multiple of 65536 and contributes no fractional bits to the shift. This
// package uses the single-multiply form (a*85627)>>16 throughout: it is
// arithmetically identical to the RFC's two-step form and easier to derive
// the forward transform from.
const (
	cospi8Q16 = 85627 // 65536 * sqrt(2) * cos(pi/8)
	sinpi8Q16 = 35468 // 65536 * sqrt(2) * sin(pi/8)
)

// IDCT4x4 inverts VP8's 4x4 DCT-like transform, RFC 6386 section 14.4,
// bit-exact. coeff holds dequantized coefficients in raster order; the
// result is the residual block in raster order (not yet added to a
// predictor, not clipped to uint8 -- callers that reconstruct pixels add the
// predictor and clip separately, matching the RFC's section 14.5 split
// between transform inversion and predictor summation).
//
// This is the exact algorithm golang.org/x/image/vp8's unexported
// (*Decoder).inverseDCT4 implements (reindexed there into a 2D array instead
// of a flat, strided one); see oracle_test.go for the cross-oracle test that
// pins the two against each other on randomized input.
func IDCT4x4(coeff *[16]int16) [16]int16 {
	var mid [16]int32
	for i := 0; i < 4; i++ {
		p0 := int32(coeff[i+0])
		p1 := int32(coeff[i+4])
		p2 := int32(coeff[i+8])
		p3 := int32(coeff[i+12])

		a1 := p0 + p2
		b1 := p0 - p2
		c1 := ((p1 * sinpi8Q16) >> 16) - ((p3 * cospi8Q16) >> 16)
		d1 := ((p1 * cospi8Q16) >> 16) + ((p3 * sinpi8Q16) >> 16)

		mid[i+0] = a1 + d1
		mid[i+4] = b1 + c1
		mid[i+8] = b1 - c1
		mid[i+12] = a1 - d1
	}

	var out [16]int16
	for i := 0; i < 4; i++ {
		row := i * 4
		p0 := mid[row+0]
		p1 := mid[row+1]
		p2 := mid[row+2]
		p3 := mid[row+3]

		a1 := p0 + p2
		b1 := p0 - p2
		c1 := ((p1 * sinpi8Q16) >> 16) - ((p3 * cospi8Q16) >> 16)
		d1 := ((p1 * cospi8Q16) >> 16) + ((p3 * sinpi8Q16) >> 16)

		out[row+0] = int16((a1 + d1 + 4) >> 3)
		out[row+1] = int16((b1 + c1 + 4) >> 3)
		out[row+2] = int16((b1 - c1 + 4) >> 3)
		out[row+3] = int16((a1 - d1 + 4) >> 3)
	}
	return out
}

// FDCT4x4 is the forward companion to IDCT4x4. RFC 6386 section 14 states
// that the forward transform is not normative, so this derivation, not
// libwebp's or any other encoder's source, defines it.
//
// # Derivation
//
// Model IDCT4x4's per-axis butterfly as a linear map: for an input vector
// (p0,p1,p2,p3), the RFC computes (with alpha = sqrt(2)cos(pi/8), beta =
// sqrt(2)sin(pi/8), ignoring fixed-point truncation for this derivation):
//
//	q0 = p0 + p2 + alpha*p1 + beta*p3
//	q1 = p0 - p2 + beta*p1  - alpha*p3
//	q2 = p0 - p2 - beta*p1  + alpha*p3
//	q3 = p0 + p2 - alpha*p1 - beta*p3
//
// i.e. q = B*p for the 4x4 matrix
//
//	B = [ 1  alpha  1   beta ]
//	    [ 1  beta  -1  -alpha]
//	    [ 1 -beta  -1   alpha]
//	    [ 1 -alpha  1  -beta ]
//
// Since alpha^2 + beta^2 = 2, B's rows are pairwise orthogonal and each row
// has squared norm 1+alpha^2+1+beta^2 = 4, so B*B^T = 4*I and
// B^-1 = B^T/4. IDCT4x4 applies B on the left (down columns), then B^T on
// the right (across rows: op[c] = sum_k row[k]*B[k][c], which is the row
// vector times B^T's transpose == B's columns, algebraically B applied to
// rows via B^T), then divides by 8 with a round-half-up bias of 4:
//
//	IDCT(C) = round_div8(B * C * B^T)
//
// Solving round_div8(B*C*B^T) = X for C (ignoring rounding):
//
//	C = B^-1 * (8*X) * (B^T)^-1 = (B^T/4)*(8X)*(B/4) = (1/2)*B^T*X*B
//
// FDCT4x4 below computes exactly this: pass 1 applies B^T down columns
// (same alpha/beta fixed-point constants as the inverse, combined
// differently: y1 = alpha*(p0-p3) + beta*(p1-p2), y3 = beta*(p0-p3) -
// alpha*(p1-p2)); pass 2 applies B across rows the same way; the final
// step divides by 2 with a round-half-up bias of 1.
//
// # Round-trip bound
//
// For a DC-only (flat) input block, X[i] = v for all 16 positions: pass 1
// collapses every column difference term to zero (all rows equal), leaving
// only the pure-sum terms, giving an intermediate matrix that is 16v at
// position 0 and exactly 0 everywhere else -- no fixed-point multiply, hence
// no truncation, is on the path from v to 16v. The final >>1 divides this
// exact even number to 8v exactly. Feeding C = {8v at position 0, else 0}
// through IDCT4x4 hits the same zero-multiply path in reverse and yields v
// everywhere, exactly. So DC-only blocks round-trip through
// FDCT4x4+IDCT4x4 with zero error, by construction, not by observation.
//
// For general blocks, each of the four fixed-point multiply-shift sites in
// FDCT4x4 (2 per pass) truncates toward -infinity, losing under 1 unit of
// the ideal real-valued alpha/beta product; the final round-half-up >>1
// adds at most another 0.5 unit of bias. IDCT4x4 has the same shape of
// error budget on the way back (2 truncating multiply-shifts per pass, one
// round-half-up >>3). TestFDCTIDCTRoundTrip asserts the resulting bound
// empirically over the full realistic residual domain ([-255,255], since
// FDCT4x4 transforms src-minus-predictor differences between two uint8
// pixels) plus adversarial max-amplitude int16 input, and documents the
// observed maximum next to the assertion.
func FDCT4x4(src *[16]int16) [16]int16 {
	var mid [16]int32
	for i := 0; i < 4; i++ {
		p0 := int32(src[i+0])
		p1 := int32(src[i+4])
		p2 := int32(src[i+8])
		p3 := int32(src[i+12])

		diffA := p0 - p3
		diffB := p1 - p2

		mid[i+0] = p0 + p1 + p2 + p3
		mid[i+4] = (diffA*cospi8Q16 + diffB*sinpi8Q16) >> 16
		mid[i+8] = p0 - p1 - p2 + p3
		mid[i+12] = (diffA*sinpi8Q16 - diffB*cospi8Q16) >> 16
	}

	var out [16]int16
	for i := 0; i < 4; i++ {
		row := i * 4
		p0 := mid[row+0]
		p1 := mid[row+1]
		p2 := mid[row+2]
		p3 := mid[row+3]

		diffA := p0 - p3
		diffB := p1 - p2

		r0 := p0 + p1 + p2 + p3
		r1 := (diffA*cospi8Q16 + diffB*sinpi8Q16) >> 16
		r2 := p0 - p1 - p2 + p3
		r3 := (diffA*sinpi8Q16 - diffB*cospi8Q16) >> 16

		out[row+0] = int16((r0 + 1) >> 1)
		out[row+1] = int16((r1 + 1) >> 1)
		out[row+2] = int16((r2 + 1) >> 1)
		out[row+3] = int16((r3 + 1) >> 1)
	}
	return out
}

// IWHT4x4 inverts VP8's 4x4 Walsh-Hadamard transform, RFC 6386 section 14.3,
// bit-exact, including the section's specified +3 rounding bias (not the
// round-half-up +4 a naive >>3 would use -- the RFC specifies +3, and a
// conforming decoder must match it exactly).
//
// coeff holds the 16 dequantized Y2 (luma DC) coefficients in raster order;
// the result is the 16 DC residual values, one per luma subblock, also in
// raster order, matching RFC 6386 section 14.2's placement rule (the
// subblock at row i, column j takes result[4*i+j] as its own coefficient
// 0). This is a direct transcription of RFC 6386's
// vp8_short_inv_walsh4x4_c; see oracle_test.go for the cross-oracle pin
// against golang.org/x/image/vp8's inverseWHT16.
func IWHT4x4(coeff *[16]int16) [16]int16 {
	var mid [16]int32
	for i := 0; i < 4; i++ {
		p0 := int32(coeff[i+0])
		p1 := int32(coeff[i+4])
		p2 := int32(coeff[i+8])
		p3 := int32(coeff[i+12])

		a1 := p0 + p3
		b1 := p1 + p2
		c1 := p1 - p2
		d1 := p0 - p3

		mid[i+0] = a1 + b1
		mid[i+4] = c1 + d1
		mid[i+8] = a1 - b1
		mid[i+12] = d1 - c1
	}

	var out [16]int16
	for i := 0; i < 4; i++ {
		row := i * 4
		p0 := mid[row+0]
		p1 := mid[row+1]
		p2 := mid[row+2]
		p3 := mid[row+3]

		a1 := p0 + p3
		b1 := p1 + p2
		c1 := p1 - p2
		d1 := p0 - p3

		a2 := a1 + b1
		b2 := c1 + d1
		c2 := a1 - b1
		d2 := d1 - c1

		out[row+0] = int16((a2 + 3) >> 3)
		out[row+1] = int16((b2 + 3) >> 3)
		out[row+2] = int16((c2 + 3) >> 3)
		out[row+3] = int16((d2 + 3) >> 3)
	}
	return out
}

// FWHT4x4 is the forward companion to IWHT4x4.
//
// # Derivation
//
// IWHT4x4's per-axis butterfly is linear: for (p0,p1,p2,p3), the RFC's
// vp8_short_inv_walsh4x4_c computes q = M*p for
//
//	M = [ 1  1  1  1]
//	    [ 1  1 -1 -1]
//	    [ 1 -1 -1  1]
//	    [ 1 -1  1 -1]
//
// M is symmetric (M = M^T) and its rows are pairwise orthogonal with
// squared norm 4, so M*M = 4*I and M^-1 = M/4. By the same reasoning as
// FDCT4x4's derivation, IWHT4x4(C) = round_div8(M*C*M), so the exact
// forward is C = M^-1*(8X)*M^-1 = (M/4)*(8X)*(M/4) = (1/2)*M*X*M: apply the
// identical add/subtract butterfly M uses (no fixed-point multiply
// anywhere -- M has only +-1 entries, so both passes are exact integer
// arithmetic), then divide by 2 with a round-half-up bias of 1.
//
// # Round-trip bound
//
// Because M has no fixed-point multiply, the only source of round-trip
// error is the forward's own final >>1: M*X*M is always computed exactly
// in integers, and it is divisible by 2 as often as the input parity makes
// it so. For a DC-only (flat, X[i] = v for all 16 positions) block, M*X*M
// is 16v at position 0 and exactly 0 elsewhere (the same collapse argument
// as FDCT4x4's derivation, with M's all-ones first row playing the role of
// B's first row), so the final >>1 divides an exact even number and
// FWHT4x4+IWHT4x4 round-trips flat blocks with zero error. For general
// blocks, TestFWHTIWHTRoundTrip proves and asserts a tight closed-form
// bound: each of the 16 forward outputs loses at most a rounding
// remainder of exactly 1 unit (from >>1 on a value that may be odd), that
// error is at most 0.5 relative to the exact real-valued C = (M*X*M)/2 (the
// stored integer either equals that value, when M*X*M is even, or is 0.5
// below it, when M*X*M is odd). Write the stored C as (M*X*M)/2 - eps with
// eps in {0, 0.5} per coefficient. Then, using M*M = 4*I:
//
//	M*C*M = M*((M*X*M)/2 - eps)*M = 8*X - M*eps*M
//
// M*eps*M sums 16 terms of the form +-1 * eps * +-1, each with magnitude at
// most 0.5, so |M*eps*M| <= 8 per coefficient in the worst case (all 16
// terms aligned in sign). The RFC's inverse computes
// floor((M*C*M + 3)/8) = X + floor((3 - M*eps*M)/8); with M*eps*M ranging
// over [-8, 8], the argument of the outer floor ranges over [-5, 11], whose
// floor-divide-by-8 is in {-1, 0, 1}. So the round-trip error is bounded by
// exactly 1, derived (not measured) from the truncation budget of the
// forward transform's single >>1 site.
func FWHT4x4(src *[16]int16) [16]int16 {
	var mid [16]int32
	for i := 0; i < 4; i++ {
		p0 := int32(src[i+0])
		p1 := int32(src[i+4])
		p2 := int32(src[i+8])
		p3 := int32(src[i+12])

		a1 := p0 + p3
		b1 := p1 + p2
		c1 := p1 - p2
		d1 := p0 - p3

		mid[i+0] = a1 + b1
		mid[i+4] = c1 + d1
		mid[i+8] = a1 - b1
		mid[i+12] = d1 - c1
	}

	var out [16]int16
	for i := 0; i < 4; i++ {
		row := i * 4
		p0 := mid[row+0]
		p1 := mid[row+1]
		p2 := mid[row+2]
		p3 := mid[row+3]

		a1 := p0 + p3
		b1 := p1 + p2
		c1 := p1 - p2
		d1 := p0 - p3

		a2 := a1 + b1
		b2 := c1 + d1
		c2 := a1 - b1
		d2 := d1 - c1

		out[row+0] = int16((a2 + 1) >> 1)
		out[row+1] = int16((b2 + 1) >> 1)
		out[row+2] = int16((c2 + 1) >> 1)
		out[row+3] = int16((d2 + 1) >> 1)
	}
	return out
}
