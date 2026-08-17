package blockdsp

// ZigZag is VP8's 4x4 coefficient scan order, RFC 6386 section 13.3 (the
// zigzag table used by DECODE_AND_APPLYSIGN's callers and reused unchanged
// for token coding). ZigZag[scanPos] gives the raster-order position (see
// the package doc comment for the raster layout) that scan position
// scanPos maps to. Position 0 in both scan and raster order is always the
// DC coefficient, so QuantizeBlock and DequantizeBlock operate directly on
// raster order and do not need ZigZag themselves; it is exported for
// callers that tokenize coefficients in scan order (band mapping, context
// modeling), per the tqwebp specification's token-writer package.
var ZigZag = [16]int{0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15}

// QuantizeBlock quantizes a 4x4 coefficient block with a uniform dead-zone
// quantizer matching VP8's quantizer semantics (RFC 6386 section 14.1):
// raster position 0 (the DC coefficient) is divided by dcFactor, and every
// other position (AC) is divided by acFactor, using Go's truncating integer
// division. Truncating division rounds toward zero, so any coefficient with
// |coeff| < factor quantizes to level 0 -- the dead zone. Both factors must
// be positive; QuantizeBlock panics otherwise.
//
// coeff, dcFactor, and acFactor are typically produced by dequantization
// factor tables such as RFC 6386 section 14.1's dc_qlookup/ac_qlookup
// (reproduced by the calling codec, not by this package: blockdsp is
// quantizer-index-policy-free, see the package doc comment).
func QuantizeBlock(coeff *[16]int16, dcFactor, acFactor int16) [16]int16 {
	if dcFactor <= 0 || acFactor <= 0 {
		panic("blockdsp: QuantizeBlock factors must be positive")
	}
	var levels [16]int16
	levels[0] = coeff[0] / dcFactor
	for i := 1; i < 16; i++ {
		levels[i] = coeff[i] / acFactor
	}
	return levels
}

// DequantizeBlock reconstructs a 4x4 coefficient block from quantized
// levels: raster position 0 (DC) is multiplied by dcFactor, and every other
// position (AC) is multiplied by acFactor, matching RFC 6386 section 14.1's
// "the multiplies are computed and stored using 16-bit signed integers."
// The multiply is performed in int32 and truncated to int16 on return,
// exactly as the reference decoder's 16-bit dequantized coefficient storage
// requires; DequantizeBlock does not clamp, matching the RFC's silence on
// overflow (realistic VP8 coefficient and quantizer-factor magnitudes never
// approach the int16 boundary, so no clamp is needed for RFC-conformant
// inputs; DequantizeBlock is a caller-trusts-input primitive, not a
// bitstream validator).
func DequantizeBlock(levels *[16]int16, dcFactor, acFactor int16) [16]int16 {
	var coeff [16]int16
	coeff[0] = int16(int32(levels[0]) * int32(dcFactor))
	for i := 1; i < 16; i++ {
		coeff[i] = int16(int32(levels[i]) * int32(acFactor))
	}
	return coeff
}
