package mesh

import (
	"encoding/binary"
	"fmt"
	"math"
)

// PositionParams holds the per-axis dequantization parameters for a position
// stream. Decode is the multiply-add Min[a] + code*Scale[a]; this is the
// reference shader math consumed in Phase 3, so it is deliberately
// branch-light. The struct is a comparable value type (no slices) so it can be
// embedded in a JSON-portable DequantSpec.
type PositionParams struct {
	Bits  int        `json:"bits"`
	Min   [3]float32 `json:"min"`
	Scale [3]float32 `json:"scale"`
}

// validateBits panics with a clear, parameter-naming message unless bits is one
// of the two widths the codec supports (8 or 16). It is the public-entry-point
// guard that turns a deep, misleading appendUint/readUint panic into an
// actionable one. appendUint/readUint keep their own backstop panic.
func validateBits(bits int) {
	if bits != 8 && bits != 16 {
		panic(fmt.Sprintf("mesh: bits must be 8 or 16, got %d", bits))
	}
}

// isFinite reports whether v is neither NaN nor an infinity.
func isFinite(v float32) bool {
	return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0)
}

// EncodePositions quantizes an interleaved []float32 position stream into
// bits-per-component fixed-point codes packed little-endian. It computes a
// per-axis AABB and maps each component into [0, maxCode] where maxCode =
// 2^bits - 1. A degenerate axis (min == max) gets Scale = 1 so that every code
// 0 decodes back to Min exactly.
//
// bits must be 8 or 16; any other value panics.
//
// The input length must be exactly 3*N; a trailing partial vertex (length not a
// multiple of 3) is silently dropped.
//
// Non-finite inputs are sanitized: this function always produces a finite
// DequantSpec (finite Min and Scale) and, on decode, finite positions. The AABB
// pass ignores NaN/Inf components, and an axis with no finite values at all gets
// Min = 0, Scale = 1. In the pack loop a non-finite component is treated as
// code 0 — i.e. it decodes back to the axis minimum.
func EncodePositions(pos []float32, bits int) ([]byte, PositionParams) {
	validateBits(bits)
	p := PositionParams{Bits: bits}
	n := len(pos) / 3

	if n == 0 {
		// No vertices: leave zeroed Min, unit Scale so decode is well defined.
		p.Scale = [3]float32{1, 1, 1}
		return nil, p
	}

	// Per-axis AABB over finite components only. seen[a] tracks whether the
	// axis has at least one finite value; until then mn/mx are unset.
	var mn, mx [3]float32
	var seen [3]bool
	for v := 0; v < n; v++ {
		for a := 0; a < 3; a++ {
			c := pos[v*3+a]
			if !isFinite(c) {
				continue
			}
			if !seen[a] {
				mn[a], mx[a] = c, c
				seen[a] = true
				continue
			}
			if c < mn[a] {
				mn[a] = c
			}
			if c > mx[a] {
				mx[a] = c
			}
		}
	}

	maxCode := float32((uint32(1) << bits) - 1)
	for a := 0; a < 3; a++ {
		if !seen[a] {
			// Axis has no finite values: well-defined, finite identity params.
			p.Min[a] = 0
			p.Scale[a] = 1
			continue
		}
		p.Min[a] = mn[a]
		s := (mx[a] - mn[a]) / maxCode
		if s == 0 {
			// Degenerate axis: all codes are 0 and decode to Min.
			s = 1
		}
		p.Scale[a] = s
	}

	out := make([]byte, 0, n*3*((bits+7)/8))
	for v := 0; v < n; v++ {
		for a := 0; a < 3; a++ {
			c := pos[v*3+a]
			var q float32
			if isFinite(c) {
				q = float32(math.Round(float64((c - p.Min[a]) / p.Scale[a])))
				if q < 0 {
					q = 0
				}
				if q > maxCode {
					q = maxCode
				}
			}
			// Non-finite component falls through with q == 0 (axis minimum).
			out = appendUint(out, uint32(q), bits)
		}
	}
	return out, p
}

// DecodePositions reverses EncodePositions, producing an interleaved
// []float32 of length 3n via the multiply-add Min[a] + code*Scale[a].
//
// p.Bits must be 8 or 16; any other value panics. The packed buffer must hold
// at least n vertices (3*n components); a shorter buffer panics with a message
// naming the invariant.
func DecodePositions(packed []byte, p PositionParams, n int) []float32 {
	validateBits(p.Bits)
	need := n * 3 * ((p.Bits + 7) / 8)
	if len(packed) < need {
		panic(fmt.Sprintf("mesh: positions buffer too short: have %d bytes, need %d for %d vertices",
			len(packed), need, n))
	}
	out := make([]float32, n*3)
	off := 0
	for v := 0; v < n; v++ {
		for a := 0; a < 3; a++ {
			q, next := readUint(packed, off, p.Bits)
			off = next
			out[v*3+a] = p.Min[a] + float32(q)*p.Scale[a]
		}
	}
	return out
}

// appendUint appends v as a little-endian unsigned integer of the given bit
// width (8 -> 1 byte, 16 -> 2 bytes) to dst and returns the extended slice.
// This is the single shared little-endian writer for the mesh codec.
func appendUint(dst []byte, v uint32, bits int) []byte {
	switch bits {
	case 8:
		return append(dst, byte(v))
	case 16:
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(v))
		return append(dst, b[:]...)
	default:
		panic("mesh: appendUint supports only 8 or 16 bits")
	}
}

// readUint reads a little-endian unsigned integer of the given bit width from
// src starting at off, returning the value and the new offset. It is the
// single shared little-endian reader for the mesh codec.
func readUint(src []byte, off, bits int) (uint32, int) {
	switch bits {
	case 8:
		return uint32(src[off]), off + 1
	case 16:
		return uint32(binary.LittleEndian.Uint16(src[off : off+2])), off + 2
	default:
		panic("mesh: readUint supports only 8 or 16 bits")
	}
}
