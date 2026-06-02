package mesh

import (
	"encoding/binary"
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

// EncodePositions quantizes an interleaved []float32 position stream (length
// 3N) into bits-per-component fixed-point codes packed little-endian. It
// computes a per-axis AABB and maps each component into [0, maxCode] where
// maxCode = 2^bits - 1. A degenerate axis (min == max) gets Scale = 1 so that
// every code 0 decodes back to Min exactly.
func EncodePositions(pos []float32, bits int) ([]byte, PositionParams) {
	p := PositionParams{Bits: bits}
	n := len(pos) / 3

	var mn, mx [3]float32
	if n == 0 {
		// No vertices: leave zeroed Min, unit Scale so decode is well defined.
		p.Scale = [3]float32{1, 1, 1}
		return nil, p
	}
	for a := 0; a < 3; a++ {
		mn[a] = pos[a]
		mx[a] = pos[a]
	}
	for v := 0; v < n; v++ {
		for a := 0; a < 3; a++ {
			c := pos[v*3+a]
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
			q := float32(math.Round(float64((pos[v*3+a] - p.Min[a]) / p.Scale[a])))
			if q < 0 {
				q = 0
			}
			if q > maxCode {
				q = maxCode
			}
			out = appendUint(out, uint32(q), bits)
		}
	}
	return out, p
}

// DecodePositions reverses EncodePositions, producing an interleaved
// []float32 of length 3n via the multiply-add Min[a] + code*Scale[a].
func DecodePositions(packed []byte, p PositionParams, n int) []float32 {
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
