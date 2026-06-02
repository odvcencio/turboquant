package mesh

import "math"

// Skinning attribute codecs: bone weights (u8 normalized) and joint indices
// (u8 or u16). These decode to a multiply (weights: code/255) and an integer
// widen (joints), cheap to run inline in a skinning shader.
//
// The shared little-endian helpers appendUint/readUint are defined once in
// positions.go and reused here.

// EncodeWeights stores 4 bone weights per vertex (length 4N) as u8 values.
// Each vertex's 4 weights are renormalized to sum to 1 before quantizing to
// round(wi*255). A vertex whose weights sum to zero is left as four zero bytes.
func EncodeWeights(w []float32) []byte {
	n := len(w) / 4
	out := make([]byte, 0, n*4)
	for v := 0; v < n; v++ {
		var sum float32
		for k := 0; k < 4; k++ {
			sum += w[v*4+k]
		}
		for k := 0; k < 4; k++ {
			var q uint32
			if sum != 0 {
				q = uint32(math.Round(float64(w[v*4+k] / sum * 255)))
				if q > 255 {
					q = 255
				}
			}
			out = append(out, byte(q))
		}
	}
	return out
}

// DecodeWeights reverses EncodeWeights: it reads 4 u8 codes per vertex, scales
// by 1/255, then renormalizes the 4 weights to sum to 1 (so quantization
// rounding does not bias the partition of unity). A vertex whose codes are all
// zero decodes to four zeros.
func DecodeWeights(packed []byte, n int) []float32 {
	out := make([]float32, n*4)
	for v := 0; v < n; v++ {
		var sum float32
		var tmp [4]float32
		for k := 0; k < 4; k++ {
			tmp[k] = float32(packed[v*4+k]) / 255
			sum += tmp[k]
		}
		for k := 0; k < 4; k++ {
			if sum != 0 {
				out[v*4+k] = tmp[k] / sum
			} else {
				out[v*4+k] = 0
			}
		}
	}
	return out
}

// EncodeJoints stores 4 joint indices per vertex (length 4N) at the given bit
// width (8 or 16), packed little-endian. Indices larger than the maximum
// representable value (255 for u8, 65535 for u16) are clamped to that maximum.
func EncodeJoints(j []uint32, bits int) []byte {
	max := uint32((1 << bits) - 1)
	out := make([]byte, 0, len(j)*((bits+7)/8))
	for _, idx := range j {
		if idx > max {
			idx = max
		}
		out = appendUint(out, idx, bits)
	}
	return out
}

// DecodeJoints reverses EncodeJoints, producing a []uint32 of length 4n.
func DecodeJoints(packed []byte, bits, n int) []uint32 {
	out := make([]uint32, n*4)
	off := 0
	for i := range out {
		v, next := readUint(packed, off, bits)
		off = next
		out[i] = v
	}
	return out
}
