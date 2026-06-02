package mesh

import "math"

// Octahedral normal codec. A unit normal is projected onto the octahedron and
// stored as two signed-normalized components. Decode is the octahedral
// reconstruction formula: a handful of abs/add/mul ops plus one rsqrt, with a
// single branch for the lower hemisphere. This is the reference shader math
// for Phase 3, so it is kept deliberately branch-light.

// signNZ returns +1 for v >= 0 (including +0) and -1 otherwise. The "non-zero"
// convention (zero maps to +1) matches the standard octahedral fold.
func signNZ(v float32) float32 {
	if v >= 0 {
		return 1
	}
	return -1
}

func absf(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// octEncode maps a (not necessarily normalized) direction onto the [-1,1]^2
// octahedral domain.
func octEncode(x, y, z float32) (float32, float32) {
	l := absf(x) + absf(y) + absf(z)
	if l == 0 {
		return 0, 0
	}
	ox, oy := x/l, y/l
	if z < 0 {
		ox, oy = (1-absf(oy))*signNZ(x), (1-absf(ox))*signNZ(y)
	}
	return ox, oy
}

// octDecode reconstructs a unit direction from its octahedral encoding.
func octDecode(ox, oy float32) (float32, float32, float32) {
	z := 1 - absf(ox) - absf(oy)
	x, y := ox, oy
	if z < 0 {
		x, y = (1-absf(oy))*signNZ(ox), (1-absf(ox))*signNZ(oy)
	}
	il := float32(1.0 / math.Sqrt(float64(x*x+y*y+z*z)))
	return x * il, y * il, z * il
}

// snormEncode quantizes o in [-1,1] to a signed integer of the given bit width,
// stored as the low bits of a uint32. The symmetric range uses maxS =
// 2^(bits-1) - 1, so the most-positive and most-negative representable values
// map to +1 and -1 respectively.
func snormEncode(o float32, bits int) uint32 {
	maxS := float32(int32(1)<<(bits-1) - 1)
	if o > 1 {
		o = 1
	}
	if o < -1 {
		o = -1
	}
	q := int32(math.Round(float64(o * maxS)))
	return uint32(q) & ((1 << bits) - 1)
}

// snormDecode reverses snormEncode, sign-extending the bits-wide code stored in
// the low bits of u before scaling by 1/maxS.
func snormDecode(u uint32, bits int) float32 {
	shift := 32 - bits
	s := int32(u<<shift) >> shift
	maxS := float32(int32(1)<<(bits-1) - 1)
	return float32(s) / maxS
}

// EncodeNormals octahedral-encodes an interleaved unit-normal stream (length
// 3N) into two bits-wide snorm codes per normal, packed little-endian.
func EncodeNormals(nrm []float32, bits int) []byte {
	n := len(nrm) / 3
	out := make([]byte, 0, n*2*((bits+7)/8))
	for v := 0; v < n; v++ {
		ox, oy := octEncode(nrm[v*3+0], nrm[v*3+1], nrm[v*3+2])
		out = appendUint(out, snormEncode(ox, bits), bits)
		out = appendUint(out, snormEncode(oy, bits), bits)
	}
	return out
}

// DecodeNormals reverses EncodeNormals, producing an interleaved []float32 of
// unit normals (length 3n).
func DecodeNormals(packed []byte, bits, n int) []float32 {
	out := make([]float32, n*3)
	off := 0
	for v := 0; v < n; v++ {
		uo, next := readUint(packed, off, bits)
		off = next
		uy, next2 := readUint(packed, off, bits)
		off = next2
		ox := snormDecode(uo, bits)
		oy := snormDecode(uy, bits)
		x, y, z := octDecode(ox, oy)
		out[v*3+0] = x
		out[v*3+1] = y
		out[v*3+2] = z
	}
	return out
}
