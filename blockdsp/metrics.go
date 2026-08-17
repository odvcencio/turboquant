package blockdsp

// sad computes the sum of absolute differences between two size x size
// uint8 blocks. a and b are laid out with row stride aStride and bStride
// respectively; both slices must hold at least (size-1)*stride+size bytes,
// or sad panics.
func sad(a []uint8, aStride int, b []uint8, bStride int, size int) int32 {
	requireBlock(a, aStride, size, "a")
	requireBlock(b, bStride, size, "b")
	var sum int32
	for y := 0; y < size; y++ {
		ar := a[y*aStride : y*aStride+size]
		br := b[y*bStride : y*bStride+size]
		for x := 0; x < size; x++ {
			d := int32(ar[x]) - int32(br[x])
			if d < 0 {
				d = -d
			}
			sum += d
		}
	}
	return sum
}

// sse computes the sum of squared errors between two size x size uint8
// blocks, with the same stride and panic contract as sad.
func sse(a []uint8, aStride int, b []uint8, bStride int, size int) int32 {
	requireBlock(a, aStride, size, "a")
	requireBlock(b, bStride, size, "b")
	var sum int32
	for y := 0; y < size; y++ {
		ar := a[y*aStride : y*aStride+size]
		br := b[y*bStride : y*bStride+size]
		for x := 0; x < size; x++ {
			d := int32(ar[x]) - int32(br[x])
			sum += d * d
		}
	}
	return sum
}

func requireBlock(block []uint8, stride, size int, name string) {
	if stride < size {
		panic("blockdsp: " + name + " stride smaller than block size")
	}
	need := (size-1)*stride + size
	if len(block) < need {
		panic("blockdsp: " + name + " slice too short for stride and block size")
	}
}

// SAD4x4 returns the sum of absolute differences between two 4x4 uint8
// blocks addressed by a and b with row strides aStride and bStride.
func SAD4x4(a []uint8, aStride int, b []uint8, bStride int) int32 {
	return sad(a, aStride, b, bStride, 4)
}

// SAD16x16 returns the sum of absolute differences between two 16x16 uint8
// blocks addressed by a and b with row strides aStride and bStride. This is
// VP8's whole-macroblock luma distortion metric for 16x16 mode search.
func SAD16x16(a []uint8, aStride int, b []uint8, bStride int) int32 {
	return sad(a, aStride, b, bStride, 16)
}

// SSE4x4 returns the sum of squared errors between two 4x4 uint8 blocks
// addressed by a and b with row strides aStride and bStride.
func SSE4x4(a []uint8, aStride int, b []uint8, bStride int) int32 {
	return sse(a, aStride, b, bStride, 4)
}

// SSE16x16 returns the sum of squared errors between two 16x16 uint8 blocks
// addressed by a and b with row strides aStride and bStride.
func SSE16x16(a []uint8, aStride int, b []uint8, bStride int) int32 {
	return sse(a, aStride, b, bStride, 16)
}

// SATD4x4 returns the Hadamard-transformed sum of absolute differences
// (SATD) between two 4x4 uint8 blocks: the pixel-difference block a-b is
// Hadamard transformed with the same M butterfly FWHT4x4 derives (RFC 6386
// section 14.3's pattern, applied here as a pure two-pass add/subtract with
// no final division -- this is a rate-distortion cost metric, not a
// bitstream coefficient, so there is no dequantization step to match), the
// 16 resulting coefficients are summed in absolute value, and the sum is
// right-shifted by 1.
//
// The final >>1 is a scaling convention, not a normative requirement: SATD
// is not part of the VP8 bitstream. It follows the common video-encoder
// practice (e.g. H.264/VP8/VP9 mode-decision cost functions) of halving the
// raw Hadamard-domain sum so an SATD score sits in roughly the same
// magnitude range as SAD on the same block, which keeps a single lambda
// usable across both cost functions in RD mode search. Callers that need
// the unscaled sum can multiply the result by 2.
func SATD4x4(a []uint8, aStride int, b []uint8, bStride int) int32 {
	requireBlock(a, aStride, 4, "a")
	requireBlock(b, bStride, 4, "b")

	var diff [16]int32
	for y := 0; y < 4; y++ {
		ar := a[y*aStride : y*aStride+4]
		br := b[y*bStride : y*bStride+4]
		for x := 0; x < 4; x++ {
			diff[y*4+x] = int32(ar[x]) - int32(br[x])
		}
	}

	var mid [16]int32
	for i := 0; i < 4; i++ {
		p0 := diff[i+0]
		p1 := diff[i+4]
		p2 := diff[i+8]
		p3 := diff[i+12]

		a1 := p0 + p3
		b1 := p1 + p2
		c1 := p1 - p2
		d1 := p0 - p3

		mid[i+0] = a1 + b1
		mid[i+4] = c1 + d1
		mid[i+8] = a1 - b1
		mid[i+12] = d1 - c1
	}

	var sum int32
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

		out0 := a1 + b1
		out1 := c1 + d1
		out2 := a1 - b1
		out3 := d1 - c1

		sum += abs32(out0) + abs32(out1) + abs32(out2) + abs32(out3)
	}
	return sum >> 1
}

func abs32(v int32) int32 {
	if v < 0 {
		return -v
	}
	return v
}
