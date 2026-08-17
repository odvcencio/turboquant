package blockdsp

import (
	"math/rand"
	"testing"
)

// These benchmarks are the pure-Go baseline the tqwebp specification's
// WP-3 amd64/arm64 assembly kernels must beat (see the package doc
// comment's dispatch-trio note). Reported ns/op figures are recorded in
// the blockdsp implementation report.
//
// Every benchmark below stores its result into a package-level sink and
// reports b.N distinct inputs (see benchInputs*) rather than looping over a
// single fixed input. A pure function called in a tight loop on one
// unchanging input is a classic case the Go compiler can hoist or
// constant-fold away entirely; varying the input across iterations and
// consuming the output through a sink defeats both.

var (
	sinkI16 [16]int16
	sinkI32 int32
)

const benchInputCount = 1024

func genResidualBlocks(seed int64, n int) [][16]int16 {
	rng := rand.New(rand.NewSource(seed))
	blocks := make([][16]int16, n)
	for i := range blocks {
		blocks[i] = randResidualBlock(rng)
	}
	return blocks
}

func genBoundedBlocks(seed int64, n int, maxAbs int) [][16]int16 {
	rng := rand.New(rand.NewSource(seed))
	blocks := make([][16]int16, n)
	for i := range blocks {
		blocks[i] = randBoundedBlock(rng, maxAbs)
	}
	return blocks
}

func BenchmarkFDCT4x4(b *testing.B) {
	in := genResidualBlocks(1, benchInputCount)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkI16 = FDCT4x4(&in[i%benchInputCount])
	}
}

func BenchmarkIDCT4x4(b *testing.B) {
	residuals := genResidualBlocks(2, benchInputCount)
	in := make([][16]int16, benchInputCount)
	for i, r := range residuals {
		in[i] = FDCT4x4(&r)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkI16 = IDCT4x4(&in[i%benchInputCount])
	}
}

func BenchmarkFWHT4x4(b *testing.B) {
	in := genBoundedBlocks(3, benchInputCount, 2000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkI16 = FWHT4x4(&in[i%benchInputCount])
	}
}

func BenchmarkIWHT4x4(b *testing.B) {
	dc := genBoundedBlocks(4, benchInputCount, 2000)
	in := make([][16]int16, benchInputCount)
	for i, x := range dc {
		in[i] = FWHT4x4(&x)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkI16 = IWHT4x4(&in[i%benchInputCount])
	}
}

func genPlanes(seed int64, n, rows, stride int) [][]uint8 {
	rng := rand.New(rand.NewSource(seed))
	planes := make([][]uint8, n)
	for i := range planes {
		planes[i] = randPlane(rng, rows, stride)
	}
	return planes
}

func BenchmarkSAD4x4(b *testing.B) {
	as := genPlanes(5, benchInputCount, 4, 4)
	bs := genPlanes(6, benchInputCount, 4, 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := i % benchInputCount
		sinkI32 = SAD4x4(as[j], 4, bs[j], 4)
	}
}

func BenchmarkSAD16x16(b *testing.B) {
	as := genPlanes(7, benchInputCount, 16, 16)
	bs := genPlanes(8, benchInputCount, 16, 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := i % benchInputCount
		sinkI32 = SAD16x16(as[j], 16, bs[j], 16)
	}
}

func BenchmarkSSE4x4(b *testing.B) {
	as := genPlanes(9, benchInputCount, 4, 4)
	bs := genPlanes(10, benchInputCount, 4, 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := i % benchInputCount
		sinkI32 = SSE4x4(as[j], 4, bs[j], 4)
	}
}

func BenchmarkSSE16x16(b *testing.B) {
	as := genPlanes(11, benchInputCount, 16, 16)
	bs := genPlanes(12, benchInputCount, 16, 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := i % benchInputCount
		sinkI32 = SSE16x16(as[j], 16, bs[j], 16)
	}
}

func BenchmarkSATD4x4(b *testing.B) {
	as := genPlanes(13, benchInputCount, 4, 4)
	bs := genPlanes(14, benchInputCount, 4, 4)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := i % benchInputCount
		sinkI32 = SATD4x4(as[j], 4, bs[j], 4)
	}
}

func BenchmarkQuantizeBlock(b *testing.B) {
	in := genBoundedBlocks(15, benchInputCount, 2000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkI16 = QuantizeBlock(&in[i%benchInputCount], 21, 24)
	}
}

func BenchmarkDequantizeBlock(b *testing.B) {
	coeffs := genBoundedBlocks(16, benchInputCount, 100)
	in := make([][16]int16, benchInputCount)
	for i, c := range coeffs {
		in[i] = QuantizeBlock(&c, 21, 24)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkI16 = DequantizeBlock(&in[i%benchInputCount], 21, 24)
	}
}
