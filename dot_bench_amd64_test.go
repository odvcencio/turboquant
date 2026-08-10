//go:build amd64

package turboquant

import (
	"math/rand"
	"testing"
	"unsafe"
)

// This file holds the "generic" (plain Go) and "asm" (hand-written SSE)
// benchmark variants for the SIMD spike's target kernels. It is tagged plain
// amd64, not goexperiment.simd, so `go test -bench ...` produces the same
// generic/asm baseline numbers whether or not GOEXPERIMENT=simd is set; the
// simd variants live in dot_bench_simd_amd64_test.go and only compile under
// the experiment. Run both ways to compare:
//
//	go test -run '^$' -bench 'Dot|FWHT|Signed' -benchtime 2000x -count=3 .
//	GOEXPERIMENT=simd go test -run '^$' -bench 'Dot|FWHT|Signed' -benchtime 2000x -count=3 .

var benchDims = []int{64, 128, 256, 384, 1536}

// sinkFloat32 defeats dead-code elimination of the benchmarked call.
var sinkFloat32 float32

func randFloat32s(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

// dimName formats a dimension for use as a sub-benchmark name.
func dimName(n int) string {
	switch n {
	case 64:
		return "n=64"
	case 128:
		return "n=128"
	case 256:
		return "n=256"
	case 384:
		return "n=384"
	case 1536:
		return "n=1536"
	default:
		return "n=?"
	}
}

func rowsScalarRefBench(dst []float32, rows, vec []float32, numRows, n int) {
	for r := 0; r < numRows; r++ {
		dst[r] = dotFloat32sScalar(rows[r*n:(r+1)*n], vec)
	}
}

// fwhtBenchBlocks returns the power-of-two block sizes fwhtNormalizedInPlace
// is actually ever called with for a dimension n: hadamardBlocks' own
// decomposition (see rotation_backend.go), the same one applyHadamardRound
// uses per rotation. n itself is used when it's already a power of two.
func fwhtBenchBlocks(n int) []int {
	blocks := hadamardBlocks(n)
	sizes := make([]int, len(blocks))
	for i, blk := range blocks {
		sizes[i] = blk.size
	}
	return sizes
}

// ---- DotFloat32s: generic (plain Go loop) vs asm (hand SSE) ----

func BenchmarkDotFloat32sGeneric(b *testing.B) {
	rng := rand.New(rand.NewSource(100))
	for _, n := range benchDims {
		a, c := randFloat32s(rng, n), randFloat32s(rng, n)
		b.Run(dimName(n), func(b *testing.B) {
			var sink float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink = dotFloat32sScalar(a, c)
			}
			sinkFloat32 = sink
		})
	}
}

func BenchmarkDotFloat32sAsm(b *testing.B) {
	rng := rand.New(rand.NewSource(100))
	for _, n := range benchDims {
		a, c := randFloat32s(rng, n), randFloat32s(rng, n)
		ap, cp := unsafe.SliceData(a), unsafe.SliceData(c)
		b.Run(dimName(n), func(b *testing.B) {
			var sink float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink = dotFloat32sSSE(ap, cp, n)
			}
			sinkFloat32 = sink
		})
	}
}

// ---- dotFloat32Rows8: generic vs asm ----

func BenchmarkDotFloat32Rows8Generic(b *testing.B) {
	rng := rand.New(rand.NewSource(200))
	for _, n := range benchDims {
		rows, vec := randFloat32s(rng, 8*n), randFloat32s(rng, n)
		b.Run(dimName(n), func(b *testing.B) {
			var dst [8]float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rowsScalarRefBench(dst[:], rows, vec, 8, n)
			}
			sinkFloat32 = dst[0]
		})
	}
}

func BenchmarkDotFloat32Rows8Asm(b *testing.B) {
	rng := rand.New(rand.NewSource(200))
	for _, n := range benchDims {
		rows, vec := randFloat32s(rng, 8*n), randFloat32s(rng, n)
		rp, vp := unsafe.SliceData(rows), unsafe.SliceData(vec)
		b.Run(dimName(n), func(b *testing.B) {
			var dst [8]float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dotFloat32Rows8SSE(rp, vp, n, &dst[0])
			}
			sinkFloat32 = dst[0]
		})
	}
}

// ---- dotFloat32Rows8Blocked: generic vs asm, both on the same
// blockProjectionRowGroup8 buffer layout (unchanged by this spike) ----

func BenchmarkDotFloat32Rows8BlockedGeneric(b *testing.B) {
	rng := rand.New(rand.NewSource(300))
	for _, n := range benchDims {
		rows, vec := randFloat32s(rng, 8*n), randFloat32s(rng, n)
		b.Run(dimName(n), func(b *testing.B) {
			var dst [8]float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rowsScalarRefBench(dst[:], rows, vec, 8, n)
			}
			sinkFloat32 = dst[0]
		})
	}
}

func BenchmarkDotFloat32Rows8BlockedAsm(b *testing.B) {
	rng := rand.New(rand.NewSource(300))
	for _, n := range benchDims {
		rows, vec := randFloat32s(rng, 8*n), randFloat32s(rng, n)
		blocked := make([]float32, len(rows))
		blockProjectionRowGroup8(blocked, rows, n)
		bp, vp := unsafe.SliceData(blocked), unsafe.SliceData(vec)
		b.Run(dimName(n), func(b *testing.B) {
			var dst [8]float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dotFloat32Rows8BlockedSSE(bp, vp, n, &dst[0])
			}
			sinkFloat32 = dst[0]
		})
	}
}

// ---- FWHT: generic (pre-spike scalar butterfly) ----
// No "asm" baseline exists for FWHT in this repo (rotation_backend.go always
// used a pure Go butterfly); this is a two-way (generic vs simd) comparison.

func BenchmarkFWHTGeneric(b *testing.B) {
	rng := rand.New(rand.NewSource(400))
	for _, n := range benchDims {
		sizes := fwhtBenchBlocks(n)
		src := randFloat32s(rng, n)
		buf := make([]float32, n)
		b.Run(dimName(n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(buf, src)
				offset := 0
				for _, size := range sizes {
					fwhtScalarRef(buf[offset : offset+size])
					offset += size
				}
			}
			sinkFloat32 = buf[0]
		})
	}
}

// ---- signedProjectedSum8: generic (branchy) ----

func BenchmarkSignedProjectedSum8Generic(b *testing.B) {
	rng := rand.New(rand.NewSource(500))
	var dots [8]float32
	for i := range dots {
		dots[i] = float32(rng.NormFloat64())
	}
	signs := byte(rng.Intn(256))
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		sink = signedProjectedSum8ScalarRef(signs, dots)
	}
	sinkFloat32 = sink
}
