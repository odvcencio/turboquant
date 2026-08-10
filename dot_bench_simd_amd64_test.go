//go:build goexperiment.simd && amd64

package turboquant

import (
	"math/rand"
	"testing"
	"unsafe"
)

// This file holds the "simd" (experimental simd/archsimd, AVX2) benchmark
// variants for the SIMD spike's target kernels, matching the generic/asm
// baselines in dot_bench_amd64_test.go dimension-for-dimension and RNG-seed-
// for-RNG-seed. Only compiles under GOEXPERIMENT=simd.

func BenchmarkDotFloat32sSIMD(b *testing.B) {
	rng := rand.New(rand.NewSource(100))
	for _, n := range benchDims {
		a, c := randFloat32s(rng, n), randFloat32s(rng, n)
		ap, cp := unsafe.SliceData(a), unsafe.SliceData(c)
		b.Run(dimName(n), func(b *testing.B) {
			var sink float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sink = dotFloat32sSIMD(ap, cp, n)
			}
			sinkFloat32 = sink
		})
	}
}

func BenchmarkDotFloat32Rows8SIMD(b *testing.B) {
	rng := rand.New(rand.NewSource(200))
	for _, n := range benchDims {
		rows, vec := randFloat32s(rng, 8*n), randFloat32s(rng, n)
		rp, vp := unsafe.SliceData(rows), unsafe.SliceData(vec)
		b.Run(dimName(n), func(b *testing.B) {
			var dst [8]float32
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				dotFloat32Rows8SIMD(rp, vp, n, &dst[0])
			}
			sinkFloat32 = dst[0]
		})
	}
}

func BenchmarkDotFloat32Rows8BlockedSIMD(b *testing.B) {
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
				dotFloat32Rows8BlockedSIMD(bp, vp, n, &dst[0])
			}
			sinkFloat32 = dst[0]
		})
	}
}

func BenchmarkFWHTSIMD(b *testing.B) {
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
					fwhtNormalizedInPlace(buf[offset : offset+size])
					offset += size
				}
			}
			sinkFloat32 = buf[0]
		})
	}
}

func BenchmarkSignedProjectedSum8SIMD(b *testing.B) {
	rng := rand.New(rand.NewSource(500))
	var dots [8]float32
	for i := range dots {
		dots[i] = float32(rng.NormFloat64())
	}
	signs := byte(rng.Intn(256))
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		sink = signedProjectedSum8(signs, dots)
	}
	sinkFloat32 = sink
}
