//go:build goexperiment.simd && amd64

package turboquant

import (
	"math"
	"simd/archsimd"
	"unsafe"
)

// fwhtNormalizedInPlace runs the in-place normalized fast Walsh-Hadamard
// transform on values, whose length must be a power of two (see the generic
// implementation's doc comment for why that invariant always holds here).
//
// The butterfly at a given step either combines two step-length contiguous
// runs (step >= 8, a plain 8-wide vector loop over both runs) or, for
// step == 4, combines the low and high 128-bit halves of a single 8-wide
// load (jump == 8 exactly matches the vector width, so values[i:i+8] holds
// both the "a" and "b" run for that butterfly in one register). Steps 1 and
// 2 interleave a/b at finer granularity than one lane and fall back to the
// scalar butterfly; those two stages are a fixed, small fraction of the
// O(n log n) work (2 of log2(n) stages) and their contribution shrinks as n
// grows. See the SIMD spike report for the stage-coverage math and measured
// results.
func fwhtNormalizedInPlace(values []float32) {
	n := len(values)
	if n == 1 {
		return
	}
	base := unsafe.SliceData(values)

	step := 1
	for ; step < n && step < 4; step <<= 1 {
		jump := step << 1
		for i := 0; i < n; i += jump {
			for j := i; j < i+step; j++ {
				a := values[j]
				b := values[j+step]
				values[j] = a + b
				values[j+step] = a - b
			}
		}
	}
	if step < n && step == 4 {
		for i := 0; i < n; i += 8 {
			v := archsimd.LoadFloat32x8(float32x8At(base, i))
			lo := v.GetLo()
			hi := v.GetHi()
			sum := lo.Add(hi)
			diff := lo.Sub(hi)
			v.SetLo(sum).SetHi(diff).Store(float32x8At(base, i))
		}
		step = 8
	}
	for ; step < n; step <<= 1 {
		jump := step << 1
		for i := 0; i < n; i += jump {
			j := i
			for ; j+8 <= i+step; j += 8 {
				va := archsimd.LoadFloat32x8(float32x8At(base, j))
				vb := archsimd.LoadFloat32x8(float32x8At(base, j+step))
				va.Add(vb).Store(float32x8At(base, j))
				va.Sub(vb).Store(float32x8At(base, j+step))
			}
			for ; j < i+step; j++ {
				a := values[j]
				b := values[j+step]
				values[j] = a + b
				values[j+step] = a - b
			}
		}
	}

	scale := float32(1.0 / math.Sqrt(float64(n)))
	scaleVec := archsimd.BroadcastFloat32x8(scale)
	i := 0
	for ; i+8 <= n; i += 8 {
		archsimd.LoadFloat32x8(float32x8At(base, i)).Mul(scaleVec).Store(float32x8At(base, i))
	}
	for ; i < n; i++ {
		values[i] *= scale
	}
}
