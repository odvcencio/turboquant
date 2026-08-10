//go:build amd64

package turboquant

import "math"

// This file holds scalar reference copies used by the SIMD spike's parity
// tests and benchmarks (dot_simd_parity_test.go, dot_bench_amd64_test.go,
// dot_bench_simd_amd64_test.go). It is tagged plain amd64 (not gated on
// goexperiment.simd) so the "Generic" benchmarks are the SAME code whether
// or not GOEXPERIMENT=simd is set, making the non-experiment run a valid
// asm/generic baseline for the report's ns/op tables. Under a goexperiment.simd
// build, fwhtNormalizedInPlace and signedProjectedSum8 resolve to the
// vectorized kernels (the generic bodies move to rotation_fwht_generic.go /
// qjl_sign_generic.go, both excluded by that build's tag), so the "generic"
// comparison point needs its own copy here rather than calling the
// package's own (build-dependent) entry point.

// fwhtScalarRef is byte-for-byte the pre-spike fwhtNormalizedInPlace body.
func fwhtScalarRef(values []float32) {
	if len(values) == 1 {
		return
	}
	for step := 1; step < len(values); step <<= 1 {
		jump := step << 1
		for i := 0; i < len(values); i += jump {
			for j := i; j < i+step; j++ {
				a := values[j]
				b := values[j+step]
				values[j] = a + b
				values[j+step] = a - b
			}
		}
	}
	scale := float32(1.0 / math.Sqrt(float64(len(values))))
	for i := range values {
		values[i] *= scale
	}
}

// signedProjectedSum8ScalarRef is logically identical to the pre-spike
// signedProjectedSum8 body (a per-bit branch), just loop-folded here.
func signedProjectedSum8ScalarRef(signs byte, dots [8]float32) float32 {
	var sum float32
	for k := 0; k < 8; k++ {
		if signs&(1<<uint(k)) != 0 {
			sum += dots[k]
		} else {
			sum -= dots[k]
		}
	}
	return sum
}
