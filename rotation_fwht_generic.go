//go:build !(goexperiment.simd && amd64)

package turboquant

import "math"

// fwhtNormalizedInPlace runs the in-place normalized fast Walsh-Hadamard
// transform on values, whose length must be a power of two (the sole callers
// in rotation_backend.go always pass a block from hadamardBlocks, which
// decomposes dim into power-of-two chunks). A goexperiment.simd amd64 build
// uses the AVX2-vectorized variant instead; see rotation_fwht_simd_amd64.go.
func fwhtNormalizedInPlace(values []float32) {
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
