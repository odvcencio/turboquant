//go:build amd64 && !goexperiment.simd

package turboquant

import "unsafe"

// DotFloat32s computes the dot product of a and b using the hand-written SSE
// kernel. A goexperiment.simd build uses the simd-package kernel instead; see
// dot_simd_amd64.go.
func DotFloat32s(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("turboquant: DotFloat32s length mismatch")
	}
	if len(a) == 0 {
		return 0
	}
	return dotFloat32sSSE(unsafe.SliceData(a), unsafe.SliceData(b), len(a))
}
