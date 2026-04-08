//go:build arm64

package turboquant

import "unsafe"

//go:noescape
func dotFloat32sNEON(a, b *float32, n int) float32

func dotFloat32s(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("turboquant: dotFloat32s length mismatch")
	}
	if len(a) == 0 {
		return 0
	}
	return dotFloat32sNEON(unsafe.SliceData(a), unsafe.SliceData(b), len(a))
}
