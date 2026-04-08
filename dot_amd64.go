//go:build amd64

package turboquant

import "unsafe"

//go:noescape
func dotFloat32sSSE(a, b *float32, n int) float32

func DotFloat32s(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("turboquant: DotFloat32s length mismatch")
	}
	if len(a) == 0 {
		return 0
	}
	return dotFloat32sSSE(unsafe.SliceData(a), unsafe.SliceData(b), len(a))
}
