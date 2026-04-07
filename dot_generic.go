//go:build !amd64

package turboquant

func dotFloat32s(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("turboquant: dotFloat32s length mismatch")
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
