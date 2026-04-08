//go:build arm64

package turboquant

import "unsafe"

//go:noescape
func dotFloat32Rows8NEON(rows, vec *float32, n int, dst *float32)

//go:noescape
func dotFloat32Rows8BlockedNEON(rows, vec *float32, n int, dst *float32)

func dotFloat32Rows4(dst *[4]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 4*n {
		panic("turboquant: dotFloat32Rows4 row block too short")
	}
	dst[0] = DotFloat32s(rows[0:n], vec)
	dst[1] = DotFloat32s(rows[n:2*n], vec)
	dst[2] = DotFloat32s(rows[2*n:3*n], vec)
	dst[3] = DotFloat32s(rows[3*n:4*n], vec)
}

func dotFloat32Rows8(dst *[8]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 8*n {
		panic("turboquant: dotFloat32Rows8 row block too short")
	}
	if n == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	dotFloat32Rows8NEON(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &dst[0])
}

func dotFloat32Rows8Blocked(dst *[8]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 8*n {
		panic("turboquant: dotFloat32Rows8Blocked row block too short")
	}
	if n == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	dotFloat32Rows8BlockedNEON(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &dst[0])
}
