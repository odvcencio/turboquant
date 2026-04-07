//go:build amd64

package turboquant

import "unsafe"

//go:noescape
func dotFloat32Rows4SSE(rows, vec *float32, n int, dst *float32)

//go:noescape
func dotFloat32Rows8SSE(rows, vec *float32, n int, dst *float32)

//go:noescape
func dotFloat32Rows8BlockedSSE(rows, vec *float32, n int, dst *float32)

func dotFloat32Rows4(dst *[4]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 4*n {
		panic("turboquant: dotFloat32Rows4 row block too short")
	}
	if n == 0 {
		dst[0], dst[1], dst[2], dst[3] = 0, 0, 0, 0
		return
	}
	dotFloat32Rows4SSE(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &dst[0])
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
	dotFloat32Rows8SSE(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &dst[0])
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
	dotFloat32Rows8BlockedSSE(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &dst[0])
}
