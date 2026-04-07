//go:build !amd64

package turboquant

func dotFloat32Rows4(dst *[4]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 4*n {
		panic("turboquant: dotFloat32Rows4 row block too short")
	}
	dst[0] = dotFloat32s(rows[0:n], vec)
	dst[1] = dotFloat32s(rows[n:2*n], vec)
	dst[2] = dotFloat32s(rows[2*n:3*n], vec)
	dst[3] = dotFloat32s(rows[3*n:4*n], vec)
}

func dotFloat32Rows8(dst *[8]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 8*n {
		panic("turboquant: dotFloat32Rows8 row block too short")
	}
	for row := 0; row < 8; row++ {
		dst[row] = dotFloat32s(rows[row*n:(row+1)*n], vec)
	}
}

func dotFloat32Rows8Blocked(dst *[8]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 8*n {
		panic("turboquant: dotFloat32Rows8Blocked row block too short")
	}
	for i := range dst {
		dst[i] = 0
	}
	offset := 0
	for col := 0; col < n; col += 4 {
		width := 4
		if rem := n - col; rem < width {
			width = rem
		}
		for row := 0; row < 8; row++ {
			var sum float32
			for j := 0; j < width; j++ {
				sum += rows[offset+row*4+j] * vec[col+j]
			}
			dst[row] += sum
		}
		offset += 8 * 4
	}
}
