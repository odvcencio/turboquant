package turboquant

import (
	"math"
	"math/rand"
)

// generateGaussianMatrix creates a d×d matrix with i.i.d. N(0,1) entries as float32.
func generateGaussianMatrix(dim int, rng *rand.Rand) []float32 {
	m := make([]float32, dim*dim)
	for i := range m {
		m[i] = float32(rng.NormFloat64())
	}
	return m
}

func blockProjectionRows8(projMatrix []float32, dim int) []float32 {
	if dim%8 != 0 {
		return nil
	}
	blocked := make([]float32, len(projMatrix))
	groupStride := dim * 8
	for group := 0; group < dim/8; group++ {
		baseRow := group * 8
		src := projMatrix[baseRow*dim : (baseRow+8)*dim]
		dst := blocked[group*groupStride : (group+1)*groupStride]
		blockProjectionRowGroup8(dst, src, dim)
	}
	return blocked
}

func blockProjectionRowGroup8(dst, rows []float32, dim int) {
	offset := 0
	for col := 0; col < dim; col += 4 {
		for row := 0; row < 8; row++ {
			src := rows[row*dim+col : row*dim+col+4]
			copy(dst[offset:offset+4], src)
			offset += 4
		}
	}
}

// qjlProject computes sign(S · x) and packs sign bits into dst.
// Returns ||x||₂.
func qjlProject(signs []byte, x, projMatrix []float32, dim int) float32 {
	norm := vecNorm(x)
	for i := range signs {
		signs[i] = 0
	}

	fullBytes := dim / 8
	for byteIdx := 0; byteIdx < fullBytes; byteIdx++ {
		base := byteIdx * 8
		var dots [8]float32
		dotFloat32Rows8(&dots, projMatrix[base*dim:(base+8)*dim], x)
		signs[byteIdx] = packProjectedSigns8(dots)
	}
	for i := fullBytes * 8; i < dim; i++ {
		row := i * dim
		if DotFloat32s(projMatrix[row:row+dim], x) >= 0 {
			signs[i/8] |= 1 << uint(i%8)
		}
	}
	return norm
}

func qjlProjectBlocked(signs []byte, x, projMatrix, proj8 []float32, dim int) float32 {
	if len(proj8) == 0 {
		return qjlProject(signs, x, projMatrix, dim)
	}
	norm := vecNorm(x)
	for i := range signs {
		signs[i] = 0
	}
	fullBytes := dim / 8
	groupStride := dim * 8
	for byteIdx := 0; byteIdx < fullBytes; byteIdx++ {
		var dots [8]float32
		block := proj8[byteIdx*groupStride : (byteIdx+1)*groupStride]
		dotFloat32Rows8Blocked(&dots, block, x)
		signs[byteIdx] = packProjectedSigns8(dots)
	}
	return norm
}

// qjlInnerProduct estimates <x, y> from QJL-quantized x and raw y.
// signs = sign(S·x), resNorm = ||x||, projMatrix = S
// Formula: √(π/2)/d · resNorm · Σᵢ sign_i · <sᵢ, y>
func qjlInnerProduct(signs []byte, resNorm float32, y, projMatrix []float32, dim int) float32 {
	scale := float32(math.Sqrt(math.Pi/2.0)) / float32(dim) * resNorm
	var sum float32
	fullBytes := dim / 8
	for byteIdx := 0; byteIdx < fullBytes; byteIdx++ {
		base := byteIdx * 8
		var dots [8]float32
		dotFloat32Rows8(&dots, projMatrix[base*dim:(base+8)*dim], y)
		sum += signedProjectedSum8(signs[byteIdx], dots)
	}
	for i := fullBytes * 8; i < dim; i++ {
		row := i * dim
		dot := DotFloat32s(projMatrix[row:row+dim], y)
		if signs[i/8]&(1<<uint(i%8)) != 0 {
			sum += dot
		} else {
			sum -= dot
		}
	}
	return scale * sum
}

func qjlInnerProductBlocked(signs []byte, resNorm float32, y, projMatrix, proj8 []float32, dim int) float32 {
	if len(proj8) == 0 {
		return qjlInnerProduct(signs, resNorm, y, projMatrix, dim)
	}
	scale := float32(math.Sqrt(math.Pi/2.0)) / float32(dim) * resNorm
	var sum float32
	fullBytes := dim / 8
	groupStride := dim * 8
	for byteIdx := 0; byteIdx < fullBytes; byteIdx++ {
		var dots [8]float32
		block := proj8[byteIdx*groupStride : (byteIdx+1)*groupStride]
		dotFloat32Rows8Blocked(&dots, block, y)
		sum += signedProjectedSum8(signs[byteIdx], dots)
	}
	return scale * sum
}

func packProjectedSigns8(dots [8]float32) byte {
	var signs byte
	if dots[0] >= 0 {
		signs |= 1 << 0
	}
	if dots[1] >= 0 {
		signs |= 1 << 1
	}
	if dots[2] >= 0 {
		signs |= 1 << 2
	}
	if dots[3] >= 0 {
		signs |= 1 << 3
	}
	if dots[4] >= 0 {
		signs |= 1 << 4
	}
	if dots[5] >= 0 {
		signs |= 1 << 5
	}
	if dots[6] >= 0 {
		signs |= 1 << 6
	}
	if dots[7] >= 0 {
		signs |= 1 << 7
	}
	return signs
}

func signedProjectedSum8(signs byte, dots [8]float32) float32 {
	var sum float32
	if signs&(1<<0) != 0 {
		sum += dots[0]
	} else {
		sum -= dots[0]
	}
	if signs&(1<<1) != 0 {
		sum += dots[1]
	} else {
		sum -= dots[1]
	}
	if signs&(1<<2) != 0 {
		sum += dots[2]
	} else {
		sum -= dots[2]
	}
	if signs&(1<<3) != 0 {
		sum += dots[3]
	} else {
		sum -= dots[3]
	}
	if signs&(1<<4) != 0 {
		sum += dots[4]
	} else {
		sum -= dots[4]
	}
	if signs&(1<<5) != 0 {
		sum += dots[5]
	} else {
		sum -= dots[5]
	}
	if signs&(1<<6) != 0 {
		sum += dots[6]
	} else {
		sum -= dots[6]
	}
	if signs&(1<<7) != 0 {
		sum += dots[7]
	} else {
		sum -= dots[7]
	}
	return sum
}
