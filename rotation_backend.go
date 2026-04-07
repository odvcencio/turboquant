package turboquant

import (
	"math"
	"math/rand"
)

type rotationKind uint8

const (
	rotationKindDense rotationKind = iota + 1
	rotationKindHadamard
)

type rotationBlock struct {
	offset int
	size   int
}

type rotationState struct {
	kind rotationKind
	dim  int

	dense []float32

	perm   []int
	signs1 []float32
	signs2 []float32
	blocks []rotationBlock
}

func newDenseRotation(dim int, rng *rand.Rand) rotationState {
	return rotationState{
		kind:  rotationKindDense,
		dim:   dim,
		dense: generateRotation(dim, rng),
	}
}

func newDenseRotationFromMatrix(dim int, matrix []float32) rotationState {
	copied := make([]float32, len(matrix))
	copy(copied, matrix)
	return rotationState{
		kind:  rotationKindDense,
		dim:   dim,
		dense: copied,
	}
}

func newHadamardRotation(dim int, rng *rand.Rand) rotationState {
	perm := rng.Perm(dim)
	signs1 := make([]float32, dim)
	signs2 := make([]float32, dim)
	for i := 0; i < dim; i++ {
		signs1[i] = randomSign(rng)
		signs2[i] = randomSign(rng)
	}
	return rotationState{
		kind:   rotationKindHadamard,
		dim:    dim,
		perm:   perm,
		signs1: signs1,
		signs2: signs2,
		blocks: hadamardBlocks(dim),
	}
}

func randomSign(rng *rand.Rand) float32 {
	if rng.Intn(2) == 0 {
		return -1
	}
	return 1
}

func hadamardBlocks(dim int) []rotationBlock {
	blocks := make([]rotationBlock, 0, bitsSet(dim))
	offset := 0
	remaining := dim
	for remaining > 0 {
		size := highestPowerOfTwoLE(remaining)
		blocks = append(blocks, rotationBlock{offset: offset, size: size})
		offset += size
		remaining -= size
	}
	return blocks
}

func bitsSet(v int) int {
	count := 0
	for v > 0 {
		count += v & 1
		v >>= 1
	}
	if count == 0 {
		return 1
	}
	return count
}

func highestPowerOfTwoLE(v int) int {
	p := 1
	for p<<1 <= v {
		p <<= 1
	}
	return p
}

func (r rotationState) kindString() string {
	switch r.kind {
	case rotationKindDense:
		return "dense"
	case rotationKindHadamard:
		return "hadamard"
	default:
		return "unknown"
	}
}

func (r rotationState) apply(dst, src, work []float32) {
	switch r.kind {
	case rotationKindDense:
		rotate(dst, src, r.dense, r.dim)
	case rotationKindHadamard:
		for i, p := range r.perm {
			work[i] = src[p] * r.signs1[i]
		}
		for _, block := range r.blocks {
			fwhtNormalizedInPlace(work[block.offset : block.offset+block.size])
		}
		for i, p := range r.perm {
			dst[p] = work[i] * r.signs2[i]
		}
	}
}

func (r rotationState) applyInverse(dst, src, work []float32) {
	switch r.kind {
	case rotationKindDense:
		rotateInverse(dst, src, r.dense, r.dim)
	case rotationKindHadamard:
		for i, p := range r.perm {
			work[i] = src[p] * r.signs2[i]
		}
		for _, block := range r.blocks {
			fwhtNormalizedInPlace(work[block.offset : block.offset+block.size])
		}
		for i, p := range r.perm {
			dst[p] = work[i] * r.signs1[i]
		}
	}
}

func (r rotationState) matrix() []float32 {
	if r.kind == rotationKindDense {
		copied := make([]float32, len(r.dense))
		copy(copied, r.dense)
		return copied
	}

	matrix := make([]float32, r.dim*r.dim)
	basis := make([]float32, r.dim)
	column := make([]float32, r.dim)
	work := make([]float32, r.dim)
	for j := 0; j < r.dim; j++ {
		for i := range basis {
			basis[i] = 0
		}
		basis[j] = 1
		r.apply(column, basis, work)
		for i, v := range column {
			matrix[i*r.dim+j] = v
		}
	}
	return matrix
}

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
