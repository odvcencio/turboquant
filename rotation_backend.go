package turboquant

import (
	"math"
	"math/rand"
)

type rotationKind uint8

const (
	rotationKindDense         rotationKind = iota + 1 // 1
	rotationKindHadamard                              // 2, legacy single round
	rotationKindHadamardMulti                         // 3, round count in header
)

// DefaultHadamardRounds is the number of randomized Walsh-Hadamard rounds
// that the default structured-rotation constructors use. A single round
// isolates energy inside its power-of-two block decomposition: a one-hot
// input rotated with one round stays exactly zero outside its block. Fresh
// permutations between rounds mix that energy across every block, which
// matches the paper's Theorem 1 worst-case distortion bound. See
// rotation_test.go for the adversarial tests that gate this choice.
const DefaultHadamardRounds = 3

type rotationBlock struct {
	offset int
	size   int
}

// hadamardRound holds one randomized Walsh-Hadamard round: a permutation and
// two independent sign vectors. apply reads signs1 before the transform and
// signs2 after it; applyInverse reads them in the opposite order.
type hadamardRound struct {
	perm   []int
	signs1 []float32
	signs2 []float32
}

type rotationState struct {
	kind rotationKind
	dim  int

	dense []float32

	// rounds holds every Hadamard round in application order. A legacy
	// single-round rotation (kind == rotationKindHadamard) always has
	// len(rounds) == 1. Every round shares the same block decomposition,
	// because the decomposition depends only on dim.
	rounds []hadamardRound
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

// drawHadamardRound draws one round's permutation and sign vectors from rng.
func drawHadamardRound(dim int, rng *rand.Rand) hadamardRound {
	perm := rng.Perm(dim)
	signs1 := make([]float32, dim)
	signs2 := make([]float32, dim)
	for i := 0; i < dim; i++ {
		signs1[i] = randomSign(rng)
		signs2[i] = randomSign(rng)
	}
	return hadamardRound{perm: perm, signs1: signs1, signs2: signs2}
}

// newHadamardRotation builds the legacy single-round structured Walsh-
// Hadamard rotation. It exists so that pre-Phase-2 serialized payloads
// (rotationKindHadamard, the 25-byte header) decode with the exact numerics
// they were written with. New code should call newHadamardRotationRounds or
// buildHadamardRotation.
func newHadamardRotation(dim int, rng *rand.Rand) rotationState {
	return rotationState{
		kind:   rotationKindHadamard,
		dim:    dim,
		rounds: []hadamardRound{drawHadamardRound(dim, rng)},
		blocks: hadamardBlocks(dim),
	}
}

// newHadamardRotationRounds builds a randomized Walsh-Hadamard rotation with
// rounds independent rounds, each with a fresh permutation and fresh sign
// vectors drawn in sequence from rng. rounds must be 1 or more.
func newHadamardRotationRounds(dim, rounds int, rng *rand.Rand) rotationState {
	rs := make([]hadamardRound, rounds)
	for i := range rs {
		rs[i] = drawHadamardRound(dim, rng)
	}
	return rotationState{
		kind:   rotationKindHadamardMulti,
		dim:    dim,
		rounds: rs,
		blocks: hadamardBlocks(dim),
	}
}

// buildHadamardRotation builds a randomized Walsh-Hadamard rotation with the
// given round count. A round count of 1 reproduces the legacy single-round
// transform bit-for-bit, including its rotationKindHadamard tag, so that
// serialization of a one-round quantizer stays byte-compatible with the
// pre-Phase-2 25-byte header.
func buildHadamardRotation(dim, rounds int, rng *rand.Rand) rotationState {
	if rounds == 1 {
		return newHadamardRotation(dim, rng)
	}
	return newHadamardRotationRounds(dim, rounds, rng)
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
	case rotationKindHadamardMulti:
		return "hadamard-multi"
	default:
		return "unknown"
	}
}

func (r rotationState) apply(dst, src, work []float32) {
	switch r.kind {
	case rotationKindDense:
		rotate(dst, src, r.dense, r.dim)
	case rotationKindHadamard, rotationKindHadamardMulti:
		applyHadamardRound(r.rounds[0], r.blocks, dst, src, work)
		for i := 1; i < len(r.rounds); i++ {
			applyHadamardRound(r.rounds[i], r.blocks, dst, dst, work)
		}
	}
}

func (r rotationState) applyInverse(dst, src, work []float32) {
	switch r.kind {
	case rotationKindDense:
		rotateInverse(dst, src, r.dense, r.dim)
	case rotationKindHadamard, rotationKindHadamardMulti:
		last := len(r.rounds) - 1
		applyHadamardRoundInverse(r.rounds[last], r.blocks, dst, src, work)
		for i := last - 1; i >= 0; i-- {
			applyHadamardRoundInverse(r.rounds[i], r.blocks, dst, dst, work)
		}
	}
}

// applyHadamardRound applies one randomized Walsh-Hadamard round: gather with
// signs1, run the per-block fast Walsh-Hadamard transform (FWHT), then
// scatter with signs2. src is fully consumed into work before dst is
// written, so dst may alias src.
func applyHadamardRound(rd hadamardRound, blocks []rotationBlock, dst, src, work []float32) {
	for i, p := range rd.perm {
		work[i] = src[p] * rd.signs1[i]
	}
	for _, block := range blocks {
		fwhtNormalizedInPlace(work[block.offset : block.offset+block.size])
	}
	for i, p := range rd.perm {
		dst[p] = work[i] * rd.signs2[i]
	}
}

// applyHadamardRoundInverse applies the inverse of one randomized
// Walsh-Hadamard round: the same structure as applyHadamardRound with
// signs1 and signs2 swapped. src is fully consumed into work before dst is
// written, so dst may alias src.
func applyHadamardRoundInverse(rd hadamardRound, blocks []rotationBlock, dst, src, work []float32) {
	for i, p := range rd.perm {
		work[i] = src[p] * rd.signs2[i]
	}
	for _, block := range blocks {
		fwhtNormalizedInPlace(work[block.offset : block.offset+block.size])
	}
	for i, p := range rd.perm {
		dst[p] = work[i] * rd.signs1[i]
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
