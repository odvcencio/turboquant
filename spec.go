package turboquant

import "fmt"

// RotationBlock describes one power-of-two block in the structured Hadamard
// rotation used by QuantizerSpec.
type RotationBlock struct {
	Offset int
	Size   int
}

// QuantizerSpec is a portable snapshot of the data needed to reproduce a
// Quantizer's transform and codebook outside the package.
type QuantizerSpec struct {
	Dim          int
	BitWidth     int
	Seed         int64
	RotationKind string
	Dense        []float32
	Perm         []int
	Signs1       []float32
	Signs2       []float32
	Blocks       []RotationBlock
	Centroids    []float32
	Boundaries   []float32
}

// Spec returns a copy of the quantizer's portable rotation and codebook data.
func (q *Quantizer) Spec() QuantizerSpec {
	spec := QuantizerSpec{
		Dim:          q.dim,
		BitWidth:     q.bitWidth,
		Seed:         q.seed,
		RotationKind: q.rotation.kindString(),
		Centroids:    cloneFloat32s(q.cb.centroids),
		Boundaries:   cloneFloat32s(q.cb.boundaries),
	}
	switch q.rotation.kind {
	case rotationKindDense:
		spec.Dense = cloneFloat32s(q.rotation.dense)
	case rotationKindHadamard:
		spec.Perm = append([]int(nil), q.rotation.perm...)
		spec.Signs1 = cloneFloat32s(q.rotation.signs1)
		spec.Signs2 = cloneFloat32s(q.rotation.signs2)
		spec.Blocks = make([]RotationBlock, len(q.rotation.blocks))
		for i, block := range q.rotation.blocks {
			spec.Blocks[i] = RotationBlock{Offset: block.offset, Size: block.size}
		}
	}
	return spec
}

// QuantizeIndicesTo compresses vec into one scalar codebook index per
// coordinate and returns the original vector norm. dst must have length q.Dim().
func (q *Quantizer) QuantizeIndicesTo(dst []int, vec []float32) float32 {
	if len(dst) != q.dim {
		panic(fmt.Sprintf("(*Quantizer).QuantizeIndicesTo: turboquant: expected destination length %d, got %d", q.dim, len(dst)))
	}
	panicOnInvalid("(*Quantizer).QuantizeIndicesTo", ValidateVector(q.dim, vec))
	buf := q.pool.Get().(*scratchBuf)
	defer q.pool.Put(buf)

	norm := vecNorm(vec)
	scale := float32(1.0)
	if norm > 1e-12 {
		scale = 1.0 / norm
	}
	for i, v := range vec {
		buf.tmp[i] = v * scale
	}
	q.rotation.apply(buf.rotated, buf.tmp, buf.work)
	for i := 0; i < q.dim; i++ {
		dst[i] = q.cb.nearestCentroid(buf.rotated[i])
	}
	return norm
}

// DequantizeIndicesTo reconstructs an approximate unit-norm vector from one
// scalar codebook index per coordinate. dst and indices must have length q.Dim().
func (q *Quantizer) DequantizeIndicesTo(dst []float32, indices []int) {
	if len(dst) != q.dim {
		panic(fmt.Sprintf("(*Quantizer).DequantizeIndicesTo: turboquant: expected destination length %d, got %d", q.dim, len(dst)))
	}
	if len(indices) != q.dim {
		panic(fmt.Sprintf("(*Quantizer).DequantizeIndicesTo: turboquant: expected index length %d, got %d", q.dim, len(indices)))
	}
	buf := q.pool.Get().(*scratchBuf)
	defer q.pool.Put(buf)
	levels := 1 << uint(q.bitWidth)
	for i, idx := range indices {
		if idx < 0 || idx >= levels {
			panic(fmt.Sprintf("(*Quantizer).DequantizeIndicesTo: turboquant: index %d out of range at coordinate %d", idx, i))
		}
		buf.rotated[i] = q.cb.centroidValue(idx)
	}
	q.rotation.applyInverse(dst, buf.rotated, buf.work)
}

// PackIndices packs scalar codebook indices into dst.
func PackIndices(dst []byte, indices []int, bitWidth int) {
	packIndices(dst, indices, bitWidth)
}

// UnpackIndices unpacks scalar codebook indices from src into dst.
func UnpackIndices(dst []int, src []byte, count, bitWidth int) {
	unpackIndices(dst, src, count, bitWidth)
}

func cloneFloat32s(in []float32) []float32 {
	if in == nil {
		return nil
	}
	out := make([]float32, len(in))
	copy(out, in)
	return out
}
