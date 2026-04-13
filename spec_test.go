package turboquant

import "testing"

func TestQuantizerSpecCopiesPortableState(t *testing.T) {
	q := NewHadamardWithSeed(10, 4, 123)
	spec := q.Spec()
	if spec.Dim != 10 || spec.BitWidth != 4 || spec.Seed != 123 {
		t.Fatalf("unexpected spec header: %+v", spec)
	}
	if spec.RotationKind != "hadamard" {
		t.Fatalf("rotation kind = %q, want hadamard", spec.RotationKind)
	}
	if len(spec.Perm) != q.Dim() || len(spec.Signs1) != q.Dim() || len(spec.Signs2) != q.Dim() {
		t.Fatalf("unexpected hadamard spec lengths: %+v", spec)
	}
	if len(spec.Centroids) != 16 || len(spec.Boundaries) != 15 {
		t.Fatalf("unexpected codebook lengths: centroids=%d boundaries=%d", len(spec.Centroids), len(spec.Boundaries))
	}
	orig := q.Spec()
	spec.Perm[0] = -1
	spec.Signs1[0] = 0
	spec.Centroids[0] = 99
	again := q.Spec()
	if again.Perm[0] != orig.Perm[0] || again.Signs1[0] != orig.Signs1[0] || again.Centroids[0] != orig.Centroids[0] {
		t.Fatal("Spec returned aliased internal state")
	}
}

func TestQuantizerIndexAPIMatchesPackedPath(t *testing.T) {
	q := NewHadamardWithSeed(17, 4, 99)
	vec := randomUnitVector(17, newTestRNG())

	packed, norm := q.Quantize(vec)
	indices := make([]int, q.Dim())
	indexNorm := q.QuantizeIndicesTo(indices, vec)
	if indexNorm != norm {
		t.Fatalf("index norm = %v, packed norm = %v", indexNorm, norm)
	}
	fromPacked := make([]int, q.Dim())
	UnpackIndices(fromPacked, packed, q.Dim(), q.BitWidth())
	for i := range indices {
		if indices[i] != fromPacked[i] {
			t.Fatalf("index[%d] = %d, packed index = %d", i, indices[i], fromPacked[i])
		}
	}

	repacked := make([]byte, PackedSize(q.Dim(), q.BitWidth()))
	PackIndices(repacked, indices, q.BitWidth())
	for i := range packed {
		if repacked[i] != packed[i] {
			t.Fatalf("packed[%d] = %d, repacked = %d", i, packed[i], repacked[i])
		}
	}

	reconPacked := q.Dequantize(packed)
	reconIndices := make([]float32, q.Dim())
	q.DequantizeIndicesTo(reconIndices, indices)
	for i := range reconPacked {
		if reconIndices[i] != reconPacked[i] {
			t.Fatalf("recon[%d] = %v, packed recon = %v", i, reconIndices[i], reconPacked[i])
		}
	}
}

func TestQuantizerIndexAPIRejectsBadInputs(t *testing.T) {
	q := NewHadamardWithSeed(4, 2, 1)
	expectPanic(t, func() {
		q.QuantizeIndicesTo(make([]int, 3), []float32{1, 2, 3, 4})
	})
	expectPanic(t, func() {
		q.DequantizeIndicesTo(make([]float32, 4), []int{0, 1, 2, 4})
	})
}
