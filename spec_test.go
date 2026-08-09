package turboquant

import "testing"

func TestQuantizerSpecCopiesPortableState(t *testing.T) {
	q := NewHadamardWithSeed(10, 4, 123)
	spec := q.Spec()
	if spec.Dim != 10 || spec.BitWidth != 4 || spec.Seed != 123 {
		t.Fatalf("unexpected spec header: %+v", spec)
	}
	if spec.RotationKind != "hadamard-multi" {
		t.Fatalf("rotation kind = %q, want hadamard-multi", spec.RotationKind)
	}
	if len(spec.Rounds) != DefaultHadamardRounds {
		t.Fatalf("len(spec.Rounds) = %d want %d", len(spec.Rounds), DefaultHadamardRounds)
	}
	for i, round := range spec.Rounds {
		if len(round.Perm) != q.Dim() || len(round.Signs1) != q.Dim() || len(round.Signs2) != q.Dim() {
			t.Fatalf("round %d: unexpected lengths: %+v", i, round)
		}
	}
	// The deprecated single-round fields stay empty when Rounds has more
	// than one entry.
	if spec.Perm != nil || spec.Signs1 != nil || spec.Signs2 != nil {
		t.Fatalf("expected deprecated single-round fields to be empty for a multi-round spec")
	}
	if len(spec.Centroids) != 16 || len(spec.Boundaries) != 15 {
		t.Fatalf("unexpected codebook lengths: centroids=%d boundaries=%d", len(spec.Centroids), len(spec.Boundaries))
	}
	orig := q.Spec()
	spec.Rounds[0].Perm[0] = -1
	spec.Rounds[0].Signs1[0] = 0
	spec.Centroids[0] = 99
	again := q.Spec()
	if again.Rounds[0].Perm[0] != orig.Rounds[0].Perm[0] || again.Rounds[0].Signs1[0] != orig.Rounds[0].Signs1[0] || again.Centroids[0] != orig.Centroids[0] {
		t.Fatal("Spec returned aliased internal state")
	}
}

func TestQuantizerSpecSingleRoundPopulatesDeprecatedFields(t *testing.T) {
	q := NewHadamardRoundsWithSeed(10, 4, 1, 123)
	spec := q.Spec()
	if spec.RotationKind != "hadamard" {
		t.Fatalf("rotation kind = %q, want hadamard", spec.RotationKind)
	}
	if len(spec.Rounds) != 1 {
		t.Fatalf("len(spec.Rounds) = %d want 1", len(spec.Rounds))
	}
	if len(spec.Perm) != q.Dim() || len(spec.Signs1) != q.Dim() || len(spec.Signs2) != q.Dim() {
		t.Fatalf("expected deprecated single-round fields populated: %+v", spec)
	}
	for i := range spec.Perm {
		if spec.Perm[i] != spec.Rounds[0].Perm[i] {
			t.Fatalf("deprecated Perm[%d] = %d, Rounds[0].Perm[%d] = %d", i, spec.Perm[i], i, spec.Rounds[0].Perm[i])
		}
	}
}

// TestSpecRoundsMatchApply rebuilds the transform from QuantizerSpec.Rounds
// and compares against apply within a tight numeric tolerance.
func TestSpecRoundsMatchApply(t *testing.T) {
	q := NewHadamardWithSeed(64, 3, 7)
	spec := q.Spec()
	vec := randomUnitVector(64, newTestRNG())

	got := make([]float32, 64)
	q.rotation.apply(got, vec, make([]float32, 64))

	// Rebuild the transform from the exported spec fields only.
	rebuilt := append([]float32(nil), vec...)
	work := make([]float32, 64)
	next := make([]float32, 64)
	for _, round := range spec.Rounds {
		for i, p := range round.Perm {
			work[i] = rebuilt[p] * round.Signs1[i]
		}
		for _, block := range spec.Blocks {
			fwhtNormalizedInPlace(work[block.Offset : block.Offset+block.Size])
		}
		for i, p := range round.Perm {
			next[p] = work[i] * round.Signs2[i]
		}
		rebuilt, next = next, rebuilt
	}

	for i := range got {
		if diff := float64(got[i] - rebuilt[i]); diff > 1e-5 || diff < -1e-5 {
			t.Fatalf("coordinate %d: apply=%v rebuilt=%v diff=%v", i, got[i], rebuilt[i], diff)
		}
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
