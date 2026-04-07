package turboquant

import (
	"errors"
	"testing"
)

func TestPackGPUPreparedData(t *testing.T) {
	q := NewIPHadamardWithSeed(16, 3, 42)
	rng := newTestRNG()
	x0 := randomUnitVector(16, rng)
	x1 := randomUnitVector(16, rng)
	qx0 := q.Quantize(x0)
	qx1 := q.Quantize(x1)

	data := q.PackGPUPreparedData([]IPQuantized{qx0, qx1})
	mseBytes, signBytes := IPQuantizedSizes(q.Dim(), q.BitWidth())
	if got, want := len(data.MSE), 2*mseBytes; got != want {
		t.Fatalf("MSE length = %d want %d", got, want)
	}
	if got, want := len(data.Signs), 2*signBytes; got != want {
		t.Fatalf("sign length = %d want %d", got, want)
	}
	if got := len(data.ResNorms); got != 2 {
		t.Fatalf("norm length = %d want 2", got)
	}
	if got, want := data.MSE[:mseBytes], qx0.MSE; !bytesEqual(got, want) {
		t.Fatal("first packed MSE payload mismatch")
	}
	if got, want := data.Signs[signBytes:], qx1.Signs; !bytesEqual(got, want) {
		t.Fatal("second packed sign payload mismatch")
	}
	if data.ResNorms[0] != qx0.ResNorm || data.ResNorms[1] != qx1.ResNorm {
		t.Fatal("packed residual norms mismatch")
	}
	if got := len(data.TieBreakRanks); got != 2 {
		t.Fatalf("rank length = %d want 2", got)
	}
	if data.TieBreakRanks[0] != 0 || data.TieBreakRanks[1] != 1 {
		t.Fatalf("tie-break ranks = %v want [0 1]", data.TieBreakRanks)
	}
}

func TestNewGPUPreparedScorerStub(t *testing.T) {
	q := NewIPHadamardWithSeed(16, 3, 42)
	x := randomUnitVector(16, newTestRNG())
	qx := q.Quantize(x)
	scorer, err := q.NewGPUPreparedScorer([]IPQuantized{qx})
	if err == nil {
		if scorer == nil {
			t.Fatal("expected non-nil scorer when GPU backend is available")
		}
		_ = scorer.Close()
		return
	}
	if !errors.Is(err, ErrGPUBackendUnavailable) {
		t.Fatalf("err = %v want nil or %v", err, ErrGPUBackendUnavailable)
	}
	if scorer != nil {
		t.Fatal("expected nil scorer on unsupported platform")
	}
}

func TestNewGPUPreparedScorerRejectsUnsupportedMSEBitWidth(t *testing.T) {
	q := NewIPHadamardWithSeed(16, 4, 42)
	x := randomUnitVector(16, newTestRNG())
	qx := q.Quantize(x)
	_, err := q.NewGPUPreparedScorer([]IPQuantized{qx})
	if err == nil {
		t.Fatal("expected error for unsupported GPU MSE bit width")
	}
	if errors.Is(err, ErrGPUBackendUnavailable) {
		t.Fatalf("err = %v want unsupported MSE bit-width error", err)
	}
}

func TestGPUPreparedScorerTopKStub(t *testing.T) {
	var scorer GPUPreparedScorer
	_, _, err := scorer.ScorePreparedQueryTopK(PreparedQuery{}, 1)
	if !errors.Is(err, ErrGPUBackendUnavailable) {
		t.Fatalf("err = %v want %v", err, ErrGPUBackendUnavailable)
	}
}

func TestGPUPreparedScorerBatchTopKStub(t *testing.T) {
	var scorer GPUPreparedScorer
	_, _, err := scorer.ScorePreparedQueriesTopK([]PreparedQuery{{}}, 1)
	if !errors.Is(err, ErrGPUBackendUnavailable) {
		t.Fatalf("err = %v want %v", err, ErrGPUBackendUnavailable)
	}
}

func TestGPUPreparedQueryBatchStub(t *testing.T) {
	var scorer GPUPreparedScorer
	batch, err := scorer.UploadPreparedQueries([]PreparedQuery{{}})
	if !errors.Is(err, ErrGPUBackendUnavailable) {
		t.Fatalf("err = %v want %v", err, ErrGPUBackendUnavailable)
	}
	if batch != nil {
		t.Fatal("expected nil uploaded batch on unsupported platform")
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
