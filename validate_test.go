package turboquant

import (
	"math"
	"testing"
)

func TestValidateVectorRejectsDimensionMismatch(t *testing.T) {
	err := ValidateVector(4, []float32{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
}

func TestValidateVectorRejectsNaN(t *testing.T) {
	err := ValidateVector(3, []float32{1, float32(math.NaN()), 3})
	if err == nil {
		t.Fatal("expected error for NaN")
	}
}

func TestValidateVectorRejectsInf(t *testing.T) {
	err := ValidateVector(3, []float32{1, float32(math.Inf(1)), 3})
	if err == nil {
		t.Fatal("expected error for Inf")
	}
}

func TestValidateVectorRejectsNegativeInf(t *testing.T) {
	err := ValidateVector(3, []float32{1, float32(math.Inf(-1)), 3})
	if err == nil {
		t.Fatal("expected error for negative Inf")
	}
}

func TestValidateVectorAcceptsValid(t *testing.T) {
	err := ValidateVector(3, []float32{1.0, 2.0, 3.0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateVectorAcceptsZero(t *testing.T) {
	err := ValidateVector(3, []float32{0, 0, 0})
	if err != nil {
		t.Fatalf("unexpected error for zero vector: %v", err)
	}
}
