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

func TestValidatePackedRejectsWrongLength(t *testing.T) {
	err := ValidatePacked(8, 2, []byte{1})
	if err == nil {
		t.Fatal("expected error for packed length mismatch")
	}
}

func TestValidateIPQuantizedRejectsWrongShape(t *testing.T) {
	err := ValidateIPQuantized(8, 3, IPQuantized{
		MSE:     []byte{1},
		Signs:   []byte{1},
		ResNorm: 1,
	})
	if err == nil {
		t.Fatal("expected error for invalid IP payload shape")
	}
}

func TestValidatePreparedQueryRejectsWrongShape(t *testing.T) {
	err := ValidatePreparedQuery(8, PreparedQuery{
		signLUT: make([]float32, 255),
		rotY:    make([]float32, 8),
	})
	if err == nil {
		t.Fatal("expected error for invalid prepared query")
	}
}

func TestValidatePreparedQueryRejectsWrongMSELUTShape(t *testing.T) {
	err := ValidatePreparedQuery(8, PreparedQuery{
		signLUT:     make([]float32, 256),
		mseLUT:      make([]float32, 1),
		rotY:        make([]float32, 8),
		mseBitWidth: 1,
	})
	if err == nil {
		t.Fatal("expected error for invalid prepared MSE LUT")
	}
}

func TestValidateGPUPreparedDataRejectsWrongShape(t *testing.T) {
	_, err := ValidateGPUPreparedData(8, 3, GPUPreparedData{
		MSE:      []byte{1},
		Signs:    []byte{1},
		ResNorms: []float32{1},
	})
	if err == nil {
		t.Fatal("expected error for invalid GPU prepared data")
	}
}

func TestValidateGPUPreparedDataRejectsInvalidNorm(t *testing.T) {
	_, err := ValidateGPUPreparedData(8, 3, GPUPreparedData{
		MSE:      make([]byte, PackedSize(8, 2)),
		Signs:    make([]byte, 1),
		ResNorms: []float32{float32(math.NaN())},
	})
	if err == nil {
		t.Fatal("expected error for invalid GPU prepared norm")
	}
}

func TestValidateGPUPreparedDataAcceptsValidShape(t *testing.T) {
	count, err := ValidateGPUPreparedData(8, 3, GPUPreparedData{
		MSE:      make([]byte, PackedSize(8, 2)*2),
		Signs:    make([]byte, 2),
		ResNorms: []float32{1, 2},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if count != 2 {
		t.Fatalf("count = %d want 2", count)
	}
}

func TestValidateGPUPreparedDataRejectsWrongRankShape(t *testing.T) {
	_, err := ValidateGPUPreparedData(8, 3, GPUPreparedData{
		MSE:           make([]byte, PackedSize(8, 2)*2),
		Signs:         make([]byte, 2),
		ResNorms:      []float32{1, 2},
		TieBreakRanks: []uint32{1},
	})
	if err == nil {
		t.Fatal("expected error for invalid GPU tie-break rank shape")
	}
}
