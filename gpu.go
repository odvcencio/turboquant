package turboquant

import (
	"errors"
	"fmt"
	"math"
)

var ErrGPUBackendUnavailable = errors.New("turboquant: GPU backend unavailable on this platform")

const GPUPreparedTopKMax = 64

// GPUPreparedData stores a quantized IP corpus in contiguous buffers suitable
// for bulk upload into a prepared-query scorer backend.
type GPUPreparedData struct {
	MSE           []byte
	Signs         []byte
	ResNorms      []float32
	TieBreakRanks []uint32
}

// ValidateGPUPreparedData checks that data matches the given IP quantizer
// shape and returns the number of quantized vectors stored in it.
func ValidateGPUPreparedData(dim, bitWidth int, data GPUPreparedData) (int, error) {
	if err := validateDim(dim); err != nil {
		return 0, err
	}
	if err := validateIPBitWidth(bitWidth); err != nil {
		return 0, err
	}
	mseBytes, signBytes := IPQuantizedSizes(dim, bitWidth)
	count := len(data.ResNorms)
	if len(data.MSE) != count*mseBytes {
		return 0, fmt.Errorf("turboquant: expected GPU MSE payload length %d, got %d", count*mseBytes, len(data.MSE))
	}
	if len(data.Signs) != count*signBytes {
		return 0, fmt.Errorf("turboquant: expected GPU sign payload length %d, got %d", count*signBytes, len(data.Signs))
	}
	if len(data.TieBreakRanks) != 0 && len(data.TieBreakRanks) != count {
		return 0, fmt.Errorf("turboquant: expected GPU tie-break rank length %d, got %d", count, len(data.TieBreakRanks))
	}
	for i, norm := range data.ResNorms {
		if math.IsNaN(float64(norm)) || math.IsInf(float64(norm), 0) {
			return 0, fmt.Errorf("turboquant: invalid GPU residual norm at index %d: %v", i, norm)
		}
	}
	return count, nil
}

// PackGPUPreparedData flattens IP-quantized vectors into contiguous buffers for
// bulk upload into a GPU prepared-query scorer.
func (q *IPQuantizer) PackGPUPreparedData(vectors []IPQuantized) GPUPreparedData {
	mseBytes, signBytes := IPQuantizedSizes(q.dim, q.bitWidth)
	data := GPUPreparedData{
		MSE:           make([]byte, len(vectors)*mseBytes),
		Signs:         make([]byte, len(vectors)*signBytes),
		ResNorms:      make([]float32, len(vectors)),
		TieBreakRanks: make([]uint32, len(vectors)),
	}
	for i := range vectors {
		panicOnInvalid("(*IPQuantizer).PackGPUPreparedData", ValidateIPQuantized(q.dim, q.bitWidth, vectors[i]))
		copy(data.MSE[i*mseBytes:(i+1)*mseBytes], vectors[i].MSE)
		copy(data.Signs[i*signBytes:(i+1)*signBytes], vectors[i].Signs)
		data.ResNorms[i] = vectors[i].ResNorm
		data.TieBreakRanks[i] = uint32(i)
	}
	return data
}

// NewGPUPreparedScorer uploads a quantized IP corpus into an experimental GPU
// backend for repeated prepared-query scoring.
func (q *IPQuantizer) NewGPUPreparedScorer(vectors []IPQuantized) (*GPUPreparedScorer, error) {
	return q.NewGPUPreparedScorerFromData(q.PackGPUPreparedData(vectors))
}

// NewGPUPreparedScorerFromData uploads contiguous quantized buffers into an
// experimental GPU backend for repeated prepared-query scoring.
func (q *IPQuantizer) NewGPUPreparedScorerFromData(data GPUPreparedData) (*GPUPreparedScorer, error) {
	if _, err := ValidateGPUPreparedData(q.dim, q.bitWidth, data); err != nil {
		return nil, err
	}
	if preparedQueryMSELUTLen(q.dim, q.mse.bitWidth) == 0 {
		return nil, fmt.Errorf("turboquant: GPU prepared scorer requires prepared MSE LUT support for bit width %d", q.mse.bitWidth)
	}
	return newGPUPreparedScorer(q, data)
}
