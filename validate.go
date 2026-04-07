package turboquant

import (
	"fmt"
	"math"
)

func validateDim(dim int) error {
	if dim < 2 {
		return fmt.Errorf("turboquant: dim must be >= 2")
	}
	return nil
}

func validateWireDim(dim int) error {
	if err := validateDim(dim); err != nil {
		return err
	}
	if dim > 65535 {
		return fmt.Errorf("turboquant: dim must be <= 65535 for wire format, got %d", dim)
	}
	return nil
}

func validateBitWidth(bitWidth int) error {
	if bitWidth < 1 || bitWidth > 8 {
		return fmt.Errorf("turboquant: bitWidth must be 1-8")
	}
	return nil
}

func validateIPBitWidth(bitWidth int) error {
	if bitWidth < 2 || bitWidth > 8 {
		return fmt.Errorf("turboquant: IP quantizer bitWidth must be 2-8")
	}
	return nil
}

// ValidateVector checks that vec has the expected dimension and contains
// no NaN or Inf values. Returns nil if valid.
func ValidateVector(dim int, vec []float32) error {
	if len(vec) != dim {
		return fmt.Errorf("turboquant: expected dimension %d, got %d", dim, len(vec))
	}
	for i, v := range vec {
		if math.IsNaN(float64(v)) {
			return fmt.Errorf("turboquant: NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			return fmt.Errorf("turboquant: Inf at index %d", i)
		}
	}
	return nil
}

// ValidatePacked checks that packed has the exact length expected for the given
// dimension and bit width.
func ValidatePacked(dim, bitWidth int, packed []byte) error {
	if err := validateDim(dim); err != nil {
		return err
	}
	if err := validateBitWidth(bitWidth); err != nil {
		return err
	}
	want := PackedSize(dim, bitWidth)
	if len(packed) != want {
		return fmt.Errorf("turboquant: expected packed length %d, got %d", want, len(packed))
	}
	return nil
}

// ValidateIPQuantized checks that qx has the correct MSE/sign payload sizes for
// the given dimension and bit width.
func ValidateIPQuantized(dim, bitWidth int, qx IPQuantized) error {
	if err := validateDim(dim); err != nil {
		return err
	}
	if err := validateIPBitWidth(bitWidth); err != nil {
		return err
	}
	if err := ValidatePacked(dim, bitWidth-1, qx.MSE); err != nil {
		return err
	}
	wantSigns := (dim + 7) / 8
	if len(qx.Signs) != wantSigns {
		return fmt.Errorf("turboquant: expected sign payload length %d, got %d", wantSigns, len(qx.Signs))
	}
	if math.IsNaN(float64(qx.ResNorm)) || math.IsInf(float64(qx.ResNorm), 0) {
		return fmt.Errorf("turboquant: invalid residual norm %v", qx.ResNorm)
	}
	return nil
}

// ValidatePreparedQuery checks that pq is sized for the given dimension.
func ValidatePreparedQuery(dim int, pq PreparedQuery) error {
	if err := validateDim(dim); err != nil {
		return err
	}
	wantSignLUT := preparedQuerySignLUTLen(dim)
	if len(pq.signLUT) != wantSignLUT {
		return fmt.Errorf("turboquant: expected prepared sign LUT length %d, got %d", wantSignLUT, len(pq.signLUT))
	}
	if len(pq.rotY) != dim {
		return fmt.Errorf("turboquant: expected prepared rotated query length %d, got %d", dim, len(pq.rotY))
	}
	if pq.mseBitWidth != 0 {
		if err := validateBitWidth(int(pq.mseBitWidth)); err != nil {
			return err
		}
		wantMSELUT := preparedQueryMSELUTLen(dim, int(pq.mseBitWidth))
		if len(pq.mseLUT) != wantMSELUT {
			return fmt.Errorf("turboquant: expected prepared MSE LUT length %d, got %d", wantMSELUT, len(pq.mseLUT))
		}
	} else if len(pq.mseLUT) != 0 {
		return fmt.Errorf("turboquant: unexpected prepared MSE LUT length %d without bit width", len(pq.mseLUT))
	}
	return nil
}

func panicOnInvalid(context string, err error) {
	if err != nil {
		panic(fmt.Sprintf("%s: %v", context, err))
	}
}
