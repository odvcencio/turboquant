package turboquant

import (
	"fmt"
	"math"
)

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
