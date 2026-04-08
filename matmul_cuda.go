//go:build linux && amd64 && cgo && cuda

package turboquant

import (
	"fmt"

	"github.com/odvcencio/turboquant/internal/cudaruntime"
)

// DenseMatmul computes C = A * B on GPU using cuBLAS.
// A is (m x k), B is (k x n), C is (m x n). All row-major float32.
func DenseMatmul(dst, a, b []float32, m, k, n int) error {
	if len(a) < m*k {
		return fmt.Errorf("turboquant: buffer A too small: need %d, got %d", m*k, len(a))
	}
	if len(b) < k*n {
		return fmt.Errorf("turboquant: buffer B too small: need %d, got %d", k*n, len(b))
	}
	if len(dst) < m*n {
		return fmt.Errorf("turboquant: buffer dst too small: need %d, got %d", m*n, len(dst))
	}

	rt, err := cudaruntime.New()
	if err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	defer rt.Close()

	dA, err := rt.Alloc(m * k * 4)
	if err != nil {
		return fmt.Errorf("turboquant: alloc A: %w", err)
	}
	defer rt.Free(dA)

	dB, err := rt.Alloc(k * n * 4)
	if err != nil {
		return fmt.Errorf("turboquant: alloc B: %w", err)
	}
	defer rt.Free(dB)

	dC, err := rt.Alloc(m * n * 4)
	if err != nil {
		return fmt.Errorf("turboquant: alloc C: %w", err)
	}
	defer rt.Free(dC)

	if err := rt.HtoDFloat32(dA, a[:m*k]); err != nil {
		return fmt.Errorf("turboquant: HtoD A: %w", err)
	}
	if err := rt.HtoDFloat32(dB, b[:k*n]); err != nil {
		return fmt.Errorf("turboquant: HtoD B: %w", err)
	}

	if err := rt.LaunchSgemm(dA, dB, dC, m, k, n); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}

	if err := rt.DtoHFloat32(dst[:m*n], dC); err != nil {
		return fmt.Errorf("turboquant: DtoH C: %w", err)
	}
	return nil
}

// DenseMatmulTransB computes C = A * B^T on GPU using cuBLAS.
// A is (m x k), B is (n x k), C is (m x n). All row-major float32.
func DenseMatmulTransB(dst, a, b []float32, m, k, n int) error {
	if len(a) < m*k {
		return fmt.Errorf("turboquant: buffer A too small: need %d, got %d", m*k, len(a))
	}
	if len(b) < n*k {
		return fmt.Errorf("turboquant: buffer B too small: need %d, got %d", n*k, len(b))
	}
	if len(dst) < m*n {
		return fmt.Errorf("turboquant: buffer dst too small: need %d, got %d", m*n, len(dst))
	}

	rt, err := cudaruntime.New()
	if err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	defer rt.Close()

	dA, err := rt.Alloc(m * k * 4)
	if err != nil {
		return fmt.Errorf("turboquant: alloc A: %w", err)
	}
	defer rt.Free(dA)

	dB, err := rt.Alloc(n * k * 4)
	if err != nil {
		return fmt.Errorf("turboquant: alloc B: %w", err)
	}
	defer rt.Free(dB)

	dC, err := rt.Alloc(m * n * 4)
	if err != nil {
		return fmt.Errorf("turboquant: alloc C: %w", err)
	}
	defer rt.Free(dC)

	if err := rt.HtoDFloat32(dA, a[:m*k]); err != nil {
		return fmt.Errorf("turboquant: HtoD A: %w", err)
	}
	if err := rt.HtoDFloat32(dB, b[:n*k]); err != nil {
		return fmt.Errorf("turboquant: HtoD B: %w", err)
	}

	if err := rt.LaunchSgemmTransB(dA, dB, dC, m, k, n); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}

	if err := rt.DtoHFloat32(dst[:m*n], dC); err != nil {
		return fmt.Errorf("turboquant: DtoH C: %w", err)
	}
	return nil
}
