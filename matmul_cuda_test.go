//go:build linux && amd64 && cgo && cuda

package turboquant

import (
	"math"
	"testing"
)

func TestDenseMatmulGPU(t *testing.T) {
	// [1,2,3; 4,5,6] x [7,8; 9,10; 11,12] = [58,64; 139,154]
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 8, 9, 10, 11, 12}
	dst := make([]float32, 4)
	if err := DenseMatmul(dst, a, b, 2, 3, 2); err != nil {
		t.Fatalf("DenseMatmul: %v", err)
	}
	expect := []float32{58, 64, 139, 154}
	for i, v := range expect {
		if math.Abs(float64(dst[i]-v)) > 1e-3 {
			t.Fatalf("matmul[%d]: got %v, want %v", i, dst[i], v)
		}
	}
}

func TestDenseMatmulTransBGPU(t *testing.T) {
	// A = [1,2,3; 4,5,6], B = [7,9,11; 8,10,12] (2x3), C = A * B^T
	// B^T = [7,8; 9,10; 11,12], so C = [58,64; 139,154]
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{7, 9, 11, 8, 10, 12} // B is (2x3), B^T is (3x2)
	dst := make([]float32, 4)
	if err := DenseMatmulTransB(dst, a, b, 2, 3, 2); err != nil {
		t.Fatalf("DenseMatmulTransB: %v", err)
	}
	expect := []float32{58, 64, 139, 154}
	for i, v := range expect {
		if math.Abs(float64(dst[i]-v)) > 1e-3 {
			t.Fatalf("matmulTransB[%d]: got %v, want %v", i, dst[i], v)
		}
	}
}

func TestDenseMatmulGPUIdentity(t *testing.T) {
	// A = I(3), B = [1,2,3; 4,5,6; 7,8,9], C = I * B = B
	a := []float32{1, 0, 0, 0, 1, 0, 0, 0, 1}
	b := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
	dst := make([]float32, 9)
	if err := DenseMatmul(dst, a, b, 3, 3, 3); err != nil {
		t.Fatalf("DenseMatmul identity: %v", err)
	}
	for i, v := range b {
		if math.Abs(float64(dst[i]-v)) > 1e-3 {
			t.Fatalf("identity matmul[%d]: got %v, want %v", i, dst[i], v)
		}
	}
}

func TestDenseMatmulGPULarger(t *testing.T) {
	m, k, n := 64, 384, 384
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%17) * 0.01
	}
	for i := range b {
		b[i] = float32(i%13) * 0.01
	}

	gpu := make([]float32, m*n)
	if err := DenseMatmul(gpu, a, b, m, k, n); err != nil {
		t.Fatalf("DenseMatmul large: %v", err)
	}

	// Compare against CPU reference
	cpu := make([]float32, m*n)
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*k+p] * b[p*n+j]
			}
			cpu[i*n+j] = sum
		}
	}

	for i := range cpu {
		if math.Abs(float64(gpu[i]-cpu[i])) > 0.1 {
			t.Fatalf("large matmul[%d]: gpu=%v cpu=%v", i, gpu[i], cpu[i])
		}
	}
}

func TestDenseMatmulTransBGPULarger(t *testing.T) {
	m, k, n := 32, 256, 128
	a := make([]float32, m*k)
	b := make([]float32, n*k) // B is (n x k)
	for i := range a {
		a[i] = float32(i%11) * 0.01
	}
	for i := range b {
		b[i] = float32(i%7) * 0.01
	}

	gpu := make([]float32, m*n)
	if err := DenseMatmulTransB(gpu, a, b, m, k, n); err != nil {
		t.Fatalf("DenseMatmulTransB large: %v", err)
	}

	// C = A * B^T: A is (m x k), B is (n x k), C is (m x n)
	cpu := make([]float32, m*n)
	for i := range m {
		for j := range n {
			var sum float32
			for p := range k {
				sum += a[i*k+p] * b[j*k+p]
			}
			cpu[i*n+j] = sum
		}
	}

	for i := range cpu {
		if math.Abs(float64(gpu[i]-cpu[i])) > 0.1 {
			t.Fatalf("large matmulTransB[%d]: gpu=%v cpu=%v", i, gpu[i], cpu[i])
		}
	}
}
