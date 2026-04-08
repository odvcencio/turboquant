//go:build linux && amd64 && cgo && cuda

package turboquant

import (
	"math"
	"testing"
)

func TestGPUDenseContextMatmul(t *testing.T) {
	// W is 3x4 (n=3, k=4), src is 2x4 (m=2, k=4), dst = src * W^T = 2x3
	w := []float32{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
	}
	ctx, err := NewGPUDenseContext([]DenseLayerSpec{{W: w, Rows: 3, Cols: 4}})
	if err != nil {
		t.Fatalf("NewGPUDenseContext: %v", err)
	}
	defer ctx.Close()

	src := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	dst := make([]float32, 6)
	if err := ctx.Matmul(dst, src, 2, 4, 3, 0); err != nil {
		t.Fatalf("Matmul: %v", err)
	}
	// Row 0: [1,2,3,4] * [1,0,0,0]^T=1, *[0,1,0,0]^T=2, *[0,0,1,0]^T=3
	expect := []float32{1, 2, 3, 5, 6, 7}
	for i, v := range expect {
		if math.Abs(float64(dst[i]-v)) > 1e-3 {
			t.Fatalf("dst[%d]: got %v want %v", i, dst[i], v)
		}
	}
}

func TestGPUDenseContextRepeatedCalls(t *testing.T) {
	// Verify buffer reuse works across multiple calls
	w := make([]float32, 384*384)
	for i := range w {
		w[i] = float32(i%17) * 0.001
	}
	ctx, err := NewGPUDenseContext([]DenseLayerSpec{{W: w, Rows: 384, Cols: 384}})
	if err != nil {
		t.Fatalf("NewGPUDenseContext: %v", err)
	}
	defer ctx.Close()

	src := make([]float32, 8*384)
	for i := range src {
		src[i] = float32(i%13) * 0.01
	}
	dst := make([]float32, 8*384)

	// Call 10 times - should reuse buffers
	for call := range 10 {
		if err := ctx.Matmul(dst, src, 8, 384, 384, 0); err != nil {
			t.Fatalf("call %d: %v", call, err)
		}
	}

	// Verify against CPU
	cpu := make([]float32, 8*384)
	for i := range 8 {
		for j := range 384 {
			var sum float32
			for p := range 384 {
				sum += src[i*384+p] * w[j*384+p]
			}
			cpu[i*384+j] = sum
		}
	}
	for i := range cpu {
		if math.Abs(float64(dst[i]-cpu[i])) > 0.5 {
			t.Fatalf("repeated[%d]: gpu=%v cpu=%v", i, dst[i], cpu[i])
		}
	}
}

func TestGPUDenseContextUpdateWeight(t *testing.T) {
	w := []float32{1, 0, 0, 1}
	ctx, err := NewGPUDenseContext([]DenseLayerSpec{{W: w, Rows: 2, Cols: 2}})
	if err != nil {
		t.Fatalf("NewGPUDenseContext: %v", err)
	}
	defer ctx.Close()

	src := []float32{3, 7}
	dst := make([]float32, 2)
	if err := ctx.Matmul(dst, src, 1, 2, 2, 0); err != nil {
		t.Fatalf("Matmul before update: %v", err)
	}
	if math.Abs(float64(dst[0]-3)) > 1e-3 || math.Abs(float64(dst[1]-7)) > 1e-3 {
		t.Fatalf("before update: got %v", dst)
	}

	// Update weight to swap columns
	newW := []float32{0, 1, 1, 0}
	if err := ctx.UpdateWeight(0, newW); err != nil {
		t.Fatalf("UpdateWeight: %v", err)
	}
	if err := ctx.Matmul(dst, src, 1, 2, 2, 0); err != nil {
		t.Fatalf("Matmul after update: %v", err)
	}
	if math.Abs(float64(dst[0]-7)) > 1e-3 || math.Abs(float64(dst[1]-3)) > 1e-3 {
		t.Fatalf("after update: got %v", dst)
	}
}

func TestGPUDenseContextMultipleWeights(t *testing.T) {
	// Two weight matrices, used alternately
	w1 := []float32{1, 0, 0, 1} // identity 2x2
	w2 := []float32{2, 0, 0, 2} // scale by 2
	ctx, err := NewGPUDenseContext([]DenseLayerSpec{
		{W: w1, Rows: 2, Cols: 2},
		{W: w2, Rows: 2, Cols: 2},
	})
	if err != nil {
		t.Fatalf("NewGPUDenseContext: %v", err)
	}
	defer ctx.Close()

	src := []float32{3, 5}
	dst := make([]float32, 2)

	if err := ctx.Matmul(dst, src, 1, 2, 2, 0); err != nil {
		t.Fatalf("Matmul weight 0: %v", err)
	}
	if math.Abs(float64(dst[0]-3)) > 1e-3 {
		t.Fatalf("weight 0: got %v", dst)
	}

	if err := ctx.Matmul(dst, src, 1, 2, 2, 1); err != nil {
		t.Fatalf("Matmul weight 1: %v", err)
	}
	if math.Abs(float64(dst[0]-6)) > 1e-3 {
		t.Fatalf("weight 1: got %v", dst)
	}
}

func TestGPUDenseContextMatmulAB(t *testing.T) {
	// W is 3x2 (k=3, n=2), src is 2x3 (m=2, k=3), dst = src * W = 2x2
	w := []float32{
		1, 2,
		3, 4,
		5, 6,
	}
	ctx, err := NewGPUDenseContext([]DenseLayerSpec{{W: w, Rows: 3, Cols: 2}})
	if err != nil {
		t.Fatalf("NewGPUDenseContext: %v", err)
	}
	defer ctx.Close()

	src := []float32{1, 0, 0, 0, 1, 0}
	dst := make([]float32, 4)
	if err := ctx.MatmulAB(dst, src, 2, 3, 2, 0); err != nil {
		t.Fatalf("MatmulAB: %v", err)
	}
	// Row 0: [1,0,0] * W = [1, 2]
	// Row 1: [0,1,0] * W = [3, 4]
	expect := []float32{1, 2, 3, 4}
	for i, v := range expect {
		if math.Abs(float64(dst[i]-v)) > 1e-3 {
			t.Fatalf("dst[%d]: got %v want %v", i, dst[i], v)
		}
	}
}
