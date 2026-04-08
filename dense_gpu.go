//go:build linux && amd64 && cgo && cuda

package turboquant

import (
	"fmt"
	"sync"

	"github.com/odvcencio/turboquant/internal/cudaruntime"
)

// DenseLayerSpec describes one weight matrix to upload to GPU.
type DenseLayerSpec struct {
	W    []float32 // weight data (rows x cols), copied to GPU at init
	Rows int
	Cols int
}

// gpuWeight holds a single weight matrix on the device.
type gpuWeight struct {
	dW   cudaruntime.DevicePtr
	rows int
	cols int
}

// GPUDenseContext holds GPU-resident weight matrices and reusable activation
// buffers. Weight matrices are uploaded once at construction time and reused
// across many Matmul calls. Only activation data (src/dst) transfers per call.
type GPUDenseContext struct {
	mu      sync.Mutex
	runtime *cudaruntime.Runtime
	weights []gpuWeight
	buffers map[int]cudaruntime.DevicePtr // reusable device buffers keyed by byte size
	closed  bool
}

// NewGPUDenseContext uploads weight matrices to GPU and returns a context that
// can perform repeated matmul operations without re-uploading weights.
func NewGPUDenseContext(specs []DenseLayerSpec) (*GPUDenseContext, error) {
	if len(specs) == 0 {
		return nil, fmt.Errorf("turboquant: NewGPUDenseContext requires at least one DenseLayerSpec")
	}
	rt, err := cudaruntime.New()
	if err != nil {
		return nil, fmt.Errorf("turboquant: %w", err)
	}
	ctx := &GPUDenseContext{
		runtime: rt,
		weights: make([]gpuWeight, len(specs)),
		buffers: make(map[int]cudaruntime.DevicePtr),
	}
	for i, spec := range specs {
		if len(spec.W) < spec.Rows*spec.Cols {
			_ = ctx.Close()
			return nil, fmt.Errorf("turboquant: DenseLayerSpec[%d]: W length %d < rows*cols %d", i, len(spec.W), spec.Rows*spec.Cols)
		}
		size := spec.Rows * spec.Cols * 4
		dW, err := rt.Alloc(size)
		if err != nil {
			_ = ctx.Close()
			return nil, fmt.Errorf("turboquant: alloc weight[%d]: %w", i, err)
		}
		if err := rt.HtoDFloat32(dW, spec.W[:spec.Rows*spec.Cols]); err != nil {
			_ = rt.Free(dW)
			_ = ctx.Close()
			return nil, fmt.Errorf("turboquant: upload weight[%d]: %w", i, err)
		}
		ctx.weights[i] = gpuWeight{dW: dW, rows: spec.Rows, cols: spec.Cols}
	}
	return ctx, nil
}

// getBuffer returns a device buffer of at least the given byte size. If a
// buffer of exactly that size exists in the pool it is reused; otherwise a new
// one is allocated.
func (ctx *GPUDenseContext) getBuffer(bytes int) (cudaruntime.DevicePtr, error) {
	if ptr, ok := ctx.buffers[bytes]; ok {
		return ptr, nil
	}
	ptr, err := ctx.runtime.Alloc(bytes)
	if err != nil {
		return 0, fmt.Errorf("turboquant: alloc buffer (%d bytes): %w", bytes, err)
	}
	ctx.buffers[bytes] = ptr
	return ptr, nil
}

// Matmul computes dst = src * weights[weightIdx]^T on GPU.
// src is (m x k) on host, weight is (n x k) on GPU, dst is (m x n) on host.
// Only src and dst transfer per call. The weight stays on GPU.
func (ctx *GPUDenseContext) Matmul(dst, src []float32, m, k, n, weightIdx int) error {
	if ctx == nil {
		return fmt.Errorf("turboquant: nil GPUDenseContext")
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.closed {
		return fmt.Errorf("turboquant: GPUDenseContext is closed")
	}
	if weightIdx < 0 || weightIdx >= len(ctx.weights) {
		return fmt.Errorf("turboquant: weightIdx %d out of range [0, %d)", weightIdx, len(ctx.weights))
	}
	w := &ctx.weights[weightIdx]
	if w.rows != n || w.cols != k {
		return fmt.Errorf("turboquant: Matmul weight[%d] dimensions (%d x %d) incompatible with n=%d k=%d", weightIdx, w.rows, w.cols, n, k)
	}
	if len(src) < m*k {
		return fmt.Errorf("turboquant: src too small: need %d, got %d", m*k, len(src))
	}
	if len(dst) < m*n {
		return fmt.Errorf("turboquant: dst too small: need %d, got %d", m*n, len(dst))
	}

	srcBytes := m * k * 4
	dstBytes := m * n * 4

	dSrc, err := ctx.getBuffer(srcBytes)
	if err != nil {
		return err
	}
	dDst, err := ctx.getBuffer(dstBytes)
	if err != nil {
		return err
	}

	if err := ctx.runtime.HtoDFloat32(dSrc, src[:m*k]); err != nil {
		return fmt.Errorf("turboquant: HtoD src: %w", err)
	}
	if err := ctx.runtime.LaunchSgemmTransB(dSrc, w.dW, dDst, m, k, n); err != nil {
		return fmt.Errorf("turboquant: sgemm transB: %w", err)
	}
	if err := ctx.runtime.DtoHFloat32(dst[:m*n], dDst); err != nil {
		return fmt.Errorf("turboquant: DtoH dst: %w", err)
	}
	return nil
}

// MatmulAB computes dst = src * weights[weightIdx] on GPU (no transpose).
// src is (m x k) on host, weight is (k x n) on GPU, dst is (m x n) on host.
func (ctx *GPUDenseContext) MatmulAB(dst, src []float32, m, k, n, weightIdx int) error {
	if ctx == nil {
		return fmt.Errorf("turboquant: nil GPUDenseContext")
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.closed {
		return fmt.Errorf("turboquant: GPUDenseContext is closed")
	}
	if weightIdx < 0 || weightIdx >= len(ctx.weights) {
		return fmt.Errorf("turboquant: weightIdx %d out of range [0, %d)", weightIdx, len(ctx.weights))
	}
	w := &ctx.weights[weightIdx]
	if w.rows != k || w.cols != n {
		return fmt.Errorf("turboquant: MatmulAB weight[%d] dimensions (%d x %d) incompatible with k=%d n=%d", weightIdx, w.rows, w.cols, k, n)
	}
	if len(src) < m*k {
		return fmt.Errorf("turboquant: src too small: need %d, got %d", m*k, len(src))
	}
	if len(dst) < m*n {
		return fmt.Errorf("turboquant: dst too small: need %d, got %d", m*n, len(dst))
	}

	srcBytes := m * k * 4
	dstBytes := m * n * 4

	dSrc, err := ctx.getBuffer(srcBytes)
	if err != nil {
		return err
	}
	dDst, err := ctx.getBuffer(dstBytes)
	if err != nil {
		return err
	}

	if err := ctx.runtime.HtoDFloat32(dSrc, src[:m*k]); err != nil {
		return fmt.Errorf("turboquant: HtoD src: %w", err)
	}
	if err := ctx.runtime.LaunchSgemm(dSrc, w.dW, dDst, m, k, n); err != nil {
		return fmt.Errorf("turboquant: sgemm: %w", err)
	}
	if err := ctx.runtime.DtoHFloat32(dst[:m*n], dDst); err != nil {
		return fmt.Errorf("turboquant: DtoH dst: %w", err)
	}
	return nil
}

// UpdateWeight copies new weight data from host to the existing GPU buffer.
// The dimensions must match the original spec (no realloc).
func (ctx *GPUDenseContext) UpdateWeight(weightIdx int, w []float32) error {
	if ctx == nil {
		return fmt.Errorf("turboquant: nil GPUDenseContext")
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.closed {
		return fmt.Errorf("turboquant: GPUDenseContext is closed")
	}
	if weightIdx < 0 || weightIdx >= len(ctx.weights) {
		return fmt.Errorf("turboquant: weightIdx %d out of range [0, %d)", weightIdx, len(ctx.weights))
	}
	gw := &ctx.weights[weightIdx]
	need := gw.rows * gw.cols
	if len(w) < need {
		return fmt.Errorf("turboquant: UpdateWeight data too small: need %d, got %d", need, len(w))
	}
	if err := ctx.runtime.HtoDFloat32(gw.dW, w[:need]); err != nil {
		return fmt.Errorf("turboquant: UpdateWeight HtoD: %w", err)
	}
	return nil
}

// Close frees all GPU memory (weights and buffers) and destroys the runtime.
func (ctx *GPUDenseContext) Close() error {
	if ctx == nil {
		return nil
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.closed {
		return nil
	}
	ctx.closed = true
	for i := range ctx.weights {
		if ctx.weights[i].dW != 0 {
			_ = ctx.runtime.Free(ctx.weights[i].dW)
			ctx.weights[i].dW = 0
		}
	}
	for size, ptr := range ctx.buffers {
		if ptr != 0 {
			_ = ctx.runtime.Free(ptr)
		}
		delete(ctx.buffers, size)
	}
	if ctx.runtime != nil {
		ctx.runtime.Close()
		ctx.runtime = nil
	}
	return nil
}
