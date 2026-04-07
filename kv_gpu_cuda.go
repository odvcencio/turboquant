//go:build linux && amd64 && cgo && cuda

package turboquant

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/odvcencio/turboquant/internal/cudaruntime"
)

type kvGPUValueBackendCUDA struct {
	mu sync.Mutex

	runtime *cudaruntime.Runtime

	dim         int
	bitWidth    int
	count       int
	packedBytes int

	dPacked    cudaruntime.DevicePtr
	dNorms     cudaruntime.DevicePtr
	dCentroids cudaruntime.DevicePtr
	dIndices   cudaruntime.DevicePtr
	dWeights   cudaruntime.DevicePtr
	dRotated   cudaruntime.DevicePtr

	indexBytes   int
	weightBytes  int
	rotatedBytes int

	hostIndices []uint32
	hostWeights []float32
	hostRotated []float32

	hostIndexPtr   unsafe.Pointer
	hostWeightPtr  unsafe.Pointer
	hostRotatedPtr unsafe.Pointer
	hostRotCap     int

	closed bool
}

func newKVGPUValueBackend(p *KVCachePage) (kvGPUValueBackend, error) {
	if p == nil || p.length == 0 {
		return nil, nil
	}
	rt, err := cudaruntime.New()
	if err != nil {
		return nil, fmt.Errorf("turboquant: %w", err)
	}
	b := &kvGPUValueBackendCUDA{
		runtime:     rt,
		dim:         p.valueQ.dim,
		bitWidth:    p.valueQ.bitWidth,
		count:       p.length,
		packedBytes: p.valueBytes,
	}
	if err := b.allocAndCopy(&b.dPacked, p.valuePacked[:p.length*p.valueBytes]); err != nil {
		b.Close()
		return nil, err
	}
	if err := b.allocAndCopy(&b.dNorms, encodeFloat32s(nil, p.valueNorms[:p.length])); err != nil {
		b.Close()
		return nil, err
	}
	if err := b.allocAndCopy(&b.dCentroids, encodeFloat32s(nil, p.valueQ.cb.centroids)); err != nil {
		b.Close()
		return nil, err
	}
	if err := b.ensureRotatedCapacity(1); err != nil {
		b.Close()
		return nil, err
	}
	return b, nil
}

func (b *kvGPUValueBackendCUDA) accumulateRotatedTo(dst []float32, indices []uint32, weights []float32) error {
	if b == nil {
		return ErrGPUBackendUnavailable
	}
	if len(dst) != b.dim {
		return fmt.Errorf("turboquant: expected rotated value destination length %d, got %d", b.dim, len(dst))
	}
	if len(indices) != len(weights) {
		return fmt.Errorf("turboquant: expected matching GPU value index/weight lengths, got %d and %d", len(indices), len(weights))
	}
	if len(indices) == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return fmt.Errorf("turboquant: GPU value backend is closed")
	}
	for _, idx := range indices {
		if int(idx) < 0 || int(idx) >= b.count {
			return fmt.Errorf("turboquant: GPU value index %d out of bounds for corpus size %d", idx, b.count)
		}
	}
	if err := b.ensureUploadCapacity(len(indices)); err != nil {
		return err
	}
	if err := b.ensureRotatedCapacity(1); err != nil {
		return err
	}
	copy(b.hostIndices[:len(indices)], indices)
	copy(b.hostWeights[:len(weights)], weights)
	if err := b.ensureHostRotatedCapacity(1); err != nil {
		return err
	}
	if err := b.runtime.LaunchValueSumToHost(b.hostRotated[:b.dim], b.dPacked, b.dNorms, b.dCentroids, b.dIndices, b.dWeights, b.dRotated, b.hostIndices[:len(indices)], b.hostWeights[:len(weights)], b.dim, b.bitWidth, b.packedBytes, len(indices)); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	copy(dst, b.hostRotated[:b.dim])
	return nil
}

func (b *kvGPUValueBackendCUDA) accumulateRotatedBatchTo(dst []float32, indices []uint32, weights []float32, queryCount int) error {
	if b == nil {
		return ErrGPUBackendUnavailable
	}
	if queryCount <= 0 {
		return fmt.Errorf("turboquant: expected positive GPU value query count, got %d", queryCount)
	}
	if queryCount == 1 {
		return b.accumulateRotatedTo(dst, indices, weights)
	}
	if len(dst) != queryCount*b.dim {
		return fmt.Errorf("turboquant: expected rotated value destination length %d, got %d", queryCount*b.dim, len(dst))
	}
	if len(indices) != len(weights) {
		return fmt.Errorf("turboquant: expected matching GPU value index/weight lengths, got %d and %d", len(indices), len(weights))
	}
	if len(indices)%queryCount != 0 {
		return fmt.Errorf("turboquant: expected flattened GPU value buffers divisible by query count %d, got %d", queryCount, len(indices))
	}
	if len(indices) == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return nil
	}
	k := len(indices) / queryCount
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return fmt.Errorf("turboquant: GPU value backend is closed")
	}
	for _, idx := range indices {
		if int(idx) < 0 || int(idx) >= b.count {
			return fmt.Errorf("turboquant: GPU value index %d out of bounds for corpus size %d", idx, b.count)
		}
	}
	if err := b.ensureUploadCapacity(len(indices)); err != nil {
		return err
	}
	if err := b.ensureRotatedCapacity(queryCount); err != nil {
		return err
	}
	copy(b.hostIndices[:len(indices)], indices)
	copy(b.hostWeights[:len(weights)], weights)
	if err := b.ensureHostRotatedCapacity(queryCount); err != nil {
		return err
	}
	if err := b.runtime.LaunchValueSumBatchToHost(b.hostRotated[:len(dst)], b.dPacked, b.dNorms, b.dCentroids, b.dIndices, b.dWeights, b.dRotated, b.hostIndices[:len(indices)], b.hostWeights[:len(weights)], b.dim, b.bitWidth, b.packedBytes, k, queryCount); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	copy(dst, b.hostRotated[:len(dst)])
	return nil
}

func (b *kvGPUValueBackendCUDA) ensureUploadCapacity(k int) error {
	wantBytes := k * 4
	if wantBytes <= b.indexBytes {
		return nil
	}
	if err := b.freeDevice(b.dIndices); err != nil {
		return err
	}
	if err := b.freeDevice(b.dWeights); err != nil {
		return err
	}
	if err := b.freeHost(b.hostIndexPtr); err != nil {
		return err
	}
	if err := b.freeHost(b.hostWeightPtr); err != nil {
		return err
	}
	ptr, err := b.runtime.Alloc(wantBytes)
	if err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	b.dIndices = ptr
	ptr, err = b.runtime.Alloc(wantBytes)
	if err != nil {
		_ = b.freeDevice(b.dIndices)
		b.dIndices = 0
		return fmt.Errorf("turboquant: %w", err)
	}
	b.dWeights = ptr
	hostPtr, err := b.runtime.AllocHost(wantBytes)
	if err != nil {
		_ = b.freeDevice(b.dIndices)
		_ = b.freeDevice(b.dWeights)
		b.dIndices = 0
		b.dWeights = 0
		return fmt.Errorf("turboquant: %w", err)
	}
	b.hostIndexPtr = hostPtr
	hostPtr, err = b.runtime.AllocHost(wantBytes)
	if err != nil {
		_ = b.freeHost(b.hostIndexPtr)
		_ = b.freeDevice(b.dIndices)
		_ = b.freeDevice(b.dWeights)
		b.hostIndexPtr = nil
		b.dIndices = 0
		b.dWeights = 0
		return fmt.Errorf("turboquant: %w", err)
	}
	b.hostWeightPtr = hostPtr
	b.indexBytes = wantBytes
	b.weightBytes = wantBytes
	b.hostIndices = unsafe.Slice((*uint32)(b.hostIndexPtr), k)
	b.hostWeights = unsafe.Slice((*float32)(b.hostWeightPtr), k)
	return nil
}

func (b *kvGPUValueBackendCUDA) ensureHostRotatedCapacity(queryCount int) error {
	want := queryCount * b.dim
	if want <= b.hostRotCap {
		b.hostRotated = b.hostRotated[:want]
		return nil
	}
	if err := b.freeHost(b.hostRotatedPtr); err != nil {
		return err
	}
	ptr, err := b.runtime.AllocHost(want * 4)
	if err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	b.hostRotatedPtr = ptr
	b.hostRotCap = want
	b.hostRotated = unsafe.Slice((*float32)(ptr), want)
	return nil
}

func (b *kvGPUValueBackendCUDA) ensureRotatedCapacity(queryCount int) error {
	wantBytes := queryCount * b.dim * 4
	if wantBytes <= b.rotatedBytes {
		return nil
	}
	if err := b.freeDevice(b.dRotated); err != nil {
		return err
	}
	ptr, err := b.runtime.Alloc(wantBytes)
	if err != nil {
		b.dRotated = 0
		b.rotatedBytes = 0
		return fmt.Errorf("turboquant: %w", err)
	}
	b.dRotated = ptr
	b.rotatedBytes = wantBytes
	return nil
}

func (b *kvGPUValueBackendCUDA) allocAndCopy(dst *cudaruntime.DevicePtr, data []byte) error {
	ptr, err := b.runtime.Alloc(len(data))
	if err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	if len(data) != 0 {
		if err := b.runtime.HtoD(ptr, data); err != nil {
			_ = b.runtime.Free(ptr)
			return fmt.Errorf("turboquant: %w", err)
		}
	}
	*dst = ptr
	return nil
}

func (b *kvGPUValueBackendCUDA) freeDevice(ptr cudaruntime.DevicePtr) error {
	if b.runtime == nil || ptr == 0 {
		return nil
	}
	if err := b.runtime.Free(ptr); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	return nil
}

func (b *kvGPUValueBackendCUDA) freeHost(ptr unsafe.Pointer) error {
	if b.runtime == nil || ptr == nil {
		return nil
	}
	if err := b.runtime.FreeHost(ptr); err != nil {
		return fmt.Errorf("turboquant: %w", err)
	}
	return nil
}

func (b *kvGPUValueBackendCUDA) Close() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return nil
	}
	b.closed = true
	_ = b.freeDevice(b.dPacked)
	_ = b.freeDevice(b.dNorms)
	_ = b.freeDevice(b.dCentroids)
	_ = b.freeDevice(b.dIndices)
	_ = b.freeDevice(b.dWeights)
	_ = b.freeDevice(b.dRotated)
	_ = b.freeHost(b.hostIndexPtr)
	_ = b.freeHost(b.hostWeightPtr)
	_ = b.freeHost(b.hostRotatedPtr)
	b.dPacked = 0
	b.dNorms = 0
	b.dCentroids = 0
	b.dIndices = 0
	b.dWeights = 0
	b.dRotated = 0
	b.hostIndexPtr = nil
	b.hostWeightPtr = nil
	b.hostRotatedPtr = nil
	if b.runtime != nil {
		b.runtime.Close()
		b.runtime = nil
	}
	return nil
}
