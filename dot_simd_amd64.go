//go:build goexperiment.simd && amd64

package turboquant

import (
	"simd/archsimd"
	"unsafe"
)

// DotFloat32s computes the dot product of a and b using the experimental
// simd package's AVX2 8-wide float32 vectors. The default (non-experiment)
// build uses the hand-written SSE kernel instead; see dot_dispatch_amd64.go.
func DotFloat32s(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("turboquant: DotFloat32s length mismatch")
	}
	if len(a) == 0 {
		return 0
	}
	return dotFloat32sSIMD(unsafe.SliceData(a), unsafe.SliceData(b), len(a))
}

// float32x8At returns an unchecked *[8]float32 view of p starting at element
// offset i. The caller is responsible for bounds safety, mirroring the raw
// pointer arithmetic the hand-written .s kernels use to avoid Go's
// slice-to-array-pointer conversion performing a redundant length check on
// every load (LoadFloat32x8Slice does this check internally).
func float32x8At(p *float32, i int) *[8]float32 {
	return (*[8]float32)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(i)*4))
}

func float32At(p *float32, i int) float32 {
	return *(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(i)*4))
}

// hsumFloat32x8 horizontally reduces an 8-wide vector to a scalar. The simd
// package exposes no reduce/horizontal-sum helper (see report), so this
// composes GetLo/GetHi/Add with four GetElem extracts; the extraction cost is
// O(1) per call, amortized over the O(n) main loop.
func hsumFloat32x8(v archsimd.Float32x8) float32 {
	s := v.GetLo().Add(v.GetHi())
	return s.GetElem(0) + s.GetElem(1) + s.GetElem(2) + s.GetElem(3)
}

// dotFloat32sSIMD mirrors dotFloat32sSSE's signature and unroll shape (four
// independent accumulators to hide FMA latency) but uses AVX2 8-wide FMA
// instead of 4-wide SSE MULPS/ADDPS.
func dotFloat32sSIMD(a, b *float32, n int) float32 {
	var acc0, acc1, acc2, acc3 archsimd.Float32x8
	i := 0
	for ; i+32 <= n; i += 32 {
		acc0 = archsimd.LoadFloat32x8(float32x8At(a, i)).MulAdd(archsimd.LoadFloat32x8(float32x8At(b, i)), acc0)
		acc1 = archsimd.LoadFloat32x8(float32x8At(a, i+8)).MulAdd(archsimd.LoadFloat32x8(float32x8At(b, i+8)), acc1)
		acc2 = archsimd.LoadFloat32x8(float32x8At(a, i+16)).MulAdd(archsimd.LoadFloat32x8(float32x8At(b, i+16)), acc2)
		acc3 = archsimd.LoadFloat32x8(float32x8At(a, i+24)).MulAdd(archsimd.LoadFloat32x8(float32x8At(b, i+24)), acc3)
	}
	for ; i+8 <= n; i += 8 {
		acc0 = archsimd.LoadFloat32x8(float32x8At(a, i)).MulAdd(archsimd.LoadFloat32x8(float32x8At(b, i)), acc0)
	}
	sum := hsumFloat32x8(acc0.Add(acc1)) + hsumFloat32x8(acc2.Add(acc3))
	for ; i < n; i++ {
		sum += float32At(a, i) * float32At(b, i)
	}
	return sum
}
