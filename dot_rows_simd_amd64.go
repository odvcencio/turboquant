//go:build goexperiment.simd && amd64

package turboquant

import (
	"simd/archsimd"
	"unsafe"
)

// dotFloat32Rows4 computes 4 row dot products against vec using AVX2 8-wide
// FMA. The default (non-experiment) build uses the hand-written SSE kernel
// instead; see dot_rows_dispatch_amd64.go.
func dotFloat32Rows4(dst *[4]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 4*n {
		panic("turboquant: dotFloat32Rows4 row block too short")
	}
	if n == 0 {
		dst[0], dst[1], dst[2], dst[3] = 0, 0, 0, 0
		return
	}
	dotFloat32Rows4SIMD(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &dst[0])
}

func dotFloat32Rows8(dst *[8]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 8*n {
		panic("turboquant: dotFloat32Rows8 row block too short")
	}
	if n == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	dotFloat32Rows8SIMD(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &dst[0])
}

func dotFloat32Rows8Blocked(dst *[8]float32, rows, vec []float32) {
	n := len(vec)
	if len(rows) < 8*n {
		panic("turboquant: dotFloat32Rows8Blocked row block too short")
	}
	if n == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	dotFloat32Rows8BlockedSIMD(unsafe.SliceData(rows), unsafe.SliceData(vec), n, &dst[0])
}

// float32PtrAt returns an unchecked *float32 view of p advanced by i
// elements, used to compute per-row base pointers into a flat row-major
// buffer without incurring a slice bounds check per row.
func float32PtrAt(p *float32, i int) *float32 {
	return (*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(i)*4))
}

func float32x4At(p *float32, i int) *[4]float32 {
	return (*[4]float32)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(i)*4))
}

func float32ArrayView4(dst *float32) *[4]float32 {
	return (*[4]float32)(unsafe.Pointer(dst))
}

func float32ArrayView8(dst *float32) *[8]float32 {
	return (*[8]float32)(unsafe.Pointer(dst))
}

func hsumFloat32x4(v archsimd.Float32x4) float32 {
	return v.GetElem(0) + v.GetElem(1) + v.GetElem(2) + v.GetElem(3)
}

// dotFloat32Rows4SIMD mirrors dotFloat32Rows4SSE's shape: one accumulator per
// row, the shared vec vector loaded once per chunk and reused across rows.
func dotFloat32Rows4SIMD(rowsPtr, vecPtr *float32, n int, dst *float32) {
	r0 := rowsPtr
	r1 := float32PtrAt(rowsPtr, n)
	r2 := float32PtrAt(rowsPtr, 2*n)
	r3 := float32PtrAt(rowsPtr, 3*n)

	var acc0, acc1, acc2, acc3 archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8(float32x8At(vecPtr, i))
		acc0 = archsimd.LoadFloat32x8(float32x8At(r0, i)).MulAdd(v, acc0)
		acc1 = archsimd.LoadFloat32x8(float32x8At(r1, i)).MulAdd(v, acc1)
		acc2 = archsimd.LoadFloat32x8(float32x8At(r2, i)).MulAdd(v, acc2)
		acc3 = archsimd.LoadFloat32x8(float32x8At(r3, i)).MulAdd(v, acc3)
	}
	out := float32ArrayView4(dst)
	out[0] = hsumFloat32x8(acc0)
	out[1] = hsumFloat32x8(acc1)
	out[2] = hsumFloat32x8(acc2)
	out[3] = hsumFloat32x8(acc3)
	for ; i < n; i++ {
		vv := float32At(vecPtr, i)
		out[0] += float32At(r0, i) * vv
		out[1] += float32At(r1, i) * vv
		out[2] += float32At(r2, i) * vv
		out[3] += float32At(r3, i) * vv
	}
}

// dotFloat32Rows8SIMD mirrors dotFloat32Rows8SSE's shape: eight independent
// accumulators, one per row, all fed from a single shared vec load per
// 8-wide chunk.
func dotFloat32Rows8SIMD(rowsPtr, vecPtr *float32, n int, dst *float32) {
	r0 := rowsPtr
	r1 := float32PtrAt(rowsPtr, n)
	r2 := float32PtrAt(rowsPtr, 2*n)
	r3 := float32PtrAt(rowsPtr, 3*n)
	r4 := float32PtrAt(rowsPtr, 4*n)
	r5 := float32PtrAt(rowsPtr, 5*n)
	r6 := float32PtrAt(rowsPtr, 6*n)
	r7 := float32PtrAt(rowsPtr, 7*n)

	var acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7 archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8(float32x8At(vecPtr, i))
		acc0 = archsimd.LoadFloat32x8(float32x8At(r0, i)).MulAdd(v, acc0)
		acc1 = archsimd.LoadFloat32x8(float32x8At(r1, i)).MulAdd(v, acc1)
		acc2 = archsimd.LoadFloat32x8(float32x8At(r2, i)).MulAdd(v, acc2)
		acc3 = archsimd.LoadFloat32x8(float32x8At(r3, i)).MulAdd(v, acc3)
		acc4 = archsimd.LoadFloat32x8(float32x8At(r4, i)).MulAdd(v, acc4)
		acc5 = archsimd.LoadFloat32x8(float32x8At(r5, i)).MulAdd(v, acc5)
		acc6 = archsimd.LoadFloat32x8(float32x8At(r6, i)).MulAdd(v, acc6)
		acc7 = archsimd.LoadFloat32x8(float32x8At(r7, i)).MulAdd(v, acc7)
	}
	out := float32ArrayView8(dst)
	out[0] = hsumFloat32x8(acc0)
	out[1] = hsumFloat32x8(acc1)
	out[2] = hsumFloat32x8(acc2)
	out[3] = hsumFloat32x8(acc3)
	out[4] = hsumFloat32x8(acc4)
	out[5] = hsumFloat32x8(acc5)
	out[6] = hsumFloat32x8(acc6)
	out[7] = hsumFloat32x8(acc7)
	for ; i < n; i++ {
		vv := float32At(vecPtr, i)
		out[0] += float32At(r0, i) * vv
		out[1] += float32At(r1, i) * vv
		out[2] += float32At(r2, i) * vv
		out[3] += float32At(r3, i) * vv
		out[4] += float32At(r4, i) * vv
		out[5] += float32At(r5, i) * vv
		out[6] += float32At(r6, i) * vv
		out[7] += float32At(r7, i) * vv
	}
}

// dotFloat32Rows8BlockedSIMD consumes the same 4-column-interleaved buffer
// that blockProjectionRowGroup8 produces (see qjl.go): for a fixed 4-column
// group, all 8 rows' 4-float chunks sit contiguously (32 bytes per group).
// That grouping predates this file and was sized for 4-wide SSE loads, one
// row per load. Rather than reshaping the buffer for 8-wide loads (which
// would require an 8-column blocking scheme, a change to the buffer's
// producer, and would ripple into IPQuantizer's proj8 field — out of scope
// for this spike; see the SIMD spike report), this kernel instead pairs
// adjacent rows: row r and row r+1 occupy consecutive 4-float slots within
// the same 32-byte group, so a single 8-wide load spans both rows' chunks
// for the same 4 columns. The shared vec chunk is duplicated into both
// 128-bit halves so one FMA advances both rows' accumulators at once. This
// reaches genuine AVX2 8-lane FMA utilization without touching the buffer
// layout or its producer.
func dotFloat32Rows8BlockedSIMD(blockPtr, vecPtr *float32, n int, dst *float32) {
	const groupStride = 32 // 8 rows * 4 floats
	nGroups := n / 4

	var acc01, acc23, acc45, acc67 archsimd.Float32x8
	for g := 0; g < nGroups; g++ {
		base := g * groupStride
		vLo := archsimd.LoadFloat32x4(float32x4At(vecPtr, g*4))
		vDup := archsimd.Float32x8{}.SetLo(vLo).SetHi(vLo)
		acc01 = archsimd.LoadFloat32x8(float32x8At(blockPtr, base)).MulAdd(vDup, acc01)
		acc23 = archsimd.LoadFloat32x8(float32x8At(blockPtr, base+8)).MulAdd(vDup, acc23)
		acc45 = archsimd.LoadFloat32x8(float32x8At(blockPtr, base+16)).MulAdd(vDup, acc45)
		acc67 = archsimd.LoadFloat32x8(float32x8At(blockPtr, base+24)).MulAdd(vDup, acc67)
	}

	out := float32ArrayView8(dst)
	out[0], out[1] = hsumFloat32x4(acc01.GetLo()), hsumFloat32x4(acc01.GetHi())
	out[2], out[3] = hsumFloat32x4(acc23.GetLo()), hsumFloat32x4(acc23.GetHi())
	out[4], out[5] = hsumFloat32x4(acc45.GetLo()), hsumFloat32x4(acc45.GetHi())
	out[6], out[7] = hsumFloat32x4(acc67.GetLo()), hsumFloat32x4(acc67.GetHi())

	// Defensive remainder, unreachable via the current call graph: proj8 is
	// only ever built by blockProjectionRows8, which refuses (returns nil)
	// unless dim % 8 == 0, so n % 4 == 0 always holds for every real caller.
	// Kept for robustness against a future, differently-shaped caller, using
	// the same one-group-slot-per-column addressing as dotFloat32Rows8BlockedSSE's
	// scalar remainder.
	for col := nGroups * 4; col < n; col++ {
		slot := nGroups + (col - nGroups*4)
		base := slot * groupStride
		vv := float32At(vecPtr, col)
		for r := 0; r < 8; r++ {
			out[r] += float32At(blockPtr, base+r*4) * vv
		}
	}
}
