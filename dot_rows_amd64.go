//go:build amd64

package turboquant

// dotFloat32Rows{4,8,8Blocked}SSE are implemented in dot_rows_amd64.s. The
// declarations stay under the plain amd64 tag (not gated on goexperiment.simd)
// so a goexperiment.simd build can still call them directly for parity tests
// and benchmarks against the simd variants in dot_rows_simd_amd64.go, even
// though the dotFloat32Rows* dispatch functions themselves call the simd
// kernels in that build (see dot_rows_dispatch_amd64.go).
//
//go:noescape
func dotFloat32Rows4SSE(rows, vec *float32, n int, dst *float32)

//go:noescape
func dotFloat32Rows8SSE(rows, vec *float32, n int, dst *float32)

//go:noescape
func dotFloat32Rows8BlockedSSE(rows, vec *float32, n int, dst *float32)
