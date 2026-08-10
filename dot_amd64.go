//go:build amd64

package turboquant

// dotFloat32sSSE is implemented in dot_amd64.s. The declaration stays under
// the plain amd64 tag (not gated on goexperiment.simd) so a goexperiment.simd
// build can still call it directly for parity tests and benchmarks against
// the simd variant in dot_simd_amd64.go, even though DotFloat32s itself
// dispatches to the simd kernel in that build (see dot_dispatch_amd64.go).
//
//go:noescape
func dotFloat32sSSE(a, b *float32, n int) float32
