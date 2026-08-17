// Package blockdsp implements the integer block-DSP primitives that VP8-family
// codecs need: the 4x4 forward/inverse transforms, block distortion metrics,
// and uniform dead-zone quantization. m31labs.dev/tqwebp imports this package
// instead of duplicating these kernels, per the tqwebp specification's
// decision that reusable block primitives grow turboquant rather than living
// in the encoder (tqwebp spec section 5.4).
//
// # Normative source
//
// The inverse transforms (IDCT4x4, IWHT4x4) implement RFC 6386 sections 14.3
// (Walsh-Hadamard inversion) and 14.4 (DCT inversion) exactly: same integer
// constants (cospi8sqrt2minus1 = 20091, sinpi8sqrt2 = 35468), same operation
// order, same rounding biases. A conforming VP8 decoder must reproduce these
// values bit for bit, so this package is pinned against
// golang.org/x/image/vp8's idct.go in blockdsp's tests (see oracle_test.go).
//
// The forward transforms (FDCT4x4, FWHT4x4) are not normative: RFC 6386
// section 14 states plainly that "the forward transforms are not normative;
// the encoder picks any forward pair that reconstructs well through the
// normative inverse." FDCT4x4 and FWHT4x4 are derived here algebraically as
// the matrix inverse of the RFC's inverse-transform butterfly (see the
// per-function doc comments for the derivation and the resulting worst-case
// round-trip bound). They are not transcribed from libwebp or any other
// encoder's source.
//
// # Determinism
//
// Every function in this package operates on int16/int32/uint8 fixed-width
// integers only: no floating point, no data-dependent library calls, no
// goroutines. Given identical inputs, every function returns identical
// outputs on every platform and at every optimization level. This is the
// turboquant determinism rule (see the root package's caller-owned-buffer and
// seeded-quantizer discipline), and it holds here for a structural reason
// the float32 kernels do not share: integer addition, subtraction, and
// shifting are exactly reproducible, so there is no accumulation-order or
// rounding-mode hazard to guard against.
//
// # Portability and future assembly
//
// This first landing ships pure-Go implementations only, with no build
// tags. turboquant's existing SIMD kernels (DotFloat32s in dot_amd64.go,
// dot_arm64.go, dot_generic.go) establish the dispatch-trio pattern this
// package will follow once assembly kernels exist: an unconditional exported
// function per file set, gated by //go:build amd64 / arm64 / the negation,
// calling a //go:noescape assembly routine or a portable fallback. The
// tqwebp specification's work package WP-3 is the point at which SAD, SSE,
// SATD, FDCT4x4, and QuantizeBlock grow amd64/arm64 assembly bodies behind
// that same dispatch shape; this package's exported signatures are already
// stable for that migration; only the file layout changes.
//
// # Coefficient and block layout
//
// 4x4 blocks are flattened arrays of 16 elements in raster order: element
// [4*row+col] holds the pixel or coefficient at that row and column, matching
// RFC 6386's own flat-array convention in its reference C source.
package blockdsp
