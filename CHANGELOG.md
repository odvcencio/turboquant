# Changelog

## v0.1.0

First standalone TurboQuant release extracted and matured beyond the original
GoSX implementation.

Highlights:

- Hadamard rotation is the default fast path for MSE and IP quantizers.
- Zero-allocation caller-owned APIs for hot quantize, dequantize, and prepared
  query scoring paths.
- SIMD-accelerated CPU kernels for dot products and grouped QJL projection.
- Portable serialization and stricter public API validation.
- Experimental WebGPU scorer for `js/wasm` and native CUDA scorer behind
  `-tags cuda`.
- KV cache page APIs, checkpointable session primitives, and the first native
  OpenAI-compatible `tqserve` runtime surface.
- GoSX is expected to pin `github.com/odvcencio/turboquant v0.1.0` and can
  keep a local `replace` during active co-development until the tag is pushed.
