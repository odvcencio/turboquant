// Package turboquant implements deterministic vector quantization primitives
// for approximate search, transformer KV-cache compression, and local-model
// serving experiments.
//
// The package exposes MSE-oriented quantizers, inner-product-preserving
// quantizers, packed index helpers, binary and portable quantizer
// serialization, CPU dot-product kernels, optional WebGPU/CUDA scorers, and
// quantized transformer KV-cache data structures.
package turboquant
