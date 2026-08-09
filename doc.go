// Package turboquant implements the TurboQuant vector quantization
// algorithm (arXiv 2504.19874) in pure Go: deterministic, data-oblivious
// quantization primitives for approximate search, transformer KV-cache
// compression, and local-model serving.
//
// Two quantizer families cover the paper's two objectives:
//
//   - Quantizer minimizes reconstruction MSE. It rotates the input with a
//     seeded multi-round randomized Walsh-Hadamard transform, then applies a
//     Lloyd-Max codebook that is optimal for the exact coordinate
//     distribution of a random point on the unit sphere. Empirical MSE
//     matches the paper's Theorem 1 values (0.363, 0.118, 0.035, 0.0095 for
//     1-4 bits) and stays inside the Panter-Dite bound (sqrt(3)*pi/2)/4^b
//     for 5-8 bits, including on adversarial inputs and non-power-of-two
//     dimensions.
//   - IPQuantizer produces unbiased inner-product estimates: an MSE stage at
//     one bit less than the budget, then a 1-bit Quantized
//     Johnson-Lindenstrauss (QJL) sign quantizer on the residual. A bit
//     width of 1 selects the pure QJL estimator. Inputs do not need unit
//     norm; the quantizer stores each input's L2 norm and rescales every
//     estimate, so E[InnerProduct(Quantize(x), y)] = <x, y> for any x.
//
// SplitIPQuantizer and SplitQuantizer compose two independent instances over
// disjoint channel sets, which realizes the paper's fractional bit widths
// (for example 2.5-bit keys: 32 outlier channels at 4 bits plus 96 regular
// channels at 2 bits). SelectOutlierChannels ranks channels for the split.
//
// KVCachePage and SplitKVCachePage store quantized transformer keys and
// values append-only, score keys with prepared queries, and reconstruct
// approximate attention outputs. The package also provides packed index
// helpers, binary and portable quantizer serialization, CPU dot-product
// kernels with amd64/arm64 assembly, and optional WebGPU and CUDA scorers.
//
// The core path needs no CGo and builds on js/wasm.
package turboquant
