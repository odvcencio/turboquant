# Changelog

## v0.2.1

Paper-fidelity release. Brings the implementation in line with the TurboQuant
paper (arXiv 2504.19874) for worst-case inputs, non-unit-norm vectors, and all
bit widths.

### Breaking changes

- **Inner-product estimates now scale by the input vector norm.** `IPQuantized`
  gained a `Norm` field; `InnerProduct`, the prepared and batch scoring paths,
  `Dequantize`, and the GPU scorers all rescale by it. Previously every IP
  estimate was implicitly unit-space, so a non-unit input `x` was scored as
  `<x/||x||, y>`. Transformer keys are not unit-norm, so quantized attention
  was silently cosine attention. `E[InnerProduct(Quantize(x), y)] = <x, y>`
  now holds for any `x`.
- **`PreparedQuery.ScoreUpperBound` takes an input norm.** Its signature is now
  `ScoreUpperBound(norm, resNorm float32)`.
- **The embedded Lloyd-Max codebook table changed for every (dim, bitWidth).**
  The previous table did not converge at high dimension — for example the
  dim=1024 2-bit codebook produced the 1-bit MSE, losing a whole bit. The new
  compander-seeded solver sits at the Lloyd-Max optimum, under the Panter-Dite
  bound `(sqrt(3)*pi/2)/4^b`, everywhere. Because compact quantizer
  serialization re-derives the codebook from `(dim, bitWidth, seed, rotation)`,
  **packed payloads written with v0.1.x compact serialization dequantize
  against the new centroids and are silently wrong.** Re-quantize any persisted
  vectors, or use portable serialization (which embeds the centroids and is
  unaffected). tqserve checkpoints written before v0.2.1 decode with key input
  norms defaulting to 1; long-lived sessions should re-ingest for true-IP
  scoring.

## v0.2.0

Module-path release: the module path moved to `m31labs.dev/turboquant`. No
functional change to the quantizer beyond the earlier v0.1.x line; the
paper-fidelity work is v0.2.1.

### Added

- **`IPQuantizer` bit width 1**: the paper's pure 1-bit QJL estimator (no MSE
  stage).
- **Channel-split fractional bit widths**: `SplitIPQuantizer`,
  `SplitQuantizer`, `SplitKVCachePage`, and `SelectOutlierChannels` implement
  the paper's outlier-channel splitting (for example 2.5-bit keys: 32 channels
  at 4 bits plus 96 at 2 bits over head dimension 128). CPU path only; the GPU
  scorers do not support split pages, and split pages are not yet
  checkpointable (no `MarshalBinary`).
- **Multi-round randomized Walsh-Hadamard rotation** is the default (3 rounds).
  A single round left a one-hot input exactly zero outside its power-of-two
  block, breaking the paper's worst-case distortion bound; fresh permutations
  between rounds mix energy across every block. `NewHadamardRounds*` sets an
  explicit round count; a count of 1 reproduces the legacy transform.
- Wire format IP records carry the input norm in a 26-byte version-2 header;
  version-1 IP records still decode, with the norm defaulting to 1. KV page
  serialization is version 2 with a per-token key-norm block; version-1 pages
  decode with key norms defaulting to 1.
- Paper-fidelity test suite pinning the MSE and IP distortion tables, the
  Panter-Dite band, the information-theoretic lower bound, adversarial inputs,
  and non-power-of-two dimensions.

### Fixed

- `KVCachePage.Append` never wrote the QJL residual norm back into the page,
  so every stored key was scored without its residual-correction term.
- `UnmarshalKVCachePage` now bounds the declared capacity by the serialized
  size before allocating, closing an unauthenticated memory-exhaustion vector
  through tqserve checkpoint restore, and uses full reads so an empty page and
  truncated payloads round-trip correctly.
- The default rotation cost rose (2-bit quantize at dim=384 is ~5.7 us, about
  1.8x the single-round transform); prepared-query scoring, the search hot
  path, is unchanged at ~63 ns.

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
