# TurboQuant

Standalone Go implementation of the TurboQuant MSE-optimal and inner-product-preserving vector quantization algorithm ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874)). Extracted from [GoSX](https://github.com/odvcencio/gosx), where it powers CRDT vector sync, in-memory semantic search, and real-time 3D mesh compression.

Compresses float32 vectors to 1-8 bits per dimension with provably near-optimal distortion. The only Go implementation of TurboQuant. Zero external dependencies. Thread-safe after construction. Deterministic with seed. Compiles to WASM.

## Install

```
go get github.com/odvcencio/turboquant
```

## Usage

### MSE-optimal quantization

Minimizes reconstruction error. Use when you need to compress and recover vectors.

```go
import "github.com/odvcencio/turboquant"

// Create a 2-bit quantizer for 384-dimensional vectors
q := turboquant.New(384, 2)

// Quantize
packed, norm := q.Quantize(vec)

// Dequantize (approximate reconstruction)
recovered := q.Dequantize(packed)
// Scale by norm to recover original magnitude
for i := range recovered {
    recovered[i] *= norm
}

// Inner product directly from quantized form (no full dequantization)
dot := q.InnerProduct(packed, norm, queryVec)
```

### IP-optimal quantization

Unbiased inner product estimation. Use for similarity search, nearest-neighbor queries.

```go
// Create a 3-bit IP quantizer (uses 2-bit MSE + 1-bit QJL residual)
q := turboquant.NewIP(384, 3)

// Quantize
qx := q.Quantize(vec)

// Estimate inner product
dot := q.InnerProduct(qx, queryVec)

// For repeated queries against many vectors, prepare the query once
pq := q.PrepareQuery(queryVec)
dot := q.InnerProductPrepared(qx, pq) // amortized O(d) instead of O(d^2)
```

### Deterministic quantizers

Two quantizers with the same dim, bitWidth, and seed produce identical output.

```go
q1 := turboquant.NewWithSeed(384, 2, 42)
q2 := turboquant.NewWithSeed(384, 2, 42)
// q1.Quantize(v) == q2.Quantize(v) for all v
```

### Input validation

```go
err := turboquant.ValidateVector(384, vec)
// Returns error for: dimension mismatch, NaN, Inf
```

### Binary wire format

Self-describing binary format for network transmission and storage.

```go
// Encode
wire := turboquant.EncodeMSE(384, 2, packed, norm)

// Decode
dim, bitWidth, packed, norm, err := turboquant.DecodeMSE(wire)

// Also works for IP-quantized vectors
wire := turboquant.EncodeIP(384, 3, qx)
dim, bitWidth, qx, err := turboquant.DecodeIP(wire)
```

22-byte header: magic (`TQ`), version, type, dimension (uint16), bit-width, norm (float32), payload lengths. Big-endian. Max dimension: 65535.

### Serialization

Save and restore quantizers. Quantizers are deterministic, so only dim, bitWidth, and seed are stored (24 bytes).

```go
data, err := turboquant.MarshalQuantizer(q)
q2, err := turboquant.UnmarshalQuantizer(data)
// q and q2 produce identical output
```

## Used by GoSX

TurboQuant was extracted from [GoSX](https://github.com/odvcencio/gosx), a Go-native web platform that compiles to WASM without CGo. GoSX uses TurboQuant in two subsystems:

### CRDT vector values

GoSX's CRDT package stores embedding vectors inside collaborative documents that sync between replicas in real-time. The MSE quantizer compresses these vectors before transmission and storage. A fixed seed ensures every replica produces byte-identical packed output — critical for CRDT convergence, where all peers must agree on the compressed representation without coordination.

```go
// Inside GoSX's crdt package:
// Every replica uses the same seed, so quantization is deterministic
q := turboquant.NewWithSeed(dim, bitWidth, vectorQuantSeed)
packed, norm := q.Quantize(embedding)
// packed bytes are identical on every replica for the same input
```

### In-memory vector search

GoSX's `vecdb` package is a concurrent in-memory vector index backed by the IP quantizer. Vectors are quantized on insertion, and similarity search uses `PrepareQuery` to amortize the O(d^2) projection across all stored vectors. This index powers three semantic primitives built into the framework:

- **SemanticCache** — cache responses by meaning instead of exact key match. A query that's semantically similar to a cached query returns the cached result.
- **SemanticRouter** — route incoming requests to handlers by intent similarity instead of URL pattern matching.
- **ContentIndex** — search pages and documents by semantic similarity for site-wide search.

All three run entirely in-process with no external vector database. The quantized index fits in memory because TurboQuant compresses 384-dim float32 vectors (1,536 bytes) to 144 bytes at 3-bit — a 10x reduction that makes 100K-vector indexes practical at ~14MB.

### 3D scene compression (Kiln)

TurboQuant is the compression layer for [Kiln](https://github.com/odvcencio/kiln), a collaborative 3D creation platform built on GoSX. Mesh vertex data (positions, normals, UVs) is quantized for real-time streaming between collaborators over WebSocket. Vertex positions use 8-bit quantization, normals use 4-bit, and UVs use 8-bit — reducing mesh bandwidth by 4-8x while preserving visual fidelity. Because the entire GoSX stack compiles to WASM, TurboQuant runs on both server (canonical evaluation) and client (preview rendering) without native dependencies.

---

## Bit-width guide

| Bits | Compression | Storage (dim=384) | Use case |
|------|-------------|-------------------|----------|
| 1 | 32x | 48 bytes | Coarse filtering, bloom-style checks |
| 2 | 16x | 96 bytes | Fast approximate search |
| 3 | ~10x | 144 bytes | Good accuracy/size tradeoff |
| 4 | 8x | 192 bytes | High-quality search |
| 8 | 4x | 384 bytes | Near-lossless, still 4x smaller than float32 |

MSE distortion decreases exponentially with bit-width. At 2 bits per dimension, TurboQuant achieves ~2.7x the information-theoretic optimum.

## Algorithm

TurboQuant achieves near-optimal distortion through three steps:

1. **Random rotation** — Apply a random orthogonal rotation (Householder QR). This makes each coordinate of a unit vector follow a Beta distribution concentrated near zero, enabling efficient scalar quantization.

2. **Lloyd-Max codebook** — Compute MSE-optimal scalar quantization centroids for the Beta distribution via the Lloyd-Max algorithm. Centroids and boundaries are cached per (dim, bitWidth) pair.

3. **QJL residual correction** (IP quantizer only) — Apply a 1-bit Quantized Johnson-Lindenstrauss projection to the MSE residual. This corrects the inner product bias from MSE quantization, yielding an unbiased estimator.

Reference: Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." arXiv 2504.19874, 2025. Accepted at ICLR 2026.

## Performance

Benchmarks on Intel Core Ultra 9 285, pure Go (no SIMD, no CGo):

| Operation | dim=384 | Allocations |
|-----------|---------|-------------|
| Quantize (2-bit MSE) | 63 us | 1 alloc |
| Dequantize (2-bit MSE) | 50 us | 1 alloc |
| InnerProduct (2-bit MSE) | 62 us | 0 allocs |
| Quantize (3-bit IP) | 178 us | 3 allocs |
| InnerProduct (3-bit IP) | 115 us | 1 alloc |
| PreparedQuery (3-bit IP) | 52 us | 1 alloc |

## Panic conditions

Construction panics on invalid parameters:
- `New`/`NewWithSeed`: dim < 2 or bitWidth not in [1, 8]
- `NewIP`/`NewIPWithSeed`: dim < 2 or bitWidth < 2

Use `ValidateVector` to check input vectors before quantization.

## Thread safety

Quantizers are safe for concurrent use after construction. Internally, scratch buffers are pooled via `sync.Pool`. No locks are held during quantization.

## License

MIT
