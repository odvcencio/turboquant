# TurboQuant

Standalone Go implementation of the TurboQuant MSE-optimal and inner-product-preserving vector quantization algorithm ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874)). Extracted from [GoSX](https://github.com/odvcencio/gosx), where it powers CRDT vector sync, in-memory semantic search, and real-time 3D mesh compression.

Compresses float32 vectors to 1-8 bits per dimension with provably near-optimal distortion. The only Go implementation of TurboQuant. Zero external dependencies. Thread-safe after construction. Deterministic with seed. Compiles to WASM.

## Install

```
go get github.com/odvcencio/turboquant@v0.1.0
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

### Experimental WebGPU scorer

On `js/wasm`, TurboQuant can upload a quantized IP corpus into an experimental
WebGPU scorer for repeated prepared-query evaluation in the browser. Other
platforms return `ErrGPUBackendUnavailable`. The first pass currently requires
prepared-MSE-LUT-compatible IP bit widths (`2`, `3`, or `5`). It supports
single-query scores, single-query GPU top-k, batched prepared-query GPU top-k,
and uploaded prepared-query batches that stay resident on GPU across repeated
searches.

```go
q := turboquant.NewIPHadamardWithSeed(384, 3, 42)

vectors := []turboquant.IPQuantized{
    q.Quantize(vec0),
    q.Quantize(vec1),
}

scorer, err := q.NewGPUPreparedScorer(vectors)
if err != nil {
    // Browser without WebGPU, or non-js/wasm build.
}
defer scorer.Close()

pq := q.PrepareQuery(queryVec)
scores, err := scorer.ScorePreparedQuery(pq)

batchPQs := []turboquant.PreparedQuery{
    q.PrepareQuery(query0),
    q.PrepareQuery(query1),
}
indices, scores, err := scorer.ScorePreparedQueriesTopK(batchPQs, 10)

uploaded, err := scorer.UploadPreparedQueries(batchPQs)
if err != nil {
    // handle unsupported GPU path
}
defer uploaded.Close()

indices, scores, err = uploaded.ScoreTopK(10)
```

If your quantized corpus is already stored in flat buffers, use
`GPUPreparedData` with `NewGPUPreparedScorerFromData` to avoid repacking.

For a browser-backed smoke test that compares GPU scores against the CPU
prepared-query path, build and run:

```bash
npm install --no-save playwright
npx playwright install chromium
GOOS=js GOARCH=wasm go build -o examples/webgpu-smoke/main.wasm ./examples/webgpu-smoke
node scripts/webgpu_smoke.mjs
```

The script expects Playwright plus a Chromium install and enables WebGPU for
the browser run.

### Experimental native CUDA scorer

On `linux/amd64`, TurboQuant also has an experimental native CUDA backend behind
the `cuda` build tag. This keeps the quantized corpus resident on device and
supports prepared-query scoring, top-k search, batched top-k search, and
uploaded prepared-query batches for repeated reuse.

Current shape:

- CUDA backend is built with `-tags cuda`
- uses the CUDA driver API plus NVRTC for runtime kernel compilation
- top-k selection still happens on the CPU after score readback
- intended as the bridge from library-grade search kernels toward full KV-cache
  and local-model integration

Example validation:

```bash
go test -tags cuda ./... -count=1
```

### `tqserve`: OpenAI-compatible local server

TurboQuant now ships an OpenAI-compatible HTTP server in
[`cmd/tqserve`](/home/draco/work/turboquant/cmd/tqserve/main.go). It already
works as a runtime front door: it can expose local backends behind a stable
OpenAI-style surface while the native TurboQuant executor handles session
memory, checkpointing, and KV-backed retrieval underneath.

Current server shape:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/tq/status`
- `GET /v1/tq/sessions`
- `GET|POST|DELETE /v1/tq/agents`
- `GET|POST|DELETE /v1/tq/claims`
- `GET|POST /v1/tq/events`
- `GET /v1/tq/checkpoints`
- `POST /v1/tq/checkpoints`
- `POST /v1/tq/checkpoints/restore`
- `GET /metrics`

Current backend types:

- `upstream`: any OpenAI-compatible backend reachable at `/v1`
- `ollama`: native Ollama API translation through `/api/chat` and `/api/tags`
- `native`: in-process TurboQuant session runtime with live KV-backed capacity reporting and semantic-memory-style turn retrieval
- `managed_upstream`: launch and supervise a local OpenAI-compatible process
- `managed_ollama`: launch and supervise a local Ollama-compatible process

Current runtime surface:

- in-memory session tracking keyed by `X-TQ-Session-ID`
- pluggable session storage behind the `tqserve` library config
- checkpoint capture and restore plumbing for future session migration
- native in-process runtime sessions backed by real TurboQuant KV pages
- incremental turn syncing for clients that send either full history or just the latest turn
- native responses grounded in retrieved prior session turns instead of a pure status stub
- optional native executor delegation to an OpenAI-compatible or Ollama backend for real model generation on top of TurboQuant session memory
- agent presence, entity claims, and session event feeds for deterministic multi-agent coordination
- backend health/status snapshots for upstream, Ollama, managed processes, and future native executors
- backend capacity snapshots for accelerator, VRAM, KV headroom, and session limits
- optional managed-backend control URLs for live `/v1/tq/status` polling and checkpoint proxying
- Prometheus-style counters for requests, auth failures, backend errors, and active sessions

Quick start against an OpenAI-compatible local runtime:

```bash
go run ./cmd/tqserve \
  --listen :8080 \
  --api-keys sk-local \
  --backend-type upstream \
  --upstream-base-url http://127.0.0.1:8081/v1 \
  --models local-chat=meta-llama/Llama-3.1-8B-Instruct
```

Quick start against Ollama:

```bash
go run ./cmd/tqserve \
  --listen :8080 \
  --api-keys sk-local \
  --backend-type ollama \
  --ollama-base-url http://127.0.0.1:11434 \
  --models local-chat=qwen2.5:7b
```

The `native` backend can also own the session/KV lifecycle while delegating the
final grounded answer to a local model backend. That lets `tqserve` keep
TurboQuant memory, checkpoints, and capacity accounting in one place while a
real model handles text generation. In JSON config, prefer `executor_backend`
to reuse another configured backend by name; `executor_base_url` remains the
lower-level direct wiring option.

You can also run it from a JSON config file:

```json
{
  "listen": ":8080",
  "api_keys": ["sk-local"],
  "default_owner": "turboquant",
  "session_header": "X-TQ-Session-ID",
  "session_idle_ttl": "30m",
  "backends": {
    "ollama": {
      "type": "ollama",
      "base_url": "http://127.0.0.1:11434"
    },
    "llama": {
      "type": "upstream",
      "base_url": "http://127.0.0.1:8081/v1"
    },
    "native-local": {
      "type": "native",
      "model_ids": ["TurboQuant-Local-Executor"],
      "owned_by": "turboquant-native",
      "executor_backend": "ollama",
      "executor_model": "qwen2.5:7b",
      "executor_system_prompt": "Use retrieved session memory when relevant and answer directly.",
      "accelerator": "cuda",
      "device": "RTX 4090",
      "device_count": 1,
      "total_memory_bytes": 25769803776,
      "weights_bytes": 12884901888,
      "max_sessions": 8,
      "key_dim": 384,
      "key_bits": 3,
      "value_dim": 384,
      "value_bits": 2,
      "page_capacity": 4096
    },
    "managed-local": {
      "type": "managed_upstream",
      "base_url": "http://127.0.0.1:8082/v1",
      "health_url": "http://127.0.0.1:8082/v1/models",
      "status_url": "http://127.0.0.1:8082/v1/tq/status",
      "checkpoint_url": "http://127.0.0.1:8082/v1/tq/checkpoints",
      "restore_url": "http://127.0.0.1:8082/v1/tq/checkpoints/restore",
      "command": "./local-runtime",
      "args": ["serve", "--listen", ":8082"],
      "capacity": {
        "accelerator": "cuda",
        "device": "RTX 4090",
        "device_count": 1,
        "total_memory_bytes": 25769803776,
        "kv_headroom_bytes": 8589934592,
        "max_sessions": 12
      },
      "startup_timeout": "60s",
      "shutdown_timeout": "10s"
    }
  },
  "models": [
    {
      "name": "local-chat",
      "backend": "ollama",
      "target": "qwen2.5:7b"
    },
    {
      "name": "local-code",
      "backend": "llama",
      "target": "meta-llama/Llama-3.1-8B-Instruct"
    },
    {
      "name": "local-native-inline",
      "backend": "native-local",
      "target": "TurboQuant-Local-Executor"
    },
    {
      "name": "local-managed",
      "backend": "managed-local",
      "target": "TurboQuant-Local-Executor"
    }
  ]
}
```

Then launch:

```bash
go run ./cmd/tqserve --config ./tqserve.json
```

Checkpoint export and restore are ordinary authenticated JSON calls. A capture
request looks like:

```json
{"session_id":"sess-123"}
```

and a restore request looks like:

```json
{
  "session_id": "sess-restored",
  "checkpoint": {
    "version": "tqserve.session.v1",
    "session": {
      "id": "sess-123",
      "model": "local-chat",
      "backend": "default"
    },
    "state": {"cursor": 7}
  }
}
```

This is the serving layer we can keep stable while adding a native
TurboQuant-backed executor for consumer GPUs.

### Quantized KV cache pages

TurboQuant now includes an append-only quantized KV page API for local-model
workloads. Keys use the IP quantizer for fast query scoring, values use the MSE
quantizer for approximate attention output reconstruction, and the page can
optionally upload its GPU state for repeated attention-style lookups.

```go
page := turboquant.NewKVCachePageWithSeed(128, 3, 128, 2, 256, 42)

page.Append(keyVec, valueVec)
page.Append(nextKeyVec, nextValueVec)

pq := page.PrepareQuery(queryKey)
out := make([]float32, 128)
positions, weights := page.AttentionOutputPreparedTo(out, pq, 16)

_ = positions
_ = weights

if err := page.EnableGPUKeys(); err == nil {
    positions, weights = page.AttentionOutputPreparedTo(out, pq, 16)
}
```

For repeated local-model loops, use the caller-owned attention path to avoid
per-query host allocations:

```go
indices := make([]uint32, 16)
weights := make([]float32, 16)
page.AttentionOutputPreparedInto(out, indices, weights, pq)
```

For repeated multi-head or multi-query loops on native CUDA builds, upload the
prepared query batch once and reuse it across attention calls:

```go
batchPQs := []turboquant.PreparedQuery{
    page.PrepareQuery(head0),
    page.PrepareQuery(head1),
    page.PrepareQuery(head2),
    page.PrepareQuery(head3),
}

uploaded, err := page.UploadPreparedQueries(batchPQs)
if err == nil {
    defer uploaded.Close()

    outBatch := make([]float32, len(batchPQs)*128)
    batchIdx := make([]uint32, len(batchPQs)*16)
    batchWeights := make([]float32, len(batchIdx))
    _ = uploaded.AttentionOutputInto(outBatch, batchIdx, batchWeights)
}
```

On native CUDA builds, `EnableGPUKeys` now uploads both sides of the page:

- keys stay resident for prepared-query top-k scoring
- values stay resident in packed form for device-side rotated-domain weighted
  accumulation
- the host only applies the final inverse rotation once per attention output

### Deterministic quantizers

Two quantizers with the same dim, bitWidth, and seed produce identical output.

```go
q1 := turboquant.NewWithSeed(384, 2, 42)
q2 := turboquant.NewWithSeed(384, 2, 42)
// q1.Quantize(v) == q2.Quantize(v) for all v
```

### Default fast rotation

`New` and `NewIP` now use a structured orthogonal Walsh-Hadamard rotation by
default. This keeps deterministic seeded behavior while cutting rotation cost
substantially on larger dimensions.

```go
q := turboquant.NewHadamardWithSeed(384, 2, 42)
packed, norm := q.Quantize(vec)
recovered := q.Dequantize(packed)
_ = norm
_ = recovered
```

For IP-preserving quantization, use `NewIPHadamard` / `NewIPHadamardWithSeed`.
If you need the legacy dense QR rotation, use `NewDense` / `NewDenseWithSeed`
and `NewIPDense` / `NewIPDenseWithSeed`.

### Caller-owned buffers

For tight loops, reuse output buffers and avoid per-call allocations.

```go
q := turboquant.NewHadamardWithSeed(384, 2, 42)
packed := make([]byte, turboquant.PackedSize(q.Dim(), q.BitWidth()))
norm := q.QuantizeTo(packed, vec)

recovered := make([]float32, q.Dim())
q.DequantizeTo(recovered, packed)
_ = norm
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

Save and restore quantizers. Quantizers are deterministic, so dim, bitWidth,
seed, and rotation family are stored (25 bytes). Legacy 24-byte dense
serialization is still accepted on decode.

```go
data, err := turboquant.MarshalQuantizer(q)
q2, err := turboquant.UnmarshalQuantizer(data)
// q and q2 produce identical output
```

### Portable interop serialization

For non-Go consumers, serialize the full rotation matrix and codebook instead of
relying on Go's seeded reconstruction.

```go
data, err := turboquant.MarshalPortableQuantizer(q)
q2, err := turboquant.UnmarshalPortableQuantizer(data)

ipData, err := turboquant.MarshalPortableIPQuantizer(ipq)
ipq2, err := turboquant.UnmarshalPortableIPQuantizer(ipData)
```

## Used by GoSX

TurboQuant was extracted from [GoSX](https://github.com/odvcencio/gosx), a Go-native web platform that compiles to WASM without CGo. GoSX uses TurboQuant in two subsystems:

### CRDT vector values

GoSX's CRDT package stores embedding vectors inside collaborative documents that sync between replicas in real-time. The MSE quantizer compresses these vectors before transmission and storage. A fixed seed ensures every replica produces byte-identical packed output — critical for CRDT convergence, where all peers must agree on the compressed representation without coordination.

```go
// Inside GoSX's crdt package:
// Every replica uses the same seed, so quantization is deterministic
q := turboquant.NewHadamardWithSeed(dim, bitWidth, vectorQuantSeed)
packed := make([]byte, turboquant.PackedSize(dim, bitWidth))
norm := q.QuantizeTo(packed, embedding)
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

1. **Orthogonal rotation** — By default TurboQuant uses a structured Walsh-Hadamard rotation with random signs and permutation for fast `O(d log d)` application. The legacy dense QR rotation remains available via `NewDense*`. Both aim to Gaussianize coordinates so scalar quantization is effective.

2. **Lloyd-Max codebook** — Compute MSE-optimal scalar quantization centroids for the Beta distribution via the Lloyd-Max algorithm. Centroids and boundaries are cached per (dim, bitWidth) pair.

3. **QJL residual correction** (IP quantizer only) — Apply a 1-bit Quantized Johnson-Lindenstrauss projection to the MSE residual. This corrects the inner product bias from MSE quantization, yielding an unbiased estimator.

Reference: Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate." arXiv 2504.19874, 2025. Accepted at ICLR 2026.

## Performance

Benchmarks on Intel Core Ultra 9 285, pure Go with amd64 SSE row-dot kernels
(no CGo):

| Operation | dim=384 | Allocations |
|-----------|---------|-------------|
| QuantizeTo (2-bit MSE, default hadamard) | 2.3 us | 0 allocs |
| DequantizeTo (2-bit MSE, default hadamard) | 1.5 us | 0 allocs |
| Quantize (3-bit IP, default hadamard) | 11.1 us | 1 alloc |
| InnerProduct (3-bit IP, default hadamard) | 8.3 us | 0 allocs |
| PrepareQueryTo (3-bit IP, default hadamard) | 20.6 us | 0 allocs |
| PreparedQuery score (3-bit IP, default hadamard) | 53 ns | 0 allocs |

## Panic conditions

Construction panics on invalid parameters:
- `New`/`NewWithSeed`: dim < 2 or bitWidth not in [1, 8]
- `NewIP`/`NewIPWithSeed`: dim < 2 or bitWidth < 2

Quantize/dequantize/query methods also panic on malformed caller-provided slice
sizes. Use `ValidateVector`, `ValidatePacked`, `ValidateIPQuantized`, and
`ValidatePreparedQuery` when validating external inputs.

## Thread safety

Quantizers are safe for concurrent use after construction. Internally, scratch buffers are pooled via `sync.Pool`. No locks are held during quantization.

## License

MIT
