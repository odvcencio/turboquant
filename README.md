# TurboQuant

Go implementation of the TurboQuant MSE-optimal and inner-product-preserving
vector quantization algorithm ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874)).

TurboQuant compresses float32 vectors to 1-8 bits per dimension, supports
direct inner-product estimates on quantized vectors, and can run without CGo.
Quantizers are thread-safe after construction, deterministic with a seed, and
available on `js/wasm`.

## Install

```
go get github.com/odvcencio/turboquant@v0.1.2
```

Requires Go 1.25.1 or newer.

## Package surface

- `Quantizer`: MSE-oriented vector quantization, reconstruction, direct inner
  products, batch helpers, deterministic seeded construction, and caller-owned
  buffer APIs.
- `IPQuantizer`: inner-product-preserving quantization with prepared-query
  scoring for repeated search against a quantized corpus.
- `GPUPreparedScorer`: optional WebGPU (`js/wasm`) and CUDA (`linux/amd64`
  with `cgo` and `cuda`) prepared-query scoring and top-k search.
- `KVCachePage`: append-only quantized key/value pages with CPU scoring,
  optional GPU key/value upload, binary serialization, and caller-owned
  attention-output paths.
- `TransformerLayerKVCache` and `TransformerModelKVCache`: per-layer and
  multi-layer transformer KV caches with heterogeneous bit-width profiles.
- `QuantizerSpec`, portable quantizer serialization, `PackIndices`, and
  `UnpackIndices`: interop surfaces for non-Go consumers or custom storage
  layouts.
- `DotFloat32s`: exported SIMD-backed float32 dot product on supported CPU
  architectures, with a generic fallback.
- CUDA dense helpers behind `-tags cuda`: `DenseMatmul`,
  `DenseMatmulTransB`, and `GPUDenseContext` for repeated GPU-resident weight
  matmuls.

## CLI tools

| Command | Purpose |
|---------|---------|
| `cmd/tqserve` | OpenAI-compatible local server with backend routing, sessions, checkpoints, and metrics |
| `cmd/tqeval` | Session-memory comparison harness for OpenAI-compatible targets |
| `cmd/tqkvbench` | Driver for external KV-cache perplexity and throughput benchmarks |
| `cmd/tqkveval` | Offline attention reconstruction eval for captured transformer K/V tensors |
| `cmd/tqkvsweep` | Sweep key/value bit-widths, methods, and top-k settings over capture files |
| `cmd/tqkvprofile` | Build layer profile plans from sweep reports |
| `cmd/tqkvprofilebench` | Replay capture groups through emitted runtime profiles |
| `cmd/tqkvsummarize` | Produce compact summaries from sweep reports |

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

If a quantized corpus is stored in flat buffers, use
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
- the same build tag exposes `DenseMatmul`, `DenseMatmulTransB`, and
  `GPUDenseContext` for dense float32 matmul paths

Example validation:

```bash
go test -tags cuda ./... -count=1
```

### `tqserve`: OpenAI-compatible local server

[`cmd/tqserve`](cmd/tqserve/main.go) is an optional
OpenAI-compatible HTTP server for local testing. It can route requests to local
backends and keep TurboQuant session memory, checkpoints, and KV-backed
retrieval behind one API surface.

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
- `native`: in-process TurboQuant session runtime with KV-backed capacity reporting and turn retrieval
- `managed_upstream`: launch and supervise a local OpenAI-compatible process
- `managed_ollama`: launch and supervise a local Ollama-compatible process

Current runtime surface:

- in-memory session tracking keyed by `X-TQ-Session-ID`
- pluggable session storage behind the `tqserve` library config
- checkpoint capture and restore plumbing
- native in-process runtime sessions backed by real TurboQuant KV pages
- incremental turn syncing for clients that send either full history or just the latest turn
- native responses grounded in retrieved prior session turns
- optional native executor delegation to an OpenAI-compatible or Ollama backend
- agent presence, entity claims, and session event feeds
- backend health/status snapshots for upstream, Ollama, managed processes, and native executors
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

The `native` backend can own the session/KV lifecycle while delegating final
text generation to a local model backend. In JSON config, prefer
`executor_backend` to reuse another configured backend by name;
`executor_base_url` remains the lower-level direct wiring option.

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

### Session-Memory Eval Harness

[`cmd/tqeval`](cmd/tqeval/main.go) is an eval
harness for the current `tqserve` session-memory layer. It runs the same
multi-turn prompt set against two OpenAI-compatible targets:

- a direct local runtime such as `llama-server`
- `tqserve` with the `native` backend delegating final generation to that same
  local runtime

What it measures today:

- per-turn latency
- response size/content previews
- optional `tqserve` status and session snapshots

What it does not measure:

- real transformer KV-cache quality against `llama.cpp` KV quantization
- perplexity
- memory use at `32K` / `64K` / `128K` context
- long-context quality retention under real model attention

That benchmark requires an end-to-end runtime integration path that feeds live
model K/V tensors through TurboQuant during generation. The repository includes
a native transformer-layer KV cache API for per-head K/V tensors, but `tqserve`
is not wired to that path yet.

Use `cmd/tqeval` with a JSON config to drive the same prompts through both
targets and capture the current session-memory behavior:

```bash
go run ./cmd/tqeval --config ./examples/tqeval.json --out ./tqeval-report.json
```

The example config compares a direct `llama.cpp`-compatible `/v1` endpoint with
`tqserve`'s native session-memory route:

```json
{
  "prompts_file": "./examples/tqeval_prompts.txt",
  "targets": [
    {
      "name": "direct-llama",
      "base_url": "http://127.0.0.1:8081/v1",
      "model": "meta-llama/Llama-3.1-8B-Instruct"
    },
    {
      "name": "turbo-native",
      "base_url": "http://127.0.0.1:8080/v1",
      "model": "local-native-inline",
      "session_header": "X-TQ-Session-ID",
      "session_id": "tqeval-demo",
      "status_url": "http://127.0.0.1:8080/v1/tq/status",
      "sessions_url": "http://127.0.0.1:8080/v1/tq/sessions"
    }
  ]
}
```

### `llama.cpp` KV Benchmark Harness

For real KV-cache benchmarking against `llama.cpp`, use
[`cmd/tqkvbench`](cmd/tqkvbench/main.go).
It drives external `llama-perplexity` and `llama-bench` binaries, then records:

- perplexity by context size
- KV cache memory from `llama_kv_cache_init` logs
- prompt and generation throughput from `llama-bench -o json`
- baseline deltas within each run

This benchmark path covers:

- stock `llama.cpp` KV types such as `f16`, `q8_0`, `q4_0`, `iq4_nl`
- builds that expose `turbo3`, `turbo4`, or similar KV types

Example:

```bash
go run ./cmd/tqkvbench --config ./examples/tqkvbench.json --out ./tqkvbench-report.json
```

The example config includes:

- one stock upstream `llama.cpp` run
- one custom KV-type run
- long-context perplexity checkpoints at `8K`, `32K`, and `128K`
- throughput measurements at matching prefill depths

### Offline Transformer KV Capture Eval

For native per-layer K/V ingestion without a live runtime hook yet, use
[`cmd/tqkveval`](cmd/tqkveval/main.go). It reads a
captured transformer-layer attention state, ingests the real query/K/V tensors
into [`TransformerLayerKVCache`](transformer_kv.go),
and reports exact-vs-approximate attention reconstruction metrics for either
TurboQuant or a uniform scalar baseline.

Example:

```bash
go run ./cmd/tqkveval \
  --input ./examples/tqkveval_capture.json \
  --method turboquant \
  --key-bits 3 \
  --value-bits 2 \
  --top-k 2 \
  --out ./tqkveval-report.json
```

The input format is plain JSON so captures from Python, Go, or another runtime
can be dumped directly without extra tooling:

- `query`: one flattened `[heads*head_dim]` query tensor
- `keys`: flattened token-major `[tokens*heads*head_dim]` key tensor
- `values`: flattened token-major `[tokens*heads*head_dim]` value tensor
- optional `name`, explicit `tokens`, and `query_scale`

If `query_scale` is present in the capture, `tqkveval` and `tqkvsweep` will use
it automatically unless you pass `--query-scale` to override it.

Supported offline evaluation methods are:

- `turboquant`: the repo's existing Lloyd-Max plus IP-aware quantization path
- `uniform`: a uniform scalar baseline over per-vector normalized coordinates,
  using the same bit budgets and per-vector norm storage

To compare multiple TurboQuant settings on the same captured layer, use
[`cmd/tqkvsweep`](cmd/tqkvsweep/main.go):

```bash
go run ./cmd/tqkvsweep \
  --input ./examples/tqkveval_capture.json \
  --methods turboquant,uniform \
  --key-bits 2,3,4 \
  --value-bits 2,3,4 \
  --top-k 1,2,3 \
  --out ./tqkvsweep-report.json
```

That produces one report per sample with:

- preserved sample metadata such as `model`, `prompt_index`, `layer`, and
  `token_index` when present in the capture file
- all evaluated `(method, key_bits, value_bits, top_k)` cases
- best reconstruction by MSE
- best reconstruction by cosine similarity
- smallest cache footprint among evaluated cases
- an aggregate `configurations` section summarizing mean/p50/p95 MSE, cosine,
  cache bytes, compression ratio, and per-sample win counts for each evaluated
  setting across the whole capture file
- a top-level `pareto_frontier` section containing the non-dominated
  quality/compression tradeoffs by mean storage bytes and mean MSE

For real-model captures, use the optional helper script
[`scripts/export_hf_llama_kv_capture.py`](scripts/export_hf_llama_kv_capture.py).
It targets Hugging Face Llama-style models and emits the JSON capture format
consumed by both CLIs:

```bash
python3 ./scripts/export_hf_llama_kv_capture.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt "Summarize the deployment plan." \
  --layer 0 \
  --out ./layer0.json
```

The exporter can also emit a whole capture file in one run by combining
multiple prompts, layers, or token positions:

```bash
python3 ./scripts/export_hf_llama_kv_capture.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --prompt-file ./prompts.txt \
  --layer 0,8,16 \
  --token-index last,-8 \
  --out ./captures.json
```

That `captures.json` file can go straight into `tqkvsweep`, which reports both
per-sample results and aggregate per-configuration summaries across the whole
capture set.

To turn a `tqkvsweep` report into a layer-by-layer allocation plan, use
[`cmd/tqkvprofile`](cmd/tqkvprofile/main.go):

```bash
go run ./cmd/tqkvprofile \
  --input ./tqkvsweep-report.json \
  --method turboquant \
  --min-mean-compression 8 \
  --out ./tqkvprofile.json
```

Or, if you want the smallest layer configs that still clear a quality floor:

```bash
go run ./cmd/tqkvprofile \
  --input ./tqkvsweep-report.json \
  --method turboquant \
  --min-mean-cosine 0.70 \
  --min-p05-cosine 0.50 \
  --out ./tqkvprofile.json
```

Or, if you want a late-context-safe profile from the same sweep report:

```bash
go run ./cmd/tqkvprofile \
  --input ./tqkvsweep-report.json \
  --method turboquant \
  --token-position last \
  --min-mean-cosine 0.65 \
  --min-p05-cosine 0.55 \
  --out ./tqkvprofile-last.json
```

Or, if you want both the overall and late-context profiles in one file:

```bash
go run ./cmd/tqkvprofile \
  --input ./tqkvsweep-report.json \
  --method turboquant \
  --profile-set all,last \
  --min-mean-cosine 0.65 \
  --min-p05-cosine 0.50 \
  --out ./tqkvprofile-bundle.json
```

That emits:

- one chosen `(key_bits, value_bits, top_k)` setting per layer
- aggregate quality and storage for the selected plan
- a runtime-ready `profiles` array of
  [`TransformerLayerKVProfile`](transformer_model_kv.go)
  entries you can feed into a heterogeneous multi-layer KV stack
- native `kv_heads` for GQA models, so runtime profiles preserve the real KV
  cache shape instead of expanding to query-head count

To turn one of those emitted profile files back into a runtime-shaped bench over
real captured K/V tensors, use
[`cmd/tqkvprofilebench`](cmd/tqkvprofilebench/main.go):

```bash
go run ./cmd/tqkvprofilebench \
  --capture ./captures.json \
  --profile ./tqkvprofile-bundle.json \
  --profile-set all,last \
  --warmup 1 \
  --iterations 3 \
  --out ./tqkvprofilebench.json
```

That replays complete capture groups through the selected runtime profile and
reports:

- quantized KV live/storage bytes versus raw `fp32` and estimated `fp16`
- append throughput on the real TurboQuant KV-cache path
- attention-query throughput for the selected `(key_bits, value_bits, top_k)`
- optional GPU-key upload timing when `--gpu` is enabled

To turn a `tqkvsweep` report into a smaller comparison digest with overall,
per-method, per-layer, per-token, and relative-position frontiers, use
[`cmd/tqkvsummarize`](cmd/tqkvsummarize/main.go):

```bash
go run ./cmd/tqkvsummarize \
  --input ./tqkvsweep-report.json \
  --out ./tqkvsweep-summary.json
```

### Quantized KV cache pages

TurboQuant includes an append-only quantized KV page API for local-model
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

For real transformer layers, use
[`TransformerLayerKVCache`](transformer_kv.go) for
one layer or
[`TransformerModelKVCache`](transformer_model_kv.go)
to assign different bit widths per layer:

```go
profiles := []turboquant.TransformerLayerKVProfile{
    {Layer: 0, Heads: 32, KVHeads: 8, HeadDim: 128, KeyBits: 2, ValueBits: 3, Capacity: 4096, Seed: 11},
    {Layer: 15, Heads: 32, KVHeads: 8, HeadDim: 128, KeyBits: 4, ValueBits: 4, Capacity: 4096, Seed: 29},
    {Layer: 31, Heads: 32, KVHeads: 8, HeadDim: 128, KeyBits: 4, ValueBits: 4, Capacity: 4096, Seed: 47},
}

stack := turboquant.NewTransformerModelKVCache(profiles)
stack.Append(0, layer0Keys, layer0Values)
stack.Append(15, layer15Keys, layer15Values)
stack.Append(31, layer31Keys, layer31Values)
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

On native CUDA builds, `EnableGPUKeys` uploads both sides of the page:

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

`New` and `NewIP` use a structured orthogonal Walsh-Hadamard rotation by
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

### Portable quantizer specs and index buffers

`Spec` exposes the rotation and codebook data needed to reproduce a quantizer
outside Go. The index APIs expose one scalar codebook index per coordinate when
callers need their own packed format.

```go
spec := q.Spec()
_ = spec.Centroids
_ = spec.RotationKind

indices := make([]int, q.Dim())
norm := q.QuantizeIndicesTo(indices, vec)

packed := make([]byte, turboquant.PackedSize(q.Dim(), q.BitWidth()))
turboquant.PackIndices(packed, indices, q.BitWidth())

decoded := make([]int, q.Dim())
turboquant.UnpackIndices(decoded, packed, q.Dim(), q.BitWidth())

recovered := make([]float32, q.Dim())
q.DequantizeIndicesTo(recovered, decoded)
_ = norm
```

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
| QuantizeTo (2-bit MSE, default hadamard) | 3.1 us | 0 allocs |
| DequantizeTo (2-bit MSE, default hadamard) | 2.0 us | 0 allocs |
| Quantize (3-bit IP, default hadamard) | 11.1 us | 1 alloc |
| InnerProduct (3-bit IP, default hadamard) | 11.0 us | 0 allocs |
| PrepareQueryTo (3-bit IP, default hadamard) | 29.2 us | 0 allocs |
| PreparedQuery score (3-bit IP, default hadamard) | 70.7 ns | 0 allocs |

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
