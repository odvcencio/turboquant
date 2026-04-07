//go:build js && wasm

package turboquant

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
	"syscall/js"
)

const gpuPreparedScoreWGSL = `
struct Params {
	count: u32,
	mse_bytes: u32,
	sign_bytes: u32,
	top_k: u32,
	qjl_scale: f32,
	_pad1: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> mse_words: array<u32>;
@group(0) @binding(1) var<storage, read> sign_words: array<u32>;
@group(0) @binding(2) var<storage, read> res_norms: array<f32>;
@group(0) @binding(3) var<storage, read> mse_lut: array<f32>;
@group(0) @binding(4) var<storage, read> sign_lut: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;

fn load_byte(words: ptr<storage, array<u32>, read>, idx: u32) -> u32 {
	let word = (*words)[idx >> 2u];
	let shift = (idx & 3u) * 8u;
	return (word >> shift) & 0xffu;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let row = gid.x;
	if (row >= params.count) {
		return;
	}

	let mse_base = row * params.mse_bytes;
	let sign_base = row * params.sign_bytes;

	var mse_score: f32 = 0.0;
	for (var i: u32 = 0u; i < params.mse_bytes; i = i + 1u) {
		let packed = load_byte(&mse_words, mse_base + i);
		mse_score = mse_score + mse_lut[i * 256u + packed];
	}

	var sign_sum: f32 = 0.0;
	for (var i: u32 = 0u; i < params.sign_bytes; i = i + 1u) {
		let packed = load_byte(&sign_words, sign_base + i);
		sign_sum = sign_sum + sign_lut[i * 256u + packed];
	}

	scores[row] = mse_score + (params.qjl_scale * res_norms[row]) * sign_sum;
}
`

const gpuPreparedTopKWGSL = `
const MAX_TOPK: u32 = 64u;

struct Params {
	count: u32,
	mse_bytes: u32,
	sign_bytes: u32,
	top_k: u32,
	qjl_scale: f32,
	_pad1: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> scores_in: array<f32>;
@group(0) @binding(1) var<storage, read> tie_ranks: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> top_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> top_scores: array<f32>;

fn better(score_a: f32, rank_a: u32, idx_a: u32, score_b: f32, rank_b: u32, idx_b: u32) -> bool {
	if (score_a > score_b) {
		return true;
	}
	if (score_a < score_b) {
		return false;
	}
	if (rank_a < rank_b) {
		return true;
	}
	if (rank_a > rank_b) {
		return false;
	}
	return idx_a < idx_b;
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x != 0u || gid.y != 0u || gid.z != 0u) {
		return;
	}
	let k = params.top_k;
	if (k == 0u) {
		return;
	}

	var best_indices: array<u32, MAX_TOPK>;
	var best_scores: array<f32, MAX_TOPK>;
	var best_ranks: array<u32, MAX_TOPK>;

	for (var i: u32 = 0u; i < MAX_TOPK; i = i + 1u) {
		best_indices[i] = 0xffffffffu;
		best_scores[i] = -3.402823466e+38;
		best_ranks[i] = 0xffffffffu;
	}

	for (var row: u32 = 0u; row < params.count; row = row + 1u) {
		let candidate_score = scores_in[row];
		let candidate_rank = tie_ranks[row];
		var insert: u32 = k;
		for (var pos: u32 = 0u; pos < k; pos = pos + 1u) {
			if (better(candidate_score, candidate_rank, row, best_scores[pos], best_ranks[pos], best_indices[pos])) {
				insert = pos;
				break;
			}
		}
		if (insert == k) {
			continue;
		}
		var pos: i32 = i32(k) - 1;
		loop {
			if (pos <= i32(insert)) {
				break;
			}
			let dst = u32(pos);
			let src = u32(pos - 1);
			best_indices[dst] = best_indices[src];
			best_scores[dst] = best_scores[src];
			best_ranks[dst] = best_ranks[src];
			pos = pos - 1;
		}
		best_indices[insert] = row;
		best_scores[insert] = candidate_score;
		best_ranks[insert] = candidate_rank;
	}

	for (var i: u32 = 0u; i < k; i = i + 1u) {
		top_indices[i] = best_indices[i];
		top_scores[i] = best_scores[i];
	}
}
`

const gpuPreparedBatchScoreWGSL = `
struct Params {
	count: u32,
	mse_bytes: u32,
	sign_bytes: u32,
	top_k: u32,
	query_count: u32,
	qjl_scale: f32,
	_pad1: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> mse_words: array<u32>;
@group(0) @binding(1) var<storage, read> sign_words: array<u32>;
@group(0) @binding(2) var<storage, read> res_norms: array<f32>;
@group(0) @binding(3) var<storage, read> query_mse_lut: array<f32>;
@group(0) @binding(4) var<storage, read> query_sign_lut: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;

fn load_byte(words: ptr<storage, array<u32>, read>, idx: u32) -> u32 {
	let word = (*words)[idx >> 2u];
	let shift = (idx & 3u) * 8u;
	return (word >> shift) & 0xffu;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let row = gid.x;
	let query = gid.y;
	if (row >= params.count || query >= params.query_count) {
		return;
	}

	let mse_base = row * params.mse_bytes;
	let sign_base = row * params.sign_bytes;
	let query_mse_base = query * params.mse_bytes * 256u;
	let query_sign_base = query * params.sign_bytes * 256u;

	var mse_score: f32 = 0.0;
	for (var i: u32 = 0u; i < params.mse_bytes; i = i + 1u) {
		let packed = load_byte(&mse_words, mse_base + i);
		mse_score = mse_score + query_mse_lut[query_mse_base + i * 256u + packed];
	}

	var sign_sum: f32 = 0.0;
	for (var i: u32 = 0u; i < params.sign_bytes; i = i + 1u) {
		let packed = load_byte(&sign_words, sign_base + i);
		sign_sum = sign_sum + query_sign_lut[query_sign_base + i * 256u + packed];
	}

	scores[query * params.count + row] = mse_score + (params.qjl_scale * res_norms[row]) * sign_sum;
}
`

const gpuPreparedBatchTopKWGSL = `
const MAX_TOPK: u32 = 64u;

struct Params {
	count: u32,
	mse_bytes: u32,
	sign_bytes: u32,
	top_k: u32,
	query_count: u32,
	qjl_scale: f32,
	_pad1: vec2<f32>,
};

@group(0) @binding(0) var<storage, read> scores_in: array<f32>;
@group(0) @binding(1) var<storage, read> tie_ranks: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> top_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> top_scores: array<f32>;

fn better(score_a: f32, rank_a: u32, idx_a: u32, score_b: f32, rank_b: u32, idx_b: u32) -> bool {
	if (score_a > score_b) {
		return true;
	}
	if (score_a < score_b) {
		return false;
	}
	if (rank_a < rank_b) {
		return true;
	}
	if (rank_a > rank_b) {
		return false;
	}
	return idx_a < idx_b;
}

@compute @workgroup_size(1, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let query = gid.x;
	if (query >= params.query_count) {
		return;
	}
	let k = params.top_k;
	if (k == 0u) {
		return;
	}

	let score_base = query * params.count;
	let out_base = query * k;

	var best_indices: array<u32, MAX_TOPK>;
	var best_scores: array<f32, MAX_TOPK>;
	var best_ranks: array<u32, MAX_TOPK>;

	for (var i: u32 = 0u; i < MAX_TOPK; i = i + 1u) {
		best_indices[i] = 0xffffffffu;
		best_scores[i] = -3.402823466e+38;
		best_ranks[i] = 0xffffffffu;
	}

	for (var row: u32 = 0u; row < params.count; row = row + 1u) {
		let candidate_score = scores_in[score_base + row];
		let candidate_rank = tie_ranks[row];
		var insert: u32 = k;
		for (var pos: u32 = 0u; pos < k; pos = pos + 1u) {
			if (better(candidate_score, candidate_rank, row, best_scores[pos], best_ranks[pos], best_indices[pos])) {
				insert = pos;
				break;
			}
		}
		if (insert == k) {
			continue;
		}
		var pos: i32 = i32(k) - 1;
		loop {
			if (pos <= i32(insert)) {
				break;
			}
			let dst = u32(pos);
			let src = u32(pos - 1);
			best_indices[dst] = best_indices[src];
			best_scores[dst] = best_scores[src];
			best_ranks[dst] = best_ranks[src];
			pos = pos - 1;
		}
		best_indices[insert] = row;
		best_scores[insert] = candidate_score;
		best_ranks[insert] = candidate_rank;
	}

	for (var i: u32 = 0u; i < k; i = i + 1u) {
		top_indices[out_base + i] = best_indices[i];
		top_scores[out_base + i] = best_scores[i];
	}
}
`

type GPUPreparedScorer struct {
	mu sync.Mutex

	dim         int
	bitWidth    int
	mseBitWidth int
	count       int
	mseBytes    int
	signBytes   int
	qjlScale    float32

	device js.Value
	queue  js.Value

	scorePipeline  js.Value
	scoreBindGroup js.Value
	topKPipeline   js.Value
	topKBindGroup  js.Value
	batchScorePipe js.Value
	batchScoreBind js.Value
	batchTopKPipe  js.Value
	batchTopKBind  js.Value

	mseBuffer      js.Value
	signBuffer     js.Value
	resNormBuffer  js.Value
	rankBuffer     js.Value
	mseLUTBuffer   js.Value
	signLUTBuffer  js.Value
	paramsBuffer   js.Value
	outputBuffer   js.Value
	readBuffer     js.Value
	topKIndexBuf   js.Value
	topKScoreBuf   js.Value
	topKIndexRead  js.Value
	topKScoreRead  js.Value
	batchMSEBuf    js.Value
	batchSignBuf   js.Value
	batchParamBuf  js.Value
	batchOutBuf    js.Value
	batchTopKIdx   js.Value
	batchTopKSc    js.Value
	batchTopKRead  js.Value
	batchTopKSRead js.Value

	outputBytes int
	topKReady   bool
	batchCap    int
	closed      bool

	queryMSEUpload    []byte
	querySignUpload   []byte
	batchMSEUpload    []byte
	batchSignUpload   []byte
	paramsScratch     [48]byte
	batchParamScratch [32]byte
}

type GPUPreparedQueryBatch struct {
	scorer *GPUPreparedScorer
	count  int
	closed bool

	mseBuffer      js.Value
	signBuffer     js.Value
	paramBuffer    js.Value
	outputBuffer   js.Value
	topKIndexBuf   js.Value
	topKScoreBuf   js.Value
	topKIndexRead  js.Value
	topKScoreRead  js.Value
	scoreBindGroup js.Value
	topKBindGroup  js.Value
}

func newGPUPreparedScorer(q *IPQuantizer, data GPUPreparedData) (*GPUPreparedScorer, error) {
	count, err := ValidateGPUPreparedData(q.dim, q.bitWidth, data)
	if err != nil {
		return nil, err
	}

	gpu := js.Global().Get("navigator").Get("gpu")
	if gpu.IsUndefined() || gpu.IsNull() {
		return nil, ErrGPUBackendUnavailable
	}
	adapter, err := awaitJSPromise(gpu.Call("requestAdapter"))
	if err != nil {
		return nil, fmt.Errorf("turboquant: requestAdapter failed: %w", err)
	}
	if adapter.IsUndefined() || adapter.IsNull() {
		return nil, ErrGPUBackendUnavailable
	}
	device, err := awaitJSPromise(adapter.Call("requestDevice"))
	if err != nil {
		return nil, fmt.Errorf("turboquant: requestDevice failed: %w", err)
	}
	if device.IsUndefined() || device.IsNull() {
		return nil, fmt.Errorf("turboquant: requestDevice returned no device")
	}
	queue := device.Get("queue")
	if queue.IsUndefined() || queue.IsNull() {
		return nil, fmt.Errorf("turboquant: WebGPU device has no queue")
	}

	mseBytes, signBytes := IPQuantizedSizes(q.dim, q.bitWidth)
	mseData := padBytesToWord(data.MSE)
	signData := padBytesToWord(data.Signs)
	resNormData := float32Bytes(data.ResNorms)
	rankData := uint32Bytes(data.TieBreakRanks)
	mseLUTBytes := preparedQueryMSELUTLen(q.dim, q.mse.bitWidth) * 4
	signLUTBytes := preparedQuerySignLUTLen(q.dim) * 4
	outputBytes := count * 4

	bufferUsage := js.Global().Get("GPUBufferUsage")
	storageUsage := bufferUsage.Get("STORAGE").Int()
	copyDstUsage := bufferUsage.Get("COPY_DST").Int()
	copySrcUsage := bufferUsage.Get("COPY_SRC").Int()
	uniformUsage := bufferUsage.Get("UNIFORM").Int()
	mapReadUsage := bufferUsage.Get("MAP_READ").Int()

	s := &GPUPreparedScorer{
		dim:         q.dim,
		bitWidth:    q.bitWidth,
		mseBitWidth: q.mse.bitWidth,
		count:       count,
		mseBytes:    mseBytes,
		signBytes:   signBytes,
		qjlScale:    float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim),
		device:      device,
		queue:       queue,
		outputBytes: outputBytes,
	}

	s.mseBuffer = createGPUBuffer(device, len(mseData), storageUsage|copyDstUsage)
	s.signBuffer = createGPUBuffer(device, len(signData), storageUsage|copyDstUsage)
	s.resNormBuffer = createGPUBuffer(device, len(resNormData), storageUsage|copyDstUsage)
	s.mseLUTBuffer = createGPUBuffer(device, mseLUTBytes, storageUsage|copyDstUsage)
	s.signLUTBuffer = createGPUBuffer(device, signLUTBytes, storageUsage|copyDstUsage)
	s.paramsBuffer = createGPUBuffer(device, 48, uniformUsage|copyDstUsage)
	s.outputBuffer = createGPUBuffer(device, maxGPUBufferSize(outputBytes), storageUsage|copySrcUsage)
	s.readBuffer = createGPUBuffer(device, maxGPUBufferSize(outputBytes), mapReadUsage|copyDstUsage)

	writeGPUBuffer(queue, s.mseBuffer, mseData)
	writeGPUBuffer(queue, s.signBuffer, signData)
	writeGPUBuffer(queue, s.resNormBuffer, resNormData)

	scoreShader := device.Call("createShaderModule", map[string]any{
		"code": gpuPreparedScoreWGSL,
	})
	s.scorePipeline = device.Call("createComputePipeline", map[string]any{
		"layout": "auto",
		"compute": map[string]any{
			"module":     scoreShader,
			"entryPoint": "main",
		},
	})
	scoreLayout := s.scorePipeline.Call("getBindGroupLayout", 0)
	s.scoreBindGroup = device.Call("createBindGroup", map[string]any{
		"layout": scoreLayout,
		"entries": []any{
			map[string]any{"binding": 0, "resource": map[string]any{"buffer": s.mseBuffer}},
			map[string]any{"binding": 1, "resource": map[string]any{"buffer": s.signBuffer}},
			map[string]any{"binding": 2, "resource": map[string]any{"buffer": s.resNormBuffer}},
			map[string]any{"binding": 3, "resource": map[string]any{"buffer": s.mseLUTBuffer}},
			map[string]any{"binding": 4, "resource": map[string]any{"buffer": s.signLUTBuffer}},
			map[string]any{"binding": 5, "resource": map[string]any{"buffer": s.paramsBuffer}},
			map[string]any{"binding": 6, "resource": map[string]any{"buffer": s.outputBuffer}},
		},
	})

	batchScoreShader := device.Call("createShaderModule", map[string]any{
		"code": gpuPreparedBatchScoreWGSL,
	})
	s.batchScorePipe = device.Call("createComputePipeline", map[string]any{
		"layout": "auto",
		"compute": map[string]any{
			"module":     batchScoreShader,
			"entryPoint": "main",
		},
	})

	batchTopKShader := device.Call("createShaderModule", map[string]any{
		"code": gpuPreparedBatchTopKWGSL,
	})
	s.batchTopKPipe = device.Call("createComputePipeline", map[string]any{
		"layout": "auto",
		"compute": map[string]any{
			"module":     batchTopKShader,
			"entryPoint": "main",
		},
	})

	if len(data.TieBreakRanks) == count {
		s.rankBuffer = createGPUBuffer(device, len(rankData), storageUsage|copyDstUsage)
		writeGPUBuffer(queue, s.rankBuffer, rankData)
		s.topKIndexBuf = createGPUBuffer(device, GPUPreparedTopKMax*4, storageUsage|copySrcUsage)
		s.topKScoreBuf = createGPUBuffer(device, GPUPreparedTopKMax*4, storageUsage|copySrcUsage)
		s.topKIndexRead = createGPUBuffer(device, GPUPreparedTopKMax*4, mapReadUsage|copyDstUsage)
		s.topKScoreRead = createGPUBuffer(device, GPUPreparedTopKMax*4, mapReadUsage|copyDstUsage)

		topKShader := device.Call("createShaderModule", map[string]any{
			"code": gpuPreparedTopKWGSL,
		})
		s.topKPipeline = device.Call("createComputePipeline", map[string]any{
			"layout": "auto",
			"compute": map[string]any{
				"module":     topKShader,
				"entryPoint": "main",
			},
		})
		topKLayout := s.topKPipeline.Call("getBindGroupLayout", 0)
		s.topKBindGroup = device.Call("createBindGroup", map[string]any{
			"layout": topKLayout,
			"entries": []any{
				map[string]any{"binding": 0, "resource": map[string]any{"buffer": s.outputBuffer}},
				map[string]any{"binding": 1, "resource": map[string]any{"buffer": s.rankBuffer}},
				map[string]any{"binding": 2, "resource": map[string]any{"buffer": s.paramsBuffer}},
				map[string]any{"binding": 3, "resource": map[string]any{"buffer": s.topKIndexBuf}},
				map[string]any{"binding": 4, "resource": map[string]any{"buffer": s.topKScoreBuf}},
			},
		})
		s.topKReady = true
	}

	return s, nil
}

func (s *GPUPreparedScorer) Len() int {
	if s == nil {
		return 0
	}
	return s.count
}

func (s *GPUPreparedScorer) ScorePreparedQuery(pq PreparedQuery) ([]float32, error) {
	if s == nil {
		return nil, fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	dst := make([]float32, s.count)
	if err := s.ScorePreparedQueryTo(dst, pq); err != nil {
		return nil, err
	}
	return dst, nil
}

func (s *GPUPreparedScorer) ScorePreparedQueryTo(dst []float32, pq PreparedQuery) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	if err := s.validatePreparedQuery(pq); err != nil {
		return err
	}
	return s.ScorePreparedQueryToTrusted(dst, pq)
}

func (s *GPUPreparedScorer) ScorePreparedQueryToTrusted(dst []float32, pq PreparedQuery) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if len(dst) != s.count {
		return fmt.Errorf("turboquant: expected GPU score destination length %d, got %d", s.count, len(dst))
	}
	if s.count == 0 {
		return nil
	}

	writePreparedQueryBuffers(s, pq, 0)
	encoder := s.device.Call("createCommandEncoder")
	s.encodeScorePass(encoder)
	encoder.Call("copyBufferToBuffer", s.outputBuffer, 0, s.readBuffer, 0, s.outputBytes)
	s.queue.Call("submit", []any{encoder.Call("finish")})

	raw, err := readGPUBufferBytes(s.readBuffer, s.outputBytes)
	if err != nil {
		return fmt.Errorf("turboquant: GPU readback mapAsync failed: %w", err)
	}
	decodeFloat32Bytes(dst, raw)
	return nil
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopK(pq PreparedQuery, k int) ([]uint32, []float32, error) {
	if err := s.validatePreparedQuery(pq); err != nil {
		return nil, nil, err
	}
	indices := make([]uint32, k)
	scores := make([]float32, k)
	if err := s.ScorePreparedQueryTopKToTrusted(indices, scores, pq); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopKTo(indices []uint32, scores []float32, pq PreparedQuery) error {
	if err := s.validatePreparedQuery(pq); err != nil {
		return err
	}
	return s.ScorePreparedQueryTopKToTrusted(indices, scores, pq)
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopKToTrusted(indices []uint32, scores []float32, pq PreparedQuery) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if len(indices) != len(scores) {
		return fmt.Errorf("turboquant: GPU top-k destination length mismatch: %d indices vs %d scores", len(indices), len(scores))
	}
	k := len(indices)
	if err := s.validateTopK(k); err != nil {
		return err
	}
	if k == 0 {
		return nil
	}

	writePreparedQueryBuffers(s, pq, k)
	encoder := s.device.Call("createCommandEncoder")
	s.encodeScorePass(encoder)
	s.encodeTopKPass(encoder)
	readBytes := k * 4
	encoder.Call("copyBufferToBuffer", s.topKIndexBuf, 0, s.topKIndexRead, 0, readBytes)
	encoder.Call("copyBufferToBuffer", s.topKScoreBuf, 0, s.topKScoreRead, 0, readBytes)
	s.queue.Call("submit", []any{encoder.Call("finish")})

	indexBytes, err := readGPUBufferBytes(s.topKIndexRead, readBytes)
	if err != nil {
		return fmt.Errorf("turboquant: GPU top-k index readback failed: %w", err)
	}
	scoreBytes, err := readGPUBufferBytes(s.topKScoreRead, readBytes)
	if err != nil {
		return fmt.Errorf("turboquant: GPU top-k score readback failed: %w", err)
	}
	decodeUint32Bytes(indices, indexBytes)
	decodeFloat32Bytes(scores, scoreBytes)
	return nil
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopK(pqs []PreparedQuery, k int) ([]uint32, []float32, error) {
	indices := make([]uint32, len(pqs)*k)
	scores := make([]float32, len(pqs)*k)
	if err := s.ScorePreparedQueriesTopKTo(indices, scores, pqs, k); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopKTo(indices []uint32, scores []float32, pqs []PreparedQuery, k int) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	for i := range pqs {
		if err := s.validatePreparedQuery(pqs[i]); err != nil {
			return err
		}
	}
	return s.ScorePreparedQueriesTopKToTrusted(indices, scores, pqs, k)
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopKToTrusted(indices []uint32, scores []float32, pqs []PreparedQuery, k int) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	if k < 0 {
		return fmt.Errorf("turboquant: GPU top-k must be >= 0")
	}
	if len(indices) != len(pqs)*k {
		return fmt.Errorf("turboquant: expected GPU batch top-k index length %d, got %d", len(pqs)*k, len(indices))
	}
	if len(scores) != len(indices) {
		return fmt.Errorf("turboquant: GPU batch top-k destination length mismatch: %d indices vs %d scores", len(indices), len(scores))
	}
	if len(pqs) == 0 || k == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if err := s.validateTopK(k); err != nil {
		return err
	}
	if err := s.ensureBatchTopKCapacity(len(pqs)); err != nil {
		return err
	}

	writePreparedQueryBatchBuffers(s, pqs, k)
	encoder := s.device.Call("createCommandEncoder")
	s.encodeBatchScorePass(encoder, len(pqs))
	s.encodeBatchTopKPass(encoder, len(pqs))
	readBytes := len(pqs) * k * 4
	encoder.Call("copyBufferToBuffer", s.batchTopKIdx, 0, s.batchTopKRead, 0, readBytes)
	encoder.Call("copyBufferToBuffer", s.batchTopKSc, 0, s.batchTopKSRead, 0, readBytes)
	s.queue.Call("submit", []any{encoder.Call("finish")})

	indexBytes, err := readGPUBufferBytes(s.batchTopKRead, readBytes)
	if err != nil {
		return fmt.Errorf("turboquant: GPU batch top-k index readback failed: %w", err)
	}
	scoreBytes, err := readGPUBufferBytes(s.batchTopKSRead, readBytes)
	if err != nil {
		return fmt.Errorf("turboquant: GPU batch top-k score readback failed: %w", err)
	}
	decodeUint32Bytes(indices, indexBytes)
	decodeFloat32Bytes(scores, scoreBytes)
	return nil
}

func (s *GPUPreparedScorer) UploadPreparedQueries(pqs []PreparedQuery) (*GPUPreparedQueryBatch, error) {
	if s == nil {
		return nil, fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	for i := range pqs {
		if err := s.validatePreparedQuery(pqs[i]); err != nil {
			return nil, err
		}
	}
	return s.UploadPreparedQueriesTrusted(pqs)
}

func (s *GPUPreparedScorer) UploadPreparedQueriesTrusted(pqs []PreparedQuery) (*GPUPreparedQueryBatch, error) {
	if s == nil {
		return nil, fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	if len(pqs) == 0 {
		return nil, fmt.Errorf("turboquant: cannot upload an empty GPU prepared-query batch")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if !s.topKReady {
		return nil, fmt.Errorf("turboquant: GPU top-k requires tie-break ranks in GPU prepared data")
	}

	bufferUsage := js.Global().Get("GPUBufferUsage")
	storageUsage := bufferUsage.Get("STORAGE").Int()
	copyDstUsage := bufferUsage.Get("COPY_DST").Int()
	copySrcUsage := bufferUsage.Get("COPY_SRC").Int()
	uniformUsage := bufferUsage.Get("UNIFORM").Int()
	mapReadUsage := bufferUsage.Get("MAP_READ").Int()

	mseBytesPerQuery := preparedQueryMSELUTLen(s.dim, s.mseBitWidth) * 4
	signBytesPerQuery := preparedQuerySignLUTLen(s.dim) * 4
	uploadMSE := make([]byte, len(pqs)*mseBytesPerQuery)
	uploadSign := make([]byte, len(pqs)*signBytesPerQuery)
	for i := range pqs {
		encodeFloat32sBytes(uploadMSE[i*mseBytesPerQuery:(i+1)*mseBytesPerQuery], pqs[i].mseLUT)
		encodeFloat32sBytes(uploadSign[i*signBytesPerQuery:(i+1)*signBytesPerQuery], pqs[i].signLUT)
	}

	batch := &GPUPreparedQueryBatch{
		scorer:        s,
		count:         len(pqs),
		mseBuffer:     createGPUBuffer(s.device, len(uploadMSE), storageUsage|copyDstUsage),
		signBuffer:    createGPUBuffer(s.device, len(uploadSign), storageUsage|copyDstUsage),
		paramBuffer:   createGPUBuffer(s.device, 32, uniformUsage|copyDstUsage),
		outputBuffer:  createGPUBuffer(s.device, len(pqs)*s.outputBytes, storageUsage|copySrcUsage),
		topKIndexBuf:  createGPUBuffer(s.device, len(pqs)*GPUPreparedTopKMax*4, storageUsage|copySrcUsage),
		topKScoreBuf:  createGPUBuffer(s.device, len(pqs)*GPUPreparedTopKMax*4, storageUsage|copySrcUsage),
		topKIndexRead: createGPUBuffer(s.device, len(pqs)*GPUPreparedTopKMax*4, mapReadUsage|copyDstUsage),
		topKScoreRead: createGPUBuffer(s.device, len(pqs)*GPUPreparedTopKMax*4, mapReadUsage|copyDstUsage),
	}
	writeGPUBuffer(s.queue, batch.mseBuffer, uploadMSE)
	writeGPUBuffer(s.queue, batch.signBuffer, uploadSign)

	scoreLayout := s.batchScorePipe.Call("getBindGroupLayout", 0)
	batch.scoreBindGroup = s.device.Call("createBindGroup", map[string]any{
		"layout": scoreLayout,
		"entries": []any{
			map[string]any{"binding": 0, "resource": map[string]any{"buffer": s.mseBuffer}},
			map[string]any{"binding": 1, "resource": map[string]any{"buffer": s.signBuffer}},
			map[string]any{"binding": 2, "resource": map[string]any{"buffer": s.resNormBuffer}},
			map[string]any{"binding": 3, "resource": map[string]any{"buffer": batch.mseBuffer}},
			map[string]any{"binding": 4, "resource": map[string]any{"buffer": batch.signBuffer}},
			map[string]any{"binding": 5, "resource": map[string]any{"buffer": batch.paramBuffer}},
			map[string]any{"binding": 6, "resource": map[string]any{"buffer": batch.outputBuffer}},
		},
	})

	topKLayout := s.batchTopKPipe.Call("getBindGroupLayout", 0)
	batch.topKBindGroup = s.device.Call("createBindGroup", map[string]any{
		"layout": topKLayout,
		"entries": []any{
			map[string]any{"binding": 0, "resource": map[string]any{"buffer": batch.outputBuffer}},
			map[string]any{"binding": 1, "resource": map[string]any{"buffer": s.rankBuffer}},
			map[string]any{"binding": 2, "resource": map[string]any{"buffer": batch.paramBuffer}},
			map[string]any{"binding": 3, "resource": map[string]any{"buffer": batch.topKIndexBuf}},
			map[string]any{"binding": 4, "resource": map[string]any{"buffer": batch.topKScoreBuf}},
		},
	})
	return batch, nil
}

func (b *GPUPreparedQueryBatch) Len() int {
	if b == nil {
		return 0
	}
	return b.count
}

func (b *GPUPreparedQueryBatch) ScoreTopK(k int) ([]uint32, []float32, error) {
	indices := make([]uint32, b.count*k)
	scores := make([]float32, len(indices))
	if err := b.ScoreTopKTo(indices, scores, k); err != nil {
		return nil, nil, err
	}
	return indices, scores, nil
}

func (b *GPUPreparedQueryBatch) ScoreTopKTo(indices []uint32, scores []float32, k int) error {
	if b == nil {
		return fmt.Errorf("turboquant: nil GPU prepared-query batch")
	}
	return b.scoreTopKToTrusted(indices, scores, k)
}

func (b *GPUPreparedQueryBatch) scoreTopKToTrusted(indices []uint32, scores []float32, k int) error {
	if b == nil {
		return fmt.Errorf("turboquant: nil GPU prepared-query batch")
	}
	if k < 0 {
		return fmt.Errorf("turboquant: GPU top-k must be >= 0")
	}
	if len(indices) != b.count*k {
		return fmt.Errorf("turboquant: expected uploaded GPU batch top-k index length %d, got %d", b.count*k, len(indices))
	}
	if len(scores) != len(indices) {
		return fmt.Errorf("turboquant: uploaded GPU batch destination length mismatch: %d indices vs %d scores", len(indices), len(scores))
	}
	if b.count == 0 || k == 0 {
		return nil
	}
	s := b.scorer
	if s == nil {
		return fmt.Errorf("turboquant: uploaded GPU prepared-query batch has no scorer")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("turboquant: GPU prepared scorer is closed")
	}
	if b.closed {
		return fmt.Errorf("turboquant: uploaded GPU prepared-query batch is closed")
	}
	if err := s.validateTopK(k); err != nil {
		return err
	}

	writeGPUBuffer(s.queue, b.paramBuffer, s.batchParamsBytes(k, b.count))
	encoder := s.device.Call("createCommandEncoder")
	pass := encoder.Call("beginComputePass")
	pass.Call("setPipeline", s.batchScorePipe)
	pass.Call("setBindGroup", 0, b.scoreBindGroup)
	pass.Call("dispatchWorkgroups", (s.count+63)/64, b.count)
	pass.Call("end")
	pass = encoder.Call("beginComputePass")
	pass.Call("setPipeline", s.batchTopKPipe)
	pass.Call("setBindGroup", 0, b.topKBindGroup)
	pass.Call("dispatchWorkgroups", b.count)
	pass.Call("end")
	readBytes := b.count * k * 4
	encoder.Call("copyBufferToBuffer", b.topKIndexBuf, 0, b.topKIndexRead, 0, readBytes)
	encoder.Call("copyBufferToBuffer", b.topKScoreBuf, 0, b.topKScoreRead, 0, readBytes)
	s.queue.Call("submit", []any{encoder.Call("finish")})

	indexBytes, err := readGPUBufferBytes(b.topKIndexRead, readBytes)
	if err != nil {
		return fmt.Errorf("turboquant: uploaded GPU batch top-k index readback failed: %w", err)
	}
	scoreBytes, err := readGPUBufferBytes(b.topKScoreRead, readBytes)
	if err != nil {
		return fmt.Errorf("turboquant: uploaded GPU batch top-k score readback failed: %w", err)
	}
	decodeUint32Bytes(indices, indexBytes)
	decodeFloat32Bytes(scores, scoreBytes)
	return nil
}

func (b *GPUPreparedQueryBatch) Close() error {
	if b == nil {
		return nil
	}
	if b.scorer != nil {
		b.scorer.mu.Lock()
		defer b.scorer.mu.Unlock()
	}
	if b.closed {
		return nil
	}
	b.closed = true
	destroyGPUBuffer(b.mseBuffer)
	destroyGPUBuffer(b.signBuffer)
	destroyGPUBuffer(b.paramBuffer)
	destroyGPUBuffer(b.outputBuffer)
	destroyGPUBuffer(b.topKIndexBuf)
	destroyGPUBuffer(b.topKScoreBuf)
	destroyGPUBuffer(b.topKIndexRead)
	destroyGPUBuffer(b.topKScoreRead)
	return nil
}

func (s *GPUPreparedScorer) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	destroyGPUBuffer(s.mseBuffer)
	destroyGPUBuffer(s.signBuffer)
	destroyGPUBuffer(s.resNormBuffer)
	destroyGPUBuffer(s.rankBuffer)
	destroyGPUBuffer(s.mseLUTBuffer)
	destroyGPUBuffer(s.signLUTBuffer)
	destroyGPUBuffer(s.paramsBuffer)
	destroyGPUBuffer(s.outputBuffer)
	destroyGPUBuffer(s.readBuffer)
	destroyGPUBuffer(s.topKIndexBuf)
	destroyGPUBuffer(s.topKScoreBuf)
	destroyGPUBuffer(s.topKIndexRead)
	destroyGPUBuffer(s.topKScoreRead)
	destroyGPUBuffer(s.batchMSEBuf)
	destroyGPUBuffer(s.batchSignBuf)
	destroyGPUBuffer(s.batchParamBuf)
	destroyGPUBuffer(s.batchOutBuf)
	destroyGPUBuffer(s.batchTopKIdx)
	destroyGPUBuffer(s.batchTopKSc)
	destroyGPUBuffer(s.batchTopKRead)
	destroyGPUBuffer(s.batchTopKSRead)
	return nil
}

func (s *GPUPreparedScorer) validatePreparedQuery(pq PreparedQuery) error {
	if s == nil {
		return fmt.Errorf("turboquant: nil GPU prepared scorer")
	}
	if err := ValidatePreparedQuery(s.dim, pq); err != nil {
		return err
	}
	if int(pq.mseBitWidth) != s.mseBitWidth {
		return fmt.Errorf("turboquant: expected prepared query MSE bit width %d, got %d", s.mseBitWidth, pq.mseBitWidth)
	}
	return nil
}

func (s *GPUPreparedScorer) validateTopK(k int) error {
	if !s.topKReady {
		return fmt.Errorf("turboquant: GPU top-k requires tie-break ranks in GPU prepared data")
	}
	if k < 0 {
		return fmt.Errorf("turboquant: GPU top-k must be >= 0")
	}
	if k > GPUPreparedTopKMax {
		return fmt.Errorf("turboquant: GPU top-k %d exceeds max %d", k, GPUPreparedTopKMax)
	}
	if k > s.count {
		return fmt.Errorf("turboquant: GPU top-k %d exceeds corpus size %d", k, s.count)
	}
	return nil
}

func writePreparedQueryBuffers(s *GPUPreparedScorer, pq PreparedQuery, topK int) {
	s.queryMSEUpload = ensureUploadLen(s.queryMSEUpload, len(pq.mseLUT)*4)
	s.querySignUpload = ensureUploadLen(s.querySignUpload, len(pq.signLUT)*4)
	encodeFloat32sBytes(s.queryMSEUpload, pq.mseLUT)
	encodeFloat32sBytes(s.querySignUpload, pq.signLUT)
	writeGPUBuffer(s.queue, s.mseLUTBuffer, s.queryMSEUpload)
	writeGPUBuffer(s.queue, s.signLUTBuffer, s.querySignUpload)
	writeGPUBuffer(s.queue, s.paramsBuffer, s.singleParamsBytes(topK))
}

func (s *GPUPreparedScorer) encodeScorePass(encoder js.Value) {
	pass := encoder.Call("beginComputePass")
	pass.Call("setPipeline", s.scorePipeline)
	pass.Call("setBindGroup", 0, s.scoreBindGroup)
	pass.Call("dispatchWorkgroups", (s.count+63)/64)
	pass.Call("end")
}

func (s *GPUPreparedScorer) encodeTopKPass(encoder js.Value) {
	pass := encoder.Call("beginComputePass")
	pass.Call("setPipeline", s.topKPipeline)
	pass.Call("setBindGroup", 0, s.topKBindGroup)
	pass.Call("dispatchWorkgroups", 1)
	pass.Call("end")
}

func (s *GPUPreparedScorer) encodeBatchScorePass(encoder js.Value, queryCount int) {
	pass := encoder.Call("beginComputePass")
	pass.Call("setPipeline", s.batchScorePipe)
	pass.Call("setBindGroup", 0, s.batchScoreBind)
	pass.Call("dispatchWorkgroups", (s.count+63)/64, queryCount)
	pass.Call("end")
}

func (s *GPUPreparedScorer) encodeBatchTopKPass(encoder js.Value, queryCount int) {
	pass := encoder.Call("beginComputePass")
	pass.Call("setPipeline", s.batchTopKPipe)
	pass.Call("setBindGroup", 0, s.batchTopKBind)
	pass.Call("dispatchWorkgroups", queryCount)
	pass.Call("end")
}

func (s *GPUPreparedScorer) singleParamsBytes(topK int) []byte {
	buf := s.paramsScratch[:]
	binary.LittleEndian.PutUint32(buf[0:4], uint32(s.count))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(s.mseBytes))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(s.signBytes))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(topK))
	binary.LittleEndian.PutUint32(buf[16:20], math.Float32bits(s.qjlScale))
	return buf
}

func (s *GPUPreparedScorer) batchParamsBytes(topK, queryCount int) []byte {
	buf := s.batchParamScratch[:]
	binary.LittleEndian.PutUint32(buf[0:4], uint32(s.count))
	binary.LittleEndian.PutUint32(buf[4:8], uint32(s.mseBytes))
	binary.LittleEndian.PutUint32(buf[8:12], uint32(s.signBytes))
	binary.LittleEndian.PutUint32(buf[12:16], uint32(topK))
	binary.LittleEndian.PutUint32(buf[16:20], uint32(queryCount))
	binary.LittleEndian.PutUint32(buf[20:24], math.Float32bits(s.qjlScale))
	return buf
}

func (s *GPUPreparedScorer) ensureBatchTopKCapacity(queryCount int) error {
	if queryCount <= 0 {
		return nil
	}
	if s.batchCap >= queryCount && !s.batchMSEBuf.IsUndefined() && !s.batchMSEBuf.IsNull() {
		return nil
	}
	destroyGPUBuffer(s.batchMSEBuf)
	destroyGPUBuffer(s.batchSignBuf)
	destroyGPUBuffer(s.batchParamBuf)
	destroyGPUBuffer(s.batchOutBuf)
	destroyGPUBuffer(s.batchTopKIdx)
	destroyGPUBuffer(s.batchTopKSc)
	destroyGPUBuffer(s.batchTopKRead)
	destroyGPUBuffer(s.batchTopKSRead)

	bufferUsage := js.Global().Get("GPUBufferUsage")
	storageUsage := bufferUsage.Get("STORAGE").Int()
	copyDstUsage := bufferUsage.Get("COPY_DST").Int()
	copySrcUsage := bufferUsage.Get("COPY_SRC").Int()
	uniformUsage := bufferUsage.Get("UNIFORM").Int()
	mapReadUsage := bufferUsage.Get("MAP_READ").Int()

	mseLUTBytes := preparedQueryMSELUTLen(s.dim, s.mseBitWidth) * 4
	signLUTBytes := preparedQuerySignLUTLen(s.dim) * 4
	s.batchMSEBuf = createGPUBuffer(s.device, queryCount*mseLUTBytes, storageUsage|copyDstUsage)
	s.batchSignBuf = createGPUBuffer(s.device, queryCount*signLUTBytes, storageUsage|copyDstUsage)
	s.batchParamBuf = createGPUBuffer(s.device, 32, uniformUsage|copyDstUsage)
	s.batchOutBuf = createGPUBuffer(s.device, queryCount*s.outputBytes, storageUsage|copySrcUsage)
	s.batchTopKIdx = createGPUBuffer(s.device, queryCount*GPUPreparedTopKMax*4, storageUsage|copySrcUsage)
	s.batchTopKSc = createGPUBuffer(s.device, queryCount*GPUPreparedTopKMax*4, storageUsage|copySrcUsage)
	s.batchTopKRead = createGPUBuffer(s.device, queryCount*GPUPreparedTopKMax*4, mapReadUsage|copyDstUsage)
	s.batchTopKSRead = createGPUBuffer(s.device, queryCount*GPUPreparedTopKMax*4, mapReadUsage|copyDstUsage)

	scoreLayout := s.batchScorePipe.Call("getBindGroupLayout", 0)
	s.batchScoreBind = s.device.Call("createBindGroup", map[string]any{
		"layout": scoreLayout,
		"entries": []any{
			map[string]any{"binding": 0, "resource": map[string]any{"buffer": s.mseBuffer}},
			map[string]any{"binding": 1, "resource": map[string]any{"buffer": s.signBuffer}},
			map[string]any{"binding": 2, "resource": map[string]any{"buffer": s.resNormBuffer}},
			map[string]any{"binding": 3, "resource": map[string]any{"buffer": s.batchMSEBuf}},
			map[string]any{"binding": 4, "resource": map[string]any{"buffer": s.batchSignBuf}},
			map[string]any{"binding": 5, "resource": map[string]any{"buffer": s.batchParamBuf}},
			map[string]any{"binding": 6, "resource": map[string]any{"buffer": s.batchOutBuf}},
		},
	})

	topKLayout := s.batchTopKPipe.Call("getBindGroupLayout", 0)
	s.batchTopKBind = s.device.Call("createBindGroup", map[string]any{
		"layout": topKLayout,
		"entries": []any{
			map[string]any{"binding": 0, "resource": map[string]any{"buffer": s.batchOutBuf}},
			map[string]any{"binding": 1, "resource": map[string]any{"buffer": s.rankBuffer}},
			map[string]any{"binding": 2, "resource": map[string]any{"buffer": s.batchParamBuf}},
			map[string]any{"binding": 3, "resource": map[string]any{"buffer": s.batchTopKIdx}},
			map[string]any{"binding": 4, "resource": map[string]any{"buffer": s.batchTopKSc}},
		},
	})
	s.batchCap = queryCount
	return nil
}

func writePreparedQueryBatchBuffers(s *GPUPreparedScorer, pqs []PreparedQuery, topK int) {
	mseBytesPerQuery := preparedQueryMSELUTLen(s.dim, s.mseBitWidth) * 4
	signBytesPerQuery := preparedQuerySignLUTLen(s.dim) * 4
	s.batchMSEUpload = ensureUploadLen(s.batchMSEUpload, len(pqs)*mseBytesPerQuery)
	s.batchSignUpload = ensureUploadLen(s.batchSignUpload, len(pqs)*signBytesPerQuery)
	for i := range pqs {
		encodeFloat32sBytes(s.batchMSEUpload[i*mseBytesPerQuery:(i+1)*mseBytesPerQuery], pqs[i].mseLUT)
		encodeFloat32sBytes(s.batchSignUpload[i*signBytesPerQuery:(i+1)*signBytesPerQuery], pqs[i].signLUT)
	}
	writeGPUBuffer(s.queue, s.batchMSEBuf, s.batchMSEUpload)
	writeGPUBuffer(s.queue, s.batchSignBuf, s.batchSignUpload)
	writeGPUBuffer(s.queue, s.batchParamBuf, s.batchParamsBytes(topK, len(pqs)))
}

type jsPromiseResult struct {
	value js.Value
	err   error
}

func awaitJSPromise(p js.Value) (js.Value, error) {
	done := make(chan jsPromiseResult, 1)
	resolve := js.FuncOf(func(this js.Value, args []js.Value) any {
		value := js.Undefined()
		if len(args) != 0 {
			value = args[0]
		}
		done <- jsPromiseResult{value: value}
		return nil
	})
	reject := js.FuncOf(func(this js.Value, args []js.Value) any {
		message := "JavaScript promise rejected"
		if len(args) != 0 {
			message = args[0].String()
		}
		done <- jsPromiseResult{err: fmt.Errorf("%s", message)}
		return nil
	})
	p.Call("then", resolve, reject)
	result := <-done
	resolve.Release()
	reject.Release()
	return result.value, result.err
}

func createGPUBuffer(device js.Value, size, usage int) js.Value {
	return device.Call("createBuffer", map[string]any{
		"size":  maxGPUBufferSize(size),
		"usage": usage,
	})
}

func writeGPUBuffer(queue, buffer js.Value, data []byte) {
	if len(data) == 0 {
		return
	}
	jsData := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(jsData, data)
	queue.Call("writeBuffer", buffer, 0, jsData)
}

func readGPUBufferBytes(buffer js.Value, size int) ([]byte, error) {
	if _, err := awaitJSPromise(buffer.Call("mapAsync", js.Global().Get("GPUMapMode").Get("READ"), 0, size)); err != nil {
		return nil, err
	}
	defer buffer.Call("unmap")

	mapped := buffer.Call("getMappedRange", 0, size)
	jsBytes := js.Global().Get("Uint8Array").New(mapped)
	raw := make([]byte, size)
	js.CopyBytesToGo(raw, jsBytes)
	return raw, nil
}

func destroyGPUBuffer(buffer js.Value) {
	if buffer.IsUndefined() || buffer.IsNull() {
		return
	}
	buffer.Call("destroy")
}

func padBytesToWord(src []byte) []byte {
	if len(src) == 0 {
		return make([]byte, 4)
	}
	if rem := len(src) % 4; rem != 0 {
		dst := make([]byte, len(src)+(4-rem))
		copy(dst, src)
		return dst
	}
	dst := make([]byte, len(src))
	copy(dst, src)
	return dst
}

func float32Bytes(src []float32) []byte {
	if len(src) == 0 {
		return make([]byte, 4)
	}
	dst := make([]byte, len(src)*4)
	for i, value := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(value))
	}
	return dst
}

func ensureUploadLen(dst []byte, n int) []byte {
	if cap(dst) < n {
		return make([]byte, n)
	}
	return dst[:n]
}

func encodeFloat32sBytes(dst []byte, src []float32) {
	for i, value := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], math.Float32bits(value))
	}
}

func uint32Bytes(src []uint32) []byte {
	if len(src) == 0 {
		return nil
	}
	dst := make([]byte, len(src)*4)
	for i, value := range src {
		binary.LittleEndian.PutUint32(dst[i*4:], value)
	}
	return dst
}

func decodeFloat32Bytes(dst []float32, src []byte) {
	for i := range dst {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
}

func decodeUint32Bytes(dst []uint32, src []byte) {
	for i := range dst {
		dst[i] = binary.LittleEndian.Uint32(src[i*4:])
	}
}

func maxGPUBufferSize(size int) int {
	if size <= 0 {
		return 4
	}
	return size
}
