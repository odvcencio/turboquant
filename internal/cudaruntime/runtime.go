//go:build linux && amd64 && cgo && cuda

package cudaruntime

/*
#cgo pkg-config: cuda-13.0 nvrtc-13.0
#include <cuda.h>
#include <nvrtc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	CUdevice dev;
	CUcontext ctx;
	CUmodule module;
	CUfunction score_fn;
	CUfunction topk_fn;
	CUfunction value_sum_fn;
	CUfunction value_sum_batch_fn;
} tq_cuda_runtime;

static const char *tq_cuda_kernel =
"#define TQ_GPU_TOPK_MAX 64u\n"
"__device__ __forceinline__ unsigned int tq_unpack_index(const unsigned char* packed, unsigned int coord, unsigned int bit_width) {\n"
"  switch (bit_width) {\n"
"    case 1u:\n"
"      return (packed[coord >> 3] >> (coord & 7u)) & 1u;\n"
"    case 2u:\n"
"      return (packed[coord >> 2] >> ((coord & 3u) * 2u)) & 3u;\n"
"    case 4u:\n"
"      return (packed[coord >> 1] >> ((coord & 1u) * 4u)) & 15u;\n"
"    case 8u:\n"
"      return packed[coord];\n"
"    default: {\n"
"      unsigned int bit_pos = coord * bit_width;\n"
"      unsigned int byte_idx = bit_pos >> 3;\n"
"      unsigned int shift = bit_pos & 7u;\n"
"      unsigned int value = ((unsigned int)packed[byte_idx]) >> shift;\n"
"      unsigned int have = 8u - shift;\n"
"      while (have < bit_width) {\n"
"        byte_idx += 1u;\n"
"        value |= ((unsigned int)packed[byte_idx]) << have;\n"
"        have += 8u;\n"
"      }\n"
"      return value & ((1u << bit_width) - 1u);\n"
"    }\n"
"  }\n"
"}\n"
"extern \"C\" __global__ void score_kernel(\n"
"    const unsigned char* mse,\n"
"    const unsigned char* signs,\n"
"    const float* res_norms,\n"
"    const float* query_mse,\n"
"    const float* query_sign,\n"
"    float* scores,\n"
"    unsigned int count,\n"
"    unsigned int mse_bytes,\n"
"    unsigned int sign_bytes,\n"
"    unsigned int query_count,\n"
"    float qjl_scale) {\n"
"  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  unsigned int query = blockIdx.y;\n"
"  if (row >= count || query >= query_count) {\n"
"    return;\n"
"  }\n"
"  unsigned int mse_base = row * mse_bytes;\n"
"  unsigned int sign_base = row * sign_bytes;\n"
"  unsigned int query_mse_base = query * mse_bytes * 256u;\n"
"  unsigned int query_sign_base = query * sign_bytes * 256u;\n"
"  float mse_score = 0.0f;\n"
"  for (unsigned int i = 0; i < mse_bytes; ++i) {\n"
"    unsigned int packed = (unsigned int)mse[mse_base + i];\n"
"    mse_score += query_mse[query_mse_base + i * 256u + packed];\n"
"  }\n"
"  float sign_sum = 0.0f;\n"
"  for (unsigned int i = 0; i < sign_bytes; ++i) {\n"
"    unsigned int packed = (unsigned int)signs[sign_base + i];\n"
"    sign_sum += query_sign[query_sign_base + i * 256u + packed];\n"
"  }\n"
"  scores[query * count + row] = mse_score + (qjl_scale * res_norms[row]) * sign_sum;\n"
"}\n"
"__device__ __forceinline__ int tq_better_topk(float score, unsigned int rank, unsigned int idx, float best_score, unsigned int best_rank, unsigned int best_idx) {\n"
"  if (score != best_score) {\n"
"    return score > best_score;\n"
"  }\n"
"  if (rank != best_rank) {\n"
"    return rank < best_rank;\n"
"  }\n"
"  return idx < best_idx;\n"
"}\n"
"extern \"C\" __global__ void topk_kernel(\n"
"    const float* scores,\n"
"    const unsigned int* ranks,\n"
"    unsigned int* out_indices,\n"
"    float* out_scores,\n"
"    unsigned int count,\n"
"    unsigned int k,\n"
"    unsigned int query_count) {\n"
"  unsigned int query = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (query >= query_count || k == 0u || k > TQ_GPU_TOPK_MAX) {\n"
"    return;\n"
"  }\n"
"  float best_scores[TQ_GPU_TOPK_MAX];\n"
"  unsigned int best_ranks[TQ_GPU_TOPK_MAX];\n"
"  unsigned int best_indices[TQ_GPU_TOPK_MAX];\n"
"  for (unsigned int i = 0; i < k; ++i) {\n"
"    best_scores[i] = -3.402823466e+38F;\n"
"    best_ranks[i] = 0xffffffffu;\n"
"    best_indices[i] = 0xffffffffu;\n"
"  }\n"
"  const float* query_scores = scores + query * count;\n"
"  for (unsigned int row = 0; row < count; ++row) {\n"
"    float score = query_scores[row];\n"
"    unsigned int rank = ranks ? ranks[row] : row;\n"
"    unsigned int insert = k;\n"
"    for (unsigned int pos = 0; pos < k; ++pos) {\n"
"      if (tq_better_topk(score, rank, row, best_scores[pos], best_ranks[pos], best_indices[pos])) {\n"
"        insert = pos;\n"
"        break;\n"
"      }\n"
"    }\n"
"    if (insert == k) {\n"
"      continue;\n"
"    }\n"
"    for (unsigned int pos = k - 1u; pos > insert; --pos) {\n"
"      best_scores[pos] = best_scores[pos - 1u];\n"
"      best_ranks[pos] = best_ranks[pos - 1u];\n"
"      best_indices[pos] = best_indices[pos - 1u];\n"
"    }\n"
"    best_scores[insert] = score;\n"
"    best_ranks[insert] = rank;\n"
"    best_indices[insert] = row;\n"
"  }\n"
"  unsigned int out_base = query * k;\n"
"  for (unsigned int i = 0; i < k; ++i) {\n"
"    out_indices[out_base + i] = best_indices[i];\n"
"    out_scores[out_base + i] = best_scores[i];\n"
"  }\n"
"}\n"
"extern \"C\" __global__ void value_sum_kernel(\n"
"    const unsigned char* packed_values,\n"
"    const float* value_norms,\n"
"    const float* centroids,\n"
"    const unsigned int* indices,\n"
"    const float* weights,\n"
"    float* rotated_sum,\n"
"    unsigned int dim,\n"
"    unsigned int bit_width,\n"
"    unsigned int packed_bytes,\n"
"    unsigned int k) {\n"
"  unsigned int coord = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (coord >= dim) {\n"
"    return;\n"
"  }\n"
"  float sum = 0.0f;\n"
"  for (unsigned int i = 0; i < k; ++i) {\n"
"    unsigned int row = indices[i];\n"
"    const unsigned char* packed = packed_values + row * packed_bytes;\n"
"    unsigned int code = tq_unpack_index(packed, coord, bit_width);\n"
"    sum += weights[i] * value_norms[row] * centroids[code];\n"
"  }\n"
"  rotated_sum[coord] = sum;\n"
"}\n"
"extern \"C\" __global__ void value_sum_batch_kernel(\n"
"    const unsigned char* packed_values,\n"
"    const float* value_norms,\n"
"    const float* centroids,\n"
"    const unsigned int* indices,\n"
"    const float* weights,\n"
"    float* rotated_sum,\n"
"    unsigned int dim,\n"
"    unsigned int bit_width,\n"
"    unsigned int packed_bytes,\n"
"    unsigned int k,\n"
"    unsigned int query_count) {\n"
"  unsigned int coord = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  unsigned int query = blockIdx.y;\n"
"  if (coord >= dim || query >= query_count) {\n"
"    return;\n"
"  }\n"
"  const unsigned int* query_indices = indices + query * k;\n"
"  const float* query_weights = weights + query * k;\n"
"  float sum = 0.0f;\n"
"  for (unsigned int i = 0; i < k; ++i) {\n"
"    unsigned int row = query_indices[i];\n"
"    const unsigned char* packed = packed_values + row * packed_bytes;\n"
"    unsigned int code = tq_unpack_index(packed, coord, bit_width);\n"
"    sum += query_weights[i] * value_norms[row] * centroids[code];\n"
"  }\n"
"  rotated_sum[query * dim + coord] = sum;\n"
"}\n";

static void tq_set_err(char **err, const char *prefix, const char *detail) {
	if (!err) {
		return;
	}
	if (!detail) {
		detail = "unknown error";
	}
	size_t n = snprintf(NULL, 0, "%s: %s", prefix, detail);
	char *buf = (char *)malloc(n + 1);
	if (!buf) {
		return;
	}
	snprintf(buf, n + 1, "%s: %s", prefix, detail);
	*err = buf;
}

static void tq_set_cuda_err(char **err, const char *prefix, CUresult res) {
	const char *name = NULL;
	const char *detail = NULL;
	cuGetErrorName(res, &name);
	cuGetErrorString(res, &detail);
	char buf[256];
	snprintf(buf, sizeof(buf), "%s (%s)", name ? name : "CUDA_ERROR", detail ? detail : "unknown");
	tq_set_err(err, prefix, buf);
}

static void tq_set_nvrtc_err(char **err, const char *prefix, nvrtcResult res, nvrtcProgram prog) {
	const char *detail = nvrtcGetErrorString(res);
	size_t log_size = 0;
	if (prog) {
		nvrtcGetProgramLogSize(prog, &log_size);
	}
	if (log_size > 1) {
		char *log_buf = (char *)malloc(log_size);
		if (log_buf) {
			nvrtcGetProgramLog(prog, log_buf);
			size_t n = snprintf(NULL, 0, "%s: %s: %s", prefix, detail ? detail : "NVRTC_ERROR", log_buf);
			char *buf = (char *)malloc(n + 1);
			if (buf) {
				snprintf(buf, n + 1, "%s: %s: %s", prefix, detail ? detail : "NVRTC_ERROR", log_buf);
				*err = buf;
			}
			free(log_buf);
			return;
		}
	}
	tq_set_err(err, prefix, detail ? detail : "NVRTC_ERROR");
}

static int tq_cuda_set_current(tq_cuda_runtime *rt, char **err) {
	CUresult res = cuCtxSetCurrent(rt->ctx);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuCtxSetCurrent failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_runtime_init(tq_cuda_runtime *out, char **err) {
	memset(out, 0, sizeof(*out));
	CUresult res = cuInit(0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuInit failed", res);
		return 1;
	}
	int count = 0;
	res = cuDeviceGetCount(&count);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuDeviceGetCount failed", res);
		return 1;
	}
	if (count <= 0) {
		tq_set_err(err, "CUDA init failed", "no CUDA devices found");
		return 1;
	}
	CUdevice dev;
	res = cuDeviceGet(&dev, 0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuDeviceGet failed", res);
		return 1;
	}
	int major = 0;
	int minor = 0;
	res = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuDeviceGetAttribute major failed", res);
		return 1;
	}
	res = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuDeviceGetAttribute minor failed", res);
		return 1;
	}
	out->dev = dev;
	res = cuDevicePrimaryCtxRetain(&out->ctx, dev);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuDevicePrimaryCtxRetain failed", res);
		return 1;
	}
	if (tq_cuda_set_current(out, err) != 0) {
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}

	nvrtcProgram prog;
	nvrtcResult nres = nvrtcCreateProgram(&prog, tq_cuda_kernel, "turboquant_score.cu", 0, NULL, NULL);
	if (nres != NVRTC_SUCCESS) {
		tq_set_nvrtc_err(err, "nvrtcCreateProgram failed", nres, NULL);
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	char arch[64];
	snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor);
	const char *opts[] = {arch, "--std=c++11", "--use_fast_math"};
	nres = nvrtcCompileProgram(prog, 3, opts);
	if (nres != NVRTC_SUCCESS) {
		tq_set_nvrtc_err(err, "nvrtcCompileProgram failed", nres, prog);
		nvrtcDestroyProgram(&prog);
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	size_t ptx_size = 0;
	nres = nvrtcGetPTXSize(prog, &ptx_size);
	if (nres != NVRTC_SUCCESS) {
		tq_set_nvrtc_err(err, "nvrtcGetPTXSize failed", nres, prog);
		nvrtcDestroyProgram(&prog);
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	char *ptx = (char *)malloc(ptx_size);
	if (!ptx) {
		tq_set_err(err, "malloc failed", "could not allocate PTX buffer");
		nvrtcDestroyProgram(&prog);
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	nres = nvrtcGetPTX(prog, ptx);
	nvrtcDestroyProgram(&prog);
	if (nres != NVRTC_SUCCESS) {
		tq_set_nvrtc_err(err, "nvrtcGetPTX failed", nres, NULL);
		free(ptx);
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	res = cuModuleLoadData(&out->module, ptx);
	free(ptx);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuModuleLoadData failed", res);
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	res = cuModuleGetFunction(&out->score_fn, out->module, "score_kernel");
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuModuleGetFunction failed", res);
		cuModuleUnload(out->module);
		out->module = NULL;
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	res = cuModuleGetFunction(&out->topk_fn, out->module, "topk_kernel");
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuModuleGetFunction topk_kernel failed", res);
		cuModuleUnload(out->module);
		out->module = NULL;
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	res = cuModuleGetFunction(&out->value_sum_fn, out->module, "value_sum_kernel");
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuModuleGetFunction value_sum_kernel failed", res);
		cuModuleUnload(out->module);
		out->module = NULL;
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	res = cuModuleGetFunction(&out->value_sum_batch_fn, out->module, "value_sum_batch_kernel");
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuModuleGetFunction value_sum_batch_kernel failed", res);
		cuModuleUnload(out->module);
		out->module = NULL;
		cuDevicePrimaryCtxRelease(out->dev);
		out->ctx = NULL;
		return 1;
	}
	return 0;
}

static void tq_cuda_runtime_destroy(tq_cuda_runtime *rt) {
	if (!rt) {
		return;
	}
	if (rt->module) {
		cuModuleUnload(rt->module);
		rt->module = NULL;
	}
	if (rt->ctx) {
		cuDevicePrimaryCtxRelease(rt->dev);
		rt->ctx = NULL;
	}
}

static int tq_cuda_alloc(tq_cuda_runtime *rt, CUdeviceptr *out, size_t size, char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	if (size == 0) {
		size = 1;
	}
	CUresult res = cuMemAlloc(out, size);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemAlloc failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_free(tq_cuda_runtime *rt, CUdeviceptr ptr, char **err) {
	if (!ptr) {
		return 0;
	}
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	CUresult res = cuMemFree(ptr);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemFree failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_alloc_host(tq_cuda_runtime *rt, void **out, size_t size, char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	if (size == 0) {
		size = 1;
	}
	CUresult res = cuMemAllocHost(out, size);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemAllocHost failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_free_host(tq_cuda_runtime *rt, void *ptr, char **err) {
	if (!ptr) {
		return 0;
	}
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	CUresult res = cuMemFreeHost(ptr);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemFreeHost failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_htod(tq_cuda_runtime *rt, CUdeviceptr dst, const void *src, size_t size, char **err) {
	if (size == 0) {
		return 0;
	}
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	CUresult res = cuMemcpyHtoD(dst, src, size);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyHtoD failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_dtoh(tq_cuda_runtime *rt, void *dst, CUdeviceptr src, size_t size, char **err) {
	if (size == 0) {
		return 0;
	}
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	CUresult res = cuMemcpyDtoH(dst, src, size);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyDtoH failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_launch_score(
	tq_cuda_runtime *rt,
	CUdeviceptr mse,
	CUdeviceptr signs,
	CUdeviceptr res_norms,
	CUdeviceptr query_mse,
	CUdeviceptr query_sign,
	CUdeviceptr scores,
	unsigned int count,
	unsigned int mse_bytes,
	unsigned int sign_bytes,
	unsigned int query_count,
	float qjl_scale,
	char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	void *args[] = {
		&mse,
		&signs,
		&res_norms,
		&query_mse,
		&query_sign,
		&scores,
		&count,
		&mse_bytes,
		&sign_bytes,
		&query_count,
		&qjl_scale,
	};
	unsigned int block_x = 256;
	unsigned int grid_x = (count + block_x - 1u) / block_x;
	unsigned int grid_y = query_count;
	CUresult res = cuLaunchKernel(rt->score_fn, grid_x, grid_y, 1, block_x, 1, 1, 0, 0, args, 0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuLaunchKernel failed", res);
		return 1;
	}
	res = cuCtxSynchronize();
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuCtxSynchronize failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_launch_score_to_host(
	tq_cuda_runtime *rt,
	CUdeviceptr mse,
	CUdeviceptr signs,
	CUdeviceptr res_norms,
	CUdeviceptr query_mse,
	CUdeviceptr query_sign,
	CUdeviceptr scores,
	void *host_scores,
	unsigned int count,
	unsigned int mse_bytes,
	unsigned int sign_bytes,
	unsigned int query_count,
	float qjl_scale,
	char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	void *args[] = {
		&mse,
		&signs,
		&res_norms,
		&query_mse,
		&query_sign,
		&scores,
		&count,
		&mse_bytes,
		&sign_bytes,
		&query_count,
		&qjl_scale,
	};
	unsigned int block_x = 256;
	unsigned int grid_x = (count + block_x - 1u) / block_x;
	unsigned int grid_y = query_count;
	CUresult res = cuLaunchKernel(rt->score_fn, grid_x, grid_y, 1, block_x, 1, 1, 0, 0, args, 0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuLaunchKernel failed", res);
		return 1;
	}
	res = cuCtxSynchronize();
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuCtxSynchronize failed", res);
		return 1;
	}
	res = cuMemcpyDtoH(host_scores, scores, (size_t)count * (size_t)query_count * 4u);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyDtoH failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_launch_value_sum(
	tq_cuda_runtime *rt,
	CUdeviceptr packed_values,
	CUdeviceptr value_norms,
	CUdeviceptr centroids,
	CUdeviceptr indices,
	CUdeviceptr weights,
	CUdeviceptr rotated_sum,
	unsigned int dim,
	unsigned int bit_width,
	unsigned int packed_bytes,
	unsigned int k,
	char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	void *args[] = {
		&packed_values,
		&value_norms,
		&centroids,
		&indices,
		&weights,
		&rotated_sum,
		&dim,
		&bit_width,
		&packed_bytes,
		&k,
	};
	unsigned int block_x = 256;
	unsigned int grid_x = (dim + block_x - 1u) / block_x;
	CUresult res = cuLaunchKernel(rt->value_sum_fn, grid_x, 1, 1, block_x, 1, 1, 0, 0, args, 0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuLaunchKernel value_sum_kernel failed", res);
		return 1;
	}
	res = cuCtxSynchronize();
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuCtxSynchronize failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_launch_value_sum_to_host(
	tq_cuda_runtime *rt,
	CUdeviceptr packed_values,
	CUdeviceptr value_norms,
	CUdeviceptr centroids,
	CUdeviceptr indices,
	CUdeviceptr weights,
	CUdeviceptr rotated_sum,
	const void *host_indices,
	const void *host_weights,
	void *host_rotated,
	unsigned int dim,
	unsigned int bit_width,
	unsigned int packed_bytes,
	unsigned int k,
	char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	size_t shortlist_bytes = (size_t)k * 4u;
	CUresult res = cuMemcpyHtoD(indices, host_indices, shortlist_bytes);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyHtoD indices failed", res);
		return 1;
	}
	res = cuMemcpyHtoD(weights, host_weights, shortlist_bytes);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyHtoD weights failed", res);
		return 1;
	}
	void *args[] = {
		&packed_values,
		&value_norms,
		&centroids,
		&indices,
		&weights,
		&rotated_sum,
		&dim,
		&bit_width,
		&packed_bytes,
		&k,
	};
	unsigned int block_x = 256;
	unsigned int grid_x = (dim + block_x - 1u) / block_x;
	res = cuLaunchKernel(rt->value_sum_fn, grid_x, 1, 1, block_x, 1, 1, 0, 0, args, 0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuLaunchKernel value_sum_kernel failed", res);
		return 1;
	}
	res = cuCtxSynchronize();
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuCtxSynchronize failed", res);
		return 1;
	}
	res = cuMemcpyDtoH(host_rotated, rotated_sum, (size_t)dim * 4u);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyDtoH rotated_sum failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_launch_value_sum_batch(
	tq_cuda_runtime *rt,
	CUdeviceptr packed_values,
	CUdeviceptr value_norms,
	CUdeviceptr centroids,
	CUdeviceptr indices,
	CUdeviceptr weights,
	CUdeviceptr rotated_sum,
	unsigned int dim,
	unsigned int bit_width,
	unsigned int packed_bytes,
	unsigned int k,
	unsigned int query_count,
	char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	void *args[] = {
		&packed_values,
		&value_norms,
		&centroids,
		&indices,
		&weights,
		&rotated_sum,
		&dim,
		&bit_width,
		&packed_bytes,
		&k,
		&query_count,
	};
	unsigned int block_x = 256;
	unsigned int grid_x = (dim + block_x - 1u) / block_x;
	unsigned int grid_y = query_count;
	CUresult res = cuLaunchKernel(rt->value_sum_batch_fn, grid_x, grid_y, 1, block_x, 1, 1, 0, 0, args, 0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuLaunchKernel value_sum_batch_kernel failed", res);
		return 1;
	}
	res = cuCtxSynchronize();
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuCtxSynchronize failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_launch_value_sum_batch_to_host(
	tq_cuda_runtime *rt,
	CUdeviceptr packed_values,
	CUdeviceptr value_norms,
	CUdeviceptr centroids,
	CUdeviceptr indices,
	CUdeviceptr weights,
	CUdeviceptr rotated_sum,
	const void *host_indices,
	const void *host_weights,
	void *host_rotated,
	unsigned int dim,
	unsigned int bit_width,
	unsigned int packed_bytes,
	unsigned int k,
	unsigned int query_count,
	char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	size_t shortlist_bytes = (size_t)k * (size_t)query_count * 4u;
	CUresult res = cuMemcpyHtoD(indices, host_indices, shortlist_bytes);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyHtoD indices failed", res);
		return 1;
	}
	res = cuMemcpyHtoD(weights, host_weights, shortlist_bytes);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyHtoD weights failed", res);
		return 1;
	}
	void *args[] = {
		&packed_values,
		&value_norms,
		&centroids,
		&indices,
		&weights,
		&rotated_sum,
		&dim,
		&bit_width,
		&packed_bytes,
		&k,
		&query_count,
	};
	unsigned int block_x = 256;
	unsigned int grid_x = (dim + block_x - 1u) / block_x;
	unsigned int grid_y = query_count;
	res = cuLaunchKernel(rt->value_sum_batch_fn, grid_x, grid_y, 1, block_x, 1, 1, 0, 0, args, 0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuLaunchKernel value_sum_batch_kernel failed", res);
		return 1;
	}
	res = cuCtxSynchronize();
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuCtxSynchronize failed", res);
		return 1;
	}
	res = cuMemcpyDtoH(host_rotated, rotated_sum, (size_t)dim * (size_t)query_count * 4u);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuMemcpyDtoH rotated_sum failed", res);
		return 1;
	}
	return 0;
}

static int tq_cuda_launch_topk(
	tq_cuda_runtime *rt,
	CUdeviceptr scores,
	CUdeviceptr ranks,
	CUdeviceptr out_indices,
	CUdeviceptr out_scores,
	unsigned int count,
	unsigned int k,
	unsigned int query_count,
	char **err) {
	if (tq_cuda_set_current(rt, err) != 0) {
		return 1;
	}
	void *args[] = {
		&scores,
		&ranks,
		&out_indices,
		&out_scores,
		&count,
		&k,
		&query_count,
	};
	unsigned int block_x = 64;
	unsigned int grid_x = (query_count + block_x - 1u) / block_x;
	CUresult res = cuLaunchKernel(rt->topk_fn, grid_x, 1, 1, block_x, 1, 1, 0, 0, args, 0);
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuLaunchKernel topk_kernel failed", res);
		return 1;
	}
	res = cuCtxSynchronize();
	if (res != CUDA_SUCCESS) {
		tq_set_cuda_err(err, "cuCtxSynchronize failed", res);
		return 1;
	}
	return 0;
}

*/
import "C"

import (
	"fmt"
	"unsafe"
)

type DevicePtr uintptr

type Runtime struct {
	rt C.tq_cuda_runtime
}

func New() (*Runtime, error) {
	r := &Runtime{}
	var cerr *C.char
	if C.tq_cuda_runtime_init(&r.rt, &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return nil, fmt.Errorf("%s", C.GoString(cerr))
	}
	return r, nil
}

func (r *Runtime) Close() {
	if r == nil {
		return
	}
	C.tq_cuda_runtime_destroy(&r.rt)
}

func (r *Runtime) Alloc(size int) (DevicePtr, error) {
	var ptr C.CUdeviceptr
	var cerr *C.char
	if C.tq_cuda_alloc(&r.rt, &ptr, C.size_t(size), &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return 0, fmt.Errorf("%s", C.GoString(cerr))
	}
	return DevicePtr(ptr), nil
}

func (r *Runtime) AllocHost(size int) (unsafe.Pointer, error) {
	var cerr *C.char
	var ptr unsafe.Pointer
	if C.tq_cuda_alloc_host(&r.rt, (*unsafe.Pointer)(&ptr), C.size_t(size), &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return nil, fmt.Errorf("%s", C.GoString(cerr))
	}
	return ptr, nil
}

func (r *Runtime) Free(ptr DevicePtr) error {
	if ptr == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_free(&r.rt, C.CUdeviceptr(ptr), &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) FreeHost(ptr unsafe.Pointer) error {
	var cerr *C.char
	if C.tq_cuda_free_host(&r.rt, ptr, &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) HtoD(dst DevicePtr, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_htod(&r.rt, C.CUdeviceptr(dst), unsafe.Pointer(&src[0]), C.size_t(len(src)), &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) HtoDFloat32(dst DevicePtr, src []float32) error {
	if len(src) == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_htod(&r.rt, C.CUdeviceptr(dst), unsafe.Pointer(&src[0]), C.size_t(len(src)*4), &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) DtoHBytes(dst []byte, src DevicePtr) error {
	if len(dst) == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_dtoh(&r.rt, unsafe.Pointer(&dst[0]), C.CUdeviceptr(src), C.size_t(len(dst)), &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) DtoHFloat32(dst []float32, src DevicePtr) error {
	if len(dst) == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_dtoh(&r.rt, unsafe.Pointer(&dst[0]), C.CUdeviceptr(src), C.size_t(len(dst)*4), &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) DtoHUint32(dst []uint32, src DevicePtr) error {
	if len(dst) == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_dtoh(&r.rt, unsafe.Pointer(&dst[0]), C.CUdeviceptr(src), C.size_t(len(dst)*4), &cerr) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) LaunchScore(mse, signs, resNorms, queryMSE, querySign, scores DevicePtr, count, mseBytes, signBytes, queryCount int, qjlScale float32) error {
	var cerr *C.char
	if C.tq_cuda_launch_score(
		&r.rt,
		C.CUdeviceptr(mse),
		C.CUdeviceptr(signs),
		C.CUdeviceptr(resNorms),
		C.CUdeviceptr(queryMSE),
		C.CUdeviceptr(querySign),
		C.CUdeviceptr(scores),
		C.uint(count),
		C.uint(mseBytes),
		C.uint(signBytes),
		C.uint(queryCount),
		C.float(qjlScale),
		&cerr,
	) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) LaunchScoreToHost(dst []float32, mse, signs, resNorms, queryMSE, querySign, scores DevicePtr, count, mseBytes, signBytes, queryCount int, qjlScale float32) error {
	if len(dst) == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_launch_score_to_host(
		&r.rt,
		C.CUdeviceptr(mse),
		C.CUdeviceptr(signs),
		C.CUdeviceptr(resNorms),
		C.CUdeviceptr(queryMSE),
		C.CUdeviceptr(querySign),
		C.CUdeviceptr(scores),
		unsafe.Pointer(&dst[0]),
		C.uint(count),
		C.uint(mseBytes),
		C.uint(signBytes),
		C.uint(queryCount),
		C.float(qjlScale),
		&cerr,
	) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) LaunchValueSum(packedValues, valueNorms, centroids, indices, weights, rotatedSum DevicePtr, dim, bitWidth, packedBytes, k int) error {
	var cerr *C.char
	if C.tq_cuda_launch_value_sum(
		&r.rt,
		C.CUdeviceptr(packedValues),
		C.CUdeviceptr(valueNorms),
		C.CUdeviceptr(centroids),
		C.CUdeviceptr(indices),
		C.CUdeviceptr(weights),
		C.CUdeviceptr(rotatedSum),
		C.uint(dim),
		C.uint(bitWidth),
		C.uint(packedBytes),
		C.uint(k),
		&cerr,
	) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) LaunchValueSumToHost(dst []float32, packedValues, valueNorms, centroids, indices, weights, rotatedSum DevicePtr, hostIndices []uint32, hostWeights []float32, dim, bitWidth, packedBytes, k int) error {
	if len(dst) == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_launch_value_sum_to_host(
		&r.rt,
		C.CUdeviceptr(packedValues),
		C.CUdeviceptr(valueNorms),
		C.CUdeviceptr(centroids),
		C.CUdeviceptr(indices),
		C.CUdeviceptr(weights),
		C.CUdeviceptr(rotatedSum),
		unsafe.Pointer(&hostIndices[0]),
		unsafe.Pointer(&hostWeights[0]),
		unsafe.Pointer(&dst[0]),
		C.uint(dim),
		C.uint(bitWidth),
		C.uint(packedBytes),
		C.uint(k),
		&cerr,
	) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) LaunchValueSumBatch(packedValues, valueNorms, centroids, indices, weights, rotatedSum DevicePtr, dim, bitWidth, packedBytes, k, queryCount int) error {
	var cerr *C.char
	if C.tq_cuda_launch_value_sum_batch(
		&r.rt,
		C.CUdeviceptr(packedValues),
		C.CUdeviceptr(valueNorms),
		C.CUdeviceptr(centroids),
		C.CUdeviceptr(indices),
		C.CUdeviceptr(weights),
		C.CUdeviceptr(rotatedSum),
		C.uint(dim),
		C.uint(bitWidth),
		C.uint(packedBytes),
		C.uint(k),
		C.uint(queryCount),
		&cerr,
	) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) LaunchValueSumBatchToHost(dst []float32, packedValues, valueNorms, centroids, indices, weights, rotatedSum DevicePtr, hostIndices []uint32, hostWeights []float32, dim, bitWidth, packedBytes, k, queryCount int) error {
	if len(dst) == 0 {
		return nil
	}
	var cerr *C.char
	if C.tq_cuda_launch_value_sum_batch_to_host(
		&r.rt,
		C.CUdeviceptr(packedValues),
		C.CUdeviceptr(valueNorms),
		C.CUdeviceptr(centroids),
		C.CUdeviceptr(indices),
		C.CUdeviceptr(weights),
		C.CUdeviceptr(rotatedSum),
		unsafe.Pointer(&hostIndices[0]),
		unsafe.Pointer(&hostWeights[0]),
		unsafe.Pointer(&dst[0]),
		C.uint(dim),
		C.uint(bitWidth),
		C.uint(packedBytes),
		C.uint(k),
		C.uint(queryCount),
		&cerr,
	) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}

func (r *Runtime) LaunchTopK(scores, ranks, outIndices, outScores DevicePtr, count, k, queryCount int) error {
	var cerr *C.char
	if C.tq_cuda_launch_topk(
		&r.rt,
		C.CUdeviceptr(scores),
		C.CUdeviceptr(ranks),
		C.CUdeviceptr(outIndices),
		C.CUdeviceptr(outScores),
		C.uint(count),
		C.uint(k),
		C.uint(queryCount),
		&cerr,
	) != 0 {
		defer C.free(unsafe.Pointer(cerr))
		return fmt.Errorf("%s", C.GoString(cerr))
	}
	return nil
}
