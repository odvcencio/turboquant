#!/usr/bin/env python3
"""Export transformer-layer KV captures from a Hugging Face Llama-style model.

This script is intentionally optional and kept outside the Go module. It
requires Python packages such as `torch` and `transformers`, but the core
TurboQuant library remains zero-dependency.

The output matches the JSON shape consumed by `cmd/tqkveval` and `cmd/tqkvsweep`.

Memory-safe: streams JSON output, captures only requested Q/K/V projections via
forward hooks, exports native KV head counts for GQA models, and frees
intermediate tensors aggressively.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--prompt", action="append", help="Prompt text to evaluate; repeat for multiple prompts")
    parser.add_argument("--prompt-file", help="Optional UTF-8 text file with one prompt per non-empty line")
    parser.add_argument(
        "--layer",
        required=True,
        help=(
            "Layer selection. Supports comma-separated indices, negative-from-end, "
            "all, or spread:N/auto:N for evenly spaced layers"
        ),
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument(
        "--token-index",
        default="last",
        help=(
            "Prompt token index to export. Supports integers, negative-from-end, "
            "named positions like first/middle/last, percentages like 25%%, or a comma-separated list"
        ),
    )
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu, cuda, mps")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype; auto uses float16 off-CPU and float32 on CPU",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from local cache only and skip Hugging Face network fetches",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to Transformers")
    return parser.parse_args()


def load_optional_deps():
    try:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "export_hf_llama_kv_capture.py requires optional dependencies: pip install torch transformers"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(torch, name: str, device: str):
    if name == "auto":
        if device.startswith("cpu"):
            return torch.float32
        return torch.float16
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def get_decoder_layers(model):
    candidates = [
        ("layers", getattr(model, "layers", None)),
        ("model.layers", getattr(getattr(model, "model", None), "layers", None)),
        ("transformer.h", getattr(getattr(model, "transformer", None), "h", None)),
    ]
    for _, layers in candidates:
        if layers is not None:
            return layers
    raise SystemExit("could not locate decoder layers on the model; this script currently supports Llama-style HF models")


def resolve_attention_module(layer):
    candidates = ["self_attn", "attention", "attn"]
    for name in candidates:
        module = getattr(layer, name, None)
        if module is not None:
            return module
    raise SystemExit("could not locate an attention module on the selected layer")


def resolve_attention_shape(attn, config):
    num_heads = getattr(attn, "num_heads", None)
    if not isinstance(num_heads, int):
        num_heads = getattr(attn, "num_attention_heads", None)
    if not isinstance(num_heads, int):
        num_heads = getattr(config, "num_attention_heads", None)

    num_kv_heads = getattr(attn, "num_key_value_heads", None)
    if not isinstance(num_kv_heads, int):
        num_kv_heads = getattr(attn, "num_kv_heads", None)
    if not isinstance(num_kv_heads, int):
        num_kv_heads = getattr(config, "num_key_value_heads", None)
    if not isinstance(num_kv_heads, int):
        num_kv_heads = num_heads

    head_dim = getattr(attn, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = getattr(config, "head_dim", None)
    if not isinstance(head_dim, int):
        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, int) and isinstance(num_heads, int) and num_heads > 0:
            head_dim = hidden_size // num_heads

    return num_heads, num_kv_heads, head_dim


def resolve_rotary_layer_type(attn, model_body):
    rotary = getattr(model_body, "rotary_emb", None)
    layer_idx = getattr(attn, "layer_idx", None)
    layer_types = getattr(rotary, "layer_types", None)
    if isinstance(layer_idx, int) and isinstance(layer_types, (list, tuple)) and 0 <= layer_idx < len(layer_types):
        layer_type = layer_types[layer_idx]
        if isinstance(layer_type, str) and layer_type:
            return layer_type
    is_sliding = getattr(attn, "is_sliding", None)
    if isinstance(is_sliding, bool):
        return "sliding_attention" if is_sliding else "full_attention"
    return None


def compute_rotary_embeddings(attn, model_body, value_states, position_ids):
    rotary = getattr(attn, "rotary_emb", None)
    if rotary is None:
        rotary = getattr(model_body, "rotary_emb", None)
    if rotary is None:
        return None, None
    layer_type = resolve_rotary_layer_type(attn, model_body)
    attempts = [
        lambda: rotary(value_states, position_ids, layer_type=layer_type),
        lambda: rotary(value_states, position_ids=position_ids, layer_type=layer_type),
        lambda: rotary(value_states, position_ids),
        lambda: rotary(value_states, position_ids=position_ids),
        lambda: rotary(value_states, seq_len=value_states.shape[-2]),
        lambda: rotary(value_states),
    ]
    for fn in attempts:
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) == 2:
                return out
        except TypeError:
            continue
    raise SystemExit("could not compute rotary embeddings for the selected attention module")


def apply_rotary(attn, query_states, key_states, cos, sin, position_ids):
    helper = getattr(attn, "rotary_fn", None)
    if callable(helper):
        attempts = [
            lambda: helper(query_states, key_states, cos, sin),
            lambda: helper(query_states, key_states, cos, sin, unsqueeze_dim=1),
            lambda: helper(query_states, key_states, cos, sin, 1),
        ]
        for fn in attempts:
            try:
                out = fn()
                if isinstance(out, tuple) and len(out) == 2:
                    return out
            except TypeError:
                continue
    module = importlib.import_module(attn.__class__.__module__)
    helper = getattr(module, "apply_rotary_pos_emb", None)
    if helper is None:
        return query_states, key_states
    attempts = [
        lambda: helper(query_states, key_states, cos, sin, position_ids),
        lambda: helper(query_states, key_states, cos, sin, position_ids=position_ids),
        lambda: helper(query_states, key_states, cos, sin),
    ]
    for fn in attempts:
        try:
            out = fn()
            if isinstance(out, tuple) and len(out) == 2:
                return out
        except TypeError:
            continue
    raise SystemExit("could not apply rotary embeddings for the selected attention module")

def select_relative_index(seq_len: int, fraction: float) -> int:
    if seq_len <= 0:
        raise SystemExit("sequence length must be positive")
    if fraction < 0 or fraction > 1:
        raise SystemExit(f"relative index fraction must be within [0, 1], got {fraction}")
    return int(round((seq_len - 1) * fraction))


def select_token_index(raw_index: str, seq_len: int) -> int:
    raw_index = raw_index.strip().lower()
    if raw_index == "last":
        return seq_len - 1
    if raw_index in ("first", "start"):
        return 0
    if raw_index in ("middle", "mid", "center", "centre"):
        return select_relative_index(seq_len, 0.5)
    if raw_index in ("quarter", "q1"):
        return select_relative_index(seq_len, 0.25)
    if raw_index in ("three_quarter", "three-quarter", "q3"):
        return select_relative_index(seq_len, 0.75)
    if raw_index.endswith("%"):
        try:
            pct = float(raw_index[:-1])
        except ValueError as exc:
            raise SystemExit(f"token index {raw_index!r} is not a valid percentage") from exc
        return select_relative_index(seq_len, pct / 100.0)
    idx = int(raw_index)
    if idx < 0:
        idx += seq_len
    if idx < 0 or idx >= seq_len:
        raise SystemExit(f"token index {raw_index!r} is out of bounds for sequence length {seq_len}")
    return idx


def select_layer_index(raw_index: str, layer_count: int) -> int:
    idx = int(raw_index)
    if idx < 0:
        idx += layer_count
    if idx < 0 or idx >= layer_count:
        raise SystemExit(f"layer index {raw_index!r} is out of bounds for model with {layer_count} layers")
    return idx


def resolve_layer_indices(raw: str, layer_count: int) -> list[int]:
    spec = raw.strip().lower()
    if spec == "all":
        return list(range(layer_count))
    if spec.startswith("spread:") or spec.startswith("auto:"):
        _, _, count_raw = spec.partition(":")
        try:
            count = int(count_raw)
        except ValueError as exc:
            raise SystemExit(f"layer spec {raw!r} has an invalid spread count") from exc
        if count <= 0:
            raise SystemExit(f"layer spec {raw!r} must request at least one layer")
        if count >= layer_count:
            return list(range(layer_count))
        if count == 1:
            return [layer_count // 2]
        indices = []
        seen = set()
        for i in range(count):
            idx = int(round(i * (layer_count - 1) / (count - 1)))
            if idx in seen:
                continue
            seen.add(idx)
            indices.append(idx)
        if len(indices) < count:
            for idx in range(layer_count):
                if idx in seen:
                    continue
                indices.append(idx)
                if len(indices) == count:
                    break
        return indices

    indices = []
    seen = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = select_layer_index(part, layer_count)
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    if not indices:
        raise SystemExit("expected at least one layer value")
    return indices


def parse_csv_ints(raw: str, label: str) -> list[int]:
    values = []
    seen = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise SystemExit(f"{label} value {part!r} is not an integer") from exc
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise SystemExit(f"expected at least one {label} value")
    return values


def load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        for line in prompt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise SystemExit("at least one --prompt or --prompt-file entry is required")
    return prompts


def resolve_token_targets(raw: str, seq_len: int) -> list[tuple[str, int]]:
    targets = []
    seen = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        idx = select_token_index(part, seq_len)
        if idx in seen:
            continue
        seen.add(idx)
        targets.append((part, idx))
    if not targets:
        raise SystemExit("expected at least one token index")
    return targets


def clear_device_cache(torch):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except RuntimeError:
            pass


def is_oom_error(torch, exc: BaseException) -> bool:
    cuda_oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    return "out of memory" in str(exc).lower()


def is_gated_repo_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "gated repo" in text or "401" in text or "unauthorized" in text


def projection_hook_store(layer_index: int, proj_name: str, captured: dict[int, dict[str, object]]):
    def hook(_module, _inputs, output):
        captured[layer_index][proj_name] = output.detach().cpu()

    return hook


def capture_projection_outputs(torch, model_body, decoder_layers, layers_to_export, tokens):
    captured = {idx: {} for idx in layers_to_export}
    handles = []
    try:
        for layer_index in layers_to_export:
            attn = resolve_attention_module(decoder_layers[layer_index])
            for proj_name in ("q_proj", "k_proj", "v_proj"):
                proj = getattr(attn, proj_name, None)
                if proj is None:
                    raise SystemExit(f"attention layer {layer_index} is missing {proj_name}")
                handles.append(proj.register_forward_hook(projection_hook_store(layer_index, proj_name, captured)))
        with torch.no_grad():
            model_body(**tokens, output_hidden_states=False, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    for layer_index in layers_to_export:
        missing = [name for name in ("q_proj", "k_proj", "v_proj") if name not in captured[layer_index]]
        if missing:
            raise SystemExit(f"failed to capture {', '.join(missing)} for layer {layer_index}")
    return captured


def write_float_array(f, tensor, chunk_size=4096):
    """Write a 1D tensor as a JSON array, streaming in chunks to limit memory.

    Avoids materializing the full Python list: only one chunk of floats is alive
    at a time (~32KB), regardless of tensor size.
    """
    flat = tensor.contiguous().view(-1)
    n = flat.shape[0]
    f.write("[")
    pos = 0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = flat[start:end]
        for val in chunk.float().cpu().tolist():
            if pos > 0:
                f.write(", ")
            f.write(repr(val))
            pos += 1
        del chunk
    f.write("]")


def stream_sample(f, meta: dict, query, keys, values, *, first: bool):
    """Write one capture sample, streaming large float arrays directly to disk."""
    if not first:
        f.write(",\n")
    f.write("    {\n")
    for key, value in meta.items():
        f.write(f"      {json.dumps(key)}: {json.dumps(value)},\n")
    f.write('      "query": ')
    write_float_array(f, query)
    f.write(',\n      "keys": ')
    write_float_array(f, keys)
    f.write(',\n      "values": ')
    write_float_array(f, values)
    f.write("\n    }")
    f.flush()


def main() -> int:
    args = parse_args()
    torch, AutoModelForCausalLM, AutoTokenizer = load_optional_deps()
    prompts = load_prompts(args)
    resolved_dtype = resolve_dtype(torch, args.dtype, args.device)

    print(f"Loading {args.model} ({resolved_dtype}) on {args.device}...", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=resolved_dtype,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
        )
        model.to(args.device)
    except OSError as exc:
        if is_gated_repo_error(exc):
            raise SystemExit(
                "model access was denied by Hugging Face; provide HF_TOKEN/HUGGING_FACE_HUB_TOKEN, "
                "place a token file in ~/.cache/huggingface/token or ~/.config/huggingface/token, "
                "or rerun with --local-files-only if the full snapshot is already cached locally"
            ) from exc
        raise
    except RuntimeError as exc:
        clear_device_cache(torch)
        if is_oom_error(torch, exc):
            raise SystemExit(
                "out of memory while loading the model; try --dtype float16, a shorter prompt, or a smaller model"
            ) from exc
        raise
    model.eval()
    model_body = getattr(model, "model", model)

    decoder_layers = get_decoder_layers(model_body)
    layers_to_export = resolve_layer_indices(args.layer, len(decoder_layers))

    model_name = Path(str(args.model)).name or str(args.model)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "model": {json.dumps(str(args.model))},\n')
        f.write(f'  "prompts": {json.dumps(prompts)},\n')
        f.write(f'  "layer_spec": {json.dumps(args.layer)},\n')
        f.write(f'  "layers": {json.dumps(layers_to_export)},\n')
        f.write(f'  "token_index_spec": {json.dumps(args.token_index)},\n')
        f.write('  "samples": [\n')

        first_sample = True

        for prompt_index, prompt in enumerate(prompts):
            print(f"  prompt {prompt_index + 1}/{len(prompts)} ({len(prompt)} chars)...", file=sys.stderr)
            tokens = tokenizer(prompt, return_tensors="pt")
            tokens = {name: value.to(args.device) for name, value in tokens.items()}

            try:
                captured_projections = capture_projection_outputs(torch, model_body, decoder_layers, layers_to_export, tokens)
            except RuntimeError as exc:
                clear_device_cache(torch)
                if is_oom_error(torch, exc):
                    raise SystemExit(
                        f"out of memory while running prompt {prompt_index}; try --dtype float16 or shorter prompts"
                    ) from exc
                raise
            gc.collect()
            clear_device_cache(torch)

            token_targets = None

            for layer_index in layers_to_export:
                attn = resolve_attention_module(decoder_layers[layer_index])
                layer_capture = captured_projections.pop(layer_index)
                query_states = None
                key_states = None
                value_states = None
                try:
                    query_states = layer_capture["q_proj"].to(args.device)
                    key_states = layer_capture["k_proj"].to(args.device)
                    value_states = layer_capture["v_proj"].to(args.device)

                    batch_size, seq_len, _ = query_states.shape
                    if batch_size != 1:
                        raise SystemExit("only batch size 1 is supported")

                    num_heads, num_kv_heads, head_dim = resolve_attention_shape(attn, model.config)
                    if not isinstance(num_heads, int) or not isinstance(num_kv_heads, int) or not isinstance(head_dim, int):
                        raise SystemExit("attention module is missing num_heads, num_key_value_heads, or head_dim")
                    if num_heads % num_kv_heads != 0:
                        raise SystemExit(
                            f"num_heads={num_heads} is not divisible by num_key_value_heads={num_kv_heads}"
                        )

                    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
                    value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

                    position_ids = tokens.get("position_ids")
                    if position_ids is None:
                        position_ids = torch.arange(seq_len, device=query_states.device).unsqueeze(0)
                    else:
                        position_ids = position_ids.to(query_states.device)
                    cos, sin = compute_rotary_embeddings(attn, model_body, value_states, position_ids)
                    if cos is not None and sin is not None:
                        query_states, key_states = apply_rotary(attn, query_states, key_states, cos, sin, position_ids)
                    del cos, sin, position_ids

                    if token_targets is None:
                        token_targets = resolve_token_targets(args.token_index, seq_len)
                    for token_position, token_index in token_targets:
                        q = query_states[0, :, token_index, :].contiguous().reshape(-1)
                        k = (
                            key_states[0, :, : token_index + 1, :]
                            .permute(1, 0, 2)
                            .contiguous()
                            .reshape(-1)
                        )
                        v = (
                            value_states[0, :, : token_index + 1, :]
                            .permute(1, 0, 2)
                            .contiguous()
                            .reshape(-1)
                        )

                        meta = {
                            "name": f"{model_name}-prompt{prompt_index}-layer{layer_index}-token{token_index}",
                            "model": str(args.model),
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "layer": layer_index,
                            "token_index": token_index,
                            "token_position": token_position,
                            "sequence_length": seq_len,
                            "heads": num_heads,
                            "kv_heads": num_kv_heads,
                            "head_dim": head_dim,
                            "tokens": token_index + 1,
                            "query_scale": 1.0 / math.sqrt(head_dim),
                        }
                        stream_sample(f, meta, q, k, v, first=first_sample)
                        first_sample = False
                        del q, k, v
                        print(f"    layer {layer_index} token {token_index}: written", file=sys.stderr)
                except RuntimeError as exc:
                    clear_device_cache(torch)
                    if is_oom_error(torch, exc):
                        raise SystemExit(
                            f"out of memory while exporting prompt {prompt_index} layer {layer_index}; "
                            "try --dtype float16 or shorter prompts"
                        ) from exc
                    raise
                finally:
                    del layer_capture
                    if query_states is not None:
                        del query_states
                    if key_states is not None:
                        del key_states
                    if value_states is not None:
                        del value_states
                    gc.collect()
                    clear_device_cache(torch)
                gc.collect()

            del captured_projections, tokens
            gc.collect()
            clear_device_cache(torch)

        f.write("\n  ]\n}\n")

    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
