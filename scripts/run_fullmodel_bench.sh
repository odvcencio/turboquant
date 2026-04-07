#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$REPO_ROOT/examples/tqkvbench_prompts.txt"
MODEL=""
LABEL=""
DEVICE="cuda"
DTYPE="auto"
EXPORT_MEMORY="16g"
PROFILE_PATH=""
PROFILE_SET="last"
WARMUP=1
ITERATIONS=3
GPU_FLAG=""
BUILD_IMAGE=1
SKIP_EXPORT=0

usage() {
  cat <<'EOF'
Usage: run_fullmodel_bench.sh --model <hf-id-or-path> --profile <path> [options]

Runs a full-model KV benchmark:
  1. exports captures for ALL layers at token position "last"
  2. runs tqkvprofilebench with --expand-profile against the existing profile

Requires a pre-existing profile file from tqkvprofile (via the pack pipeline).

Options:
  --model <id>          Hugging Face model id or local path
  --label <name>        Output prefix label (default: derived from model id)
  --profile <path>      Path to tqkvprofile bundle JSON (required)
  --profile-set <set>   Profile set to bench (default: last)
  --prompt-file <path>  Prompt file, one non-empty line per prompt
  --device <name>       Exporter device (default: cuda)
  --dtype <name>        Exporter dtype (default: auto)
  --export-memory <n>   Export container memory limit (default: 16g)
  --warmup <n>          Bench warmup runs (default: 1)
  --iterations <n>      Bench timed runs (default: 3)
  --gpu                 Enable GPU bench path
  --skip-export         Reuse existing all-layer capture in tmp/
  --no-build            Skip docker image builds
  -h, --help            Show this help
EOF
}

sanitize_label() {
  local in="$1"
  in="${in##*/}"
  in="${in,,}"
  in="$(echo "$in" | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  if [[ -z "$in" ]]; then
    in="kv-bench"
  fi
  echo "$in"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL="$2"; shift 2 ;;
    --label)       LABEL="$2"; shift 2 ;;
    --profile)     PROFILE_PATH="$2"; shift 2 ;;
    --profile-set) PROFILE_SET="$2"; shift 2 ;;
    --prompt-file) PROMPT_FILE="$2"; shift 2 ;;
    --device)      DEVICE="$2"; shift 2 ;;
    --dtype)       DTYPE="$2"; shift 2 ;;
    --export-memory) EXPORT_MEMORY="$2"; shift 2 ;;
    --warmup)      WARMUP="$2"; shift 2 ;;
    --iterations)  ITERATIONS="$2"; shift 2 ;;
    --gpu)         GPU_FLAG="--gpu"; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --no-build)    BUILD_IMAGE=0; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2; usage >&2; exit 2
fi
if [[ -z "$PROFILE_PATH" ]]; then
  echo "--profile is required" >&2; usage >&2; exit 2
fi
if [[ ! -f "$PROFILE_PATH" ]]; then
  echo "profile file does not exist: $PROFILE_PATH" >&2; exit 2
fi

PROMPT_FILE="${PROMPT_FILE/#\~/$HOME}"
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "prompt file does not exist: $PROMPT_FILE" >&2; exit 2
fi

if [[ -z "$LABEL" ]]; then
  LABEL="$(sanitize_label "$MODEL")"
else
  LABEL="$(sanitize_label "$LABEL")"
fi

CAPTURE_OUT="$REPO_ROOT/tmp/${LABEL}-alllayers-last-capture.json"
BENCH_OUT="$REPO_ROOT/tmp/${LABEL}-fullmodel-bench.json"
mkdir -p "$REPO_ROOT/tmp"

WORKSPACE_PROMPT_FILE="$PROMPT_FILE"
if [[ "$PROMPT_FILE" != "$REPO_ROOT/"* ]]; then
  WORKSPACE_PROMPT_FILE="$REPO_ROOT/tmp/${LABEL}-fullmodel-prompts.txt"
  cp "$PROMPT_FILE" "$WORKSPACE_PROMPT_FILE"
fi

BUILD_FLAG=()
if [[ "$BUILD_IMAGE" == "0" ]]; then
  BUILD_FLAG=(--no-build)
fi

echo "model=$MODEL"
echo "label=$LABEL"
echo "profile=$PROFILE_PATH"
echo "profile_set=$PROFILE_SET"
echo "capture_out=$CAPTURE_OUT"
echo "bench_out=$BENCH_OUT"

if [[ "$SKIP_EXPORT" == "0" ]]; then
  bash "$SCRIPT_DIR/run_hf_kv_export_in_docker.sh" \
    "${BUILD_FLAG[@]}" \
    --label "${LABEL}-alllayers-export" \
    --memory "$EXPORT_MEMORY" \
    --device "$DEVICE" \
    -- \
    --model "$MODEL" \
    --prompt-file "/workspace/${WORKSPACE_PROMPT_FILE#$REPO_ROOT/}" \
    --layer all \
    --token-index last \
    --dtype "$DTYPE" \
    --out "/workspace/tmp/${LABEL}-alllayers-last-capture.json"
elif [[ ! -f "$CAPTURE_OUT" ]]; then
  echo "capture file does not exist for --skip-export: $CAPTURE_OUT" >&2
  exit 2
fi

BENCH_ARGS=(
  --capture "$CAPTURE_OUT"
  --profile "$PROFILE_PATH"
  --expand-profile
  --profile-set "$PROFILE_SET"
  --warmup "$WARMUP"
  --iterations "$ITERATIONS"
  --out "$BENCH_OUT"
)
if [[ -n "$GPU_FLAG" ]]; then
  BENCH_ARGS+=("$GPU_FLAG")
fi

go run ./cmd/tqkvprofilebench "${BENCH_ARGS[@]}"

echo
echo "capture=$CAPTURE_OUT"
echo "bench=$BENCH_OUT"
