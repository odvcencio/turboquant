#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPT_FILE="$REPO_ROOT/examples/tqkvbench_prompts.txt"
MODEL=""
LABEL=""
LAYER_SPEC="spread:4"
TOKEN_INDEX_SPEC="25%,50%,last"
DEVICE="cuda"
DTYPE="auto"
EXPORT_MEMORY="16g"
SWEEP_MEMORY="8g"
BUILD_IMAGE=1
SKIP_EXPORT=0
SKIP_SWEEP=0

usage() {
  cat <<'EOF'
Usage: run_kv_benchmark_pack.sh --model <hf-id-or-path> [options]

Runs a wider KV benchmark pack:
  1. exports captures for a prompt pack, spread layers, and relative token positions
  2. runs tqkvsweep with GPU enabled when using CUDA
  3. writes a tqkvsummarize report

Options:
  --model <id>          Hugging Face model id or local path
  --label <name>        Output prefix label (default: derived from model id)
  --prompt-file <path>  Prompt file, one non-empty line per prompt
  --layers <spec>       Layer spec passed to exporter (default: spread:4)
  --token-index <spec>  Token index spec passed to exporter (default: 25%,50%,last)
  --device <name>       Exporter device (default: cuda)
  --dtype <name>        Exporter dtype (default: auto)
  --export-memory <n>   Export container memory limit (default: 16g)
  --sweep-memory <n>    Sweep container memory limit (default: 8g)
  --skip-export         Reuse an existing capture file in tmp/ and only run sweep/summary
  --skip-sweep          Reuse an existing sweep report in tmp/ and only run summary
  --no-build            Skip docker builds for both wrappers
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
    --model)
      MODEL="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --prompt-file)
      PROMPT_FILE="$2"
      shift 2
      ;;
    --layers)
      LAYER_SPEC="$2"
      shift 2
      ;;
    --token-index)
      TOKEN_INDEX_SPEC="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --export-memory)
      EXPORT_MEMORY="$2"
      shift 2
      ;;
    --sweep-memory)
      SWEEP_MEMORY="$2"
      shift 2
      ;;
    --skip-export)
      SKIP_EXPORT=1
      shift
      ;;
    --skip-sweep)
      SKIP_SWEEP=1
      shift
      ;;
    --no-build)
      BUILD_IMAGE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  usage >&2
  exit 2
fi

PROMPT_FILE="${PROMPT_FILE/#\~/$HOME}"
if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "prompt file does not exist: $PROMPT_FILE" >&2
  exit 2
fi

if [[ -z "$LABEL" ]]; then
  LABEL="$(sanitize_label "$MODEL")"
else
  LABEL="$(sanitize_label "$LABEL")"
fi

CAPTURE_OUT="$REPO_ROOT/tmp/${LABEL}-pack-capture.json"
SWEEP_OUT="$REPO_ROOT/tmp/${LABEL}-pack-sweep-report.json"
SUMMARY_OUT="$REPO_ROOT/tmp/${LABEL}-pack-summary.json"
mkdir -p "$REPO_ROOT/tmp"

WORKSPACE_PROMPT_FILE="$PROMPT_FILE"
if [[ "$PROMPT_FILE" != "$REPO_ROOT/"* ]]; then
  WORKSPACE_PROMPT_FILE="$REPO_ROOT/tmp/${LABEL}-pack-prompts.txt"
  cp "$PROMPT_FILE" "$WORKSPACE_PROMPT_FILE"
fi

BUILD_FLAG=()
if [[ "$BUILD_IMAGE" == "0" ]]; then
  BUILD_FLAG=(--no-build)
fi

echo "model=$MODEL"
echo "label=$LABEL"
echo "prompt_file=$WORKSPACE_PROMPT_FILE"
echo "layers=$LAYER_SPEC"
echo "token_index=$TOKEN_INDEX_SPEC"
echo "capture_out=$CAPTURE_OUT"
echo "sweep_out=$SWEEP_OUT"
echo "summary_out=$SUMMARY_OUT"

if [[ "$SKIP_EXPORT" == "0" ]]; then
  bash "$SCRIPT_DIR/run_hf_kv_export_in_docker.sh" \
    "${BUILD_FLAG[@]}" \
    --label "${LABEL}-pack-export" \
    --memory "$EXPORT_MEMORY" \
    --device "$DEVICE" \
    -- \
    --model "$MODEL" \
    --prompt-file "/workspace/${WORKSPACE_PROMPT_FILE#$REPO_ROOT/}" \
    --layer "$LAYER_SPEC" \
    --token-index "$TOKEN_INDEX_SPEC" \
    --dtype "$DTYPE" \
    --out "/workspace/tmp/${LABEL}-pack-capture.json"
elif [[ ! -f "$CAPTURE_OUT" ]]; then
  echo "capture file does not exist for --skip-export: $CAPTURE_OUT" >&2
  exit 2
fi

if [[ "$SKIP_SWEEP" == "0" ]]; then
  SWEEP_WRAPPER_ARGS=(
    "${BUILD_FLAG[@]}"
    --label "${LABEL}-pack-sweep"
    --memory "$SWEEP_MEMORY"
    --
    --input "./tmp/${LABEL}-pack-capture.json"
    --out "./tmp/${LABEL}-pack-sweep-report.json"
  )
  if [[ "$DEVICE" == cuda* ]]; then
    SWEEP_WRAPPER_ARGS=(--gpu "${SWEEP_WRAPPER_ARGS[@]}")
  fi
  bash "$SCRIPT_DIR/run_tqkvsweep_in_docker.sh" "${SWEEP_WRAPPER_ARGS[@]}"
elif [[ ! -f "$SWEEP_OUT" ]]; then
  echo "sweep report does not exist for --skip-sweep: $SWEEP_OUT" >&2
  exit 2
fi

go run ./cmd/tqkvsummarize \
  --input "$SWEEP_OUT" \
  --out "$SUMMARY_OUT"

echo
echo "capture=$CAPTURE_OUT"
echo "sweep=$SWEEP_OUT"
echo "summary=$SUMMARY_OUT"
