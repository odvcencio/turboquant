#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"
OUT_ROOT="$HOME/work/gotreesitter/harness_out/docker"
LABEL=""

IMAGE_TAG="turboquant/hf-kv-export:local"
MEMORY_LIMIT="16g"
CPUS_LIMIT="4"
PIDS_LIMIT="4096"
HF_CACHE="$HOME/.cache/huggingface"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
DEVICE="cuda"
GPUS=""
BUILD_IMAGE=1
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

usage() {
  cat <<'EOF'
Usage: run_hf_kv_export_in_docker.sh [options] -- [exporter args]

Options:
  --image <tag>          Docker image tag (default: turboquant/hf-kv-export:local)
  --repo-root <path>     Repository/worktree root mounted at /workspace
  --out-root <path>      Artifact output root
                         (default: ~/work/gotreesitter/harness_out/docker)
  --label <name>         Optional run label for container/artifact naming
  --memory <limit>       Container memory limit (default: 16g)
  --cpus <count>         CPU limit passed to Docker (default: 4)
  --pids <count>         PID limit passed to Docker (default: 4096)
  --hf-cache <path>      Hugging Face cache mounted at /root/.cache/huggingface
                         (default: ~/.cache/huggingface)
  --hf-token-file <path> Hugging Face token file to mount into the container.
                         Auto-discovers ~/.cache/huggingface/token,
                         ~/.config/huggingface/token, and ~/.huggingface/token.
  --device <name>        Exporter torch device (default: cuda)
  --gpus <spec>          Docker GPU request. Default auto-enables `all` for
                         cuda devices. Use `none` to disable GPU wiring.
  --no-build             Skip docker build step
  -h, --help             Show this help

Environment passthrough (if set):
  HF_TOKEN
  HUGGING_FACE_HUB_TOKEN
  HF_HUB_ENABLE_HF_TRANSFER
  TRANSFORMERS_OFFLINE

Artifacts are written to <out-root>/<timestamp>[-<label>]/:
  - container.log
  - inspect.json
  - metadata.txt

Example:
  scripts/run_hf_kv_export_in_docker.sh \
    --label llama32-1b \
    --memory 16g \
    -- --model meta-llama/Llama-3.2-1B \
       --prompt "Summarize the deployment plan." \
       --layer 0,8,16 \
       --token-index last \
       --dtype float16 \
       --out /workspace/captures.json
EOF
}

EXPORT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="$2"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --memory)
      MEMORY_LIMIT="$2"
      shift 2
      ;;
    --cpus)
      CPUS_LIMIT="$2"
      shift 2
      ;;
    --pids)
      PIDS_LIMIT="$2"
      shift 2
      ;;
    --hf-cache)
      HF_CACHE="$2"
      shift 2
      ;;
    --hf-token-file)
      HF_TOKEN_FILE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --no-build)
      BUILD_IMAGE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXPORT_ARGS=("$@")
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#EXPORT_ARGS[@]} -eq 0 ]]; then
  echo "expected exporter arguments after --" >&2
  usage >&2
  exit 2
fi

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
OUT_ROOT="${OUT_ROOT/#\~/$HOME}"
HF_CACHE="${HF_CACHE/#\~/$HOME}"
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "repo root does not exist: $REPO_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT" "$HF_CACHE"

sanitize_label() {
  local in="$1"
  in="${in,,}"
  in="$(echo "$in" | sed -E 's/[^a-z0-9_.-]+/-/g; s/^-+//; s/-+$//; s/-+/-/g')"
  if [[ -z "$in" ]]; then
    in="run"
  fi
  echo "$in"
}

quote_cmd() {
  local quoted
  printf -v quoted '%q ' "$@"
  echo "${quoted% }"
}

discover_hf_token_file() {
  local candidate
  if [[ -n "$HF_TOKEN_FILE" ]]; then
    candidate="${HF_TOKEN_FILE/#\~/$HOME}"
    if [[ -r "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
    echo "hf token file is not readable: $candidate" >&2
    exit 2
  fi
  for candidate in \
    "$HF_CACHE/token" \
    "$HOME/.config/huggingface/token" \
    "$HOME/.huggingface/token"
  do
    if [[ -r "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

extract_model_arg() {
  local arg
  local next_is_model=0
  for arg in "${EXPORT_ARGS[@]}"; do
    if [[ "$next_is_model" == "1" ]]; then
      echo "$arg"
      return 0
    fi
    case "$arg" in
      --model)
        next_is_model=1
        ;;
      --model=*)
        echo "${arg#--model=}"
        return 0
        ;;
    esac
  done
  return 1
}

has_local_files_only_flag() {
  local arg
  for arg in "${EXPORT_ARGS[@]}"; do
    if [[ "$arg" == "--local-files-only" ]]; then
      return 0
    fi
  done
  return 1
}

hf_cache_model_dir() {
  local model_id="$1"
  printf '%s/hub/models--%s' "$HF_CACHE" "${model_id//\//--}"
}

is_hub_model_id() {
  local model="$1"
  [[ "$model" == */* && "$model" != /* && ! -e "$model" ]]
}

is_known_gated_model() {
  local model="$1"
  [[ "$model" == meta-llama/* ]]
}

LABEL_SLUG=""
if [[ -n "$LABEL" ]]; then
  LABEL_SLUG="$(sanitize_label "$LABEL")"
fi

MODEL_ARG="$(extract_model_arg || true)"
TOKEN_FILE="$(discover_hf_token_file || true)"
MODEL_CACHE_DIR=""
if [[ -n "$MODEL_ARG" ]] && is_hub_model_id "$MODEL_ARG"; then
  MODEL_CACHE_DIR="$(hf_cache_model_dir "$MODEL_ARG")"
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" && -z "$TOKEN_FILE" ]]; then
  if [[ -n "$MODEL_ARG" ]] && is_known_gated_model "$MODEL_ARG"; then
    if [[ -n "$MODEL_CACHE_DIR" && -d "$MODEL_CACHE_DIR" ]] && ! has_local_files_only_flag; then
      echo "No Hugging Face token found; using cached snapshot for $MODEL_ARG via --local-files-only." >&2
      EXPORT_ARGS+=("--local-files-only")
    elif [[ -z "$MODEL_CACHE_DIR" || ! -d "$MODEL_CACHE_DIR" ]]; then
      echo "Cannot access gated model $MODEL_ARG: no HF token found and no cached snapshot at $HF_CACHE." >&2
      echo "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN, create ~/.cache/huggingface/token, or pass a local model path." >&2
      exit 2
    fi
  fi
fi

if [[ "$BUILD_IMAGE" == "1" ]]; then
  docker build -t "$IMAGE_TAG" -f "$DOCKER_DIR/Dockerfile.hf-export" "$DOCKER_DIR"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_ROOT/$STAMP"
if [[ -n "$LABEL_SLUG" ]]; then
  OUT_DIR="${OUT_DIR}-${LABEL_SLUG}"
fi
mkdir -p "$OUT_DIR"

if [[ -z "$GPUS" && "$DEVICE" == cuda* ]]; then
  GPUS="all"
fi

RUN_CMD=(python3 scripts/export_hf_llama_kv_capture.py --device "$DEVICE")
RUN_CMD+=("${EXPORT_ARGS[@]}")
INNER_CMD="mkdir -p \"\$HOME\" \"\$XDG_CACHE_HOME\" \"\$TRITON_CACHE_DIR\" && cd /workspace && /usr/bin/time -v $(quote_cmd "${RUN_CMD[@]}")"

ENV_ARGS=()
ENV_ARGS+=(
  "-e" "HOME=/hf-cache/home"
  "-e" "HF_HOME=/hf-cache"
  "-e" "HF_HUB_CACHE=/hf-cache/hub"
  "-e" "TRANSFORMERS_CACHE=/hf-cache/transformers"
  "-e" "XDG_CACHE_HOME=/hf-cache/home/.cache"
  "-e" "TRITON_CACHE_DIR=/hf-cache/triton"
)
for var in HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_ENABLE_HF_TRANSFER TRANSFORMERS_OFFLINE; do
  if [[ -n "${!var:-}" ]]; then
    ENV_ARGS+=("-e" "$var=${!var}")
  fi
done

GPU_ARGS=()
if [[ -n "$GPUS" && "$GPUS" != "none" ]]; then
  GPU_ARGS=(--gpus "$GPUS")
fi

TOKEN_MOUNT_ARGS=()
if [[ -n "$TOKEN_FILE" && "$TOKEN_FILE" != "$HF_CACHE/token" ]]; then
  TOKEN_MOUNT_ARGS=(--mount "type=bind,src=$TOKEN_FILE,dst=/hf-cache/token,readonly")
fi

CONTAINER_NAME="turboquant-export-${STAMP,,}"
if [[ -n "$LABEL_SLUG" ]]; then
  CONTAINER_NAME="${CONTAINER_NAME}-${LABEL_SLUG}"
fi

CID=""
cleanup() {
  if [[ -n "$CID" ]]; then
    docker rm -f "$CID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

CID="$(docker create \
  --name "$CONTAINER_NAME" \
  --init \
  --user "$HOST_UID:$HOST_GID" \
  "${GPU_ARGS[@]}" \
  --memory "$MEMORY_LIMIT" \
  --memory-swap "$MEMORY_LIMIT" \
  --cpus "$CPUS_LIMIT" \
  --pids-limit "$PIDS_LIMIT" \
  --mount "type=bind,src=$REPO_ROOT,dst=/workspace" \
  --mount "type=bind,src=$HF_CACHE,dst=/hf-cache" \
  "${TOKEN_MOUNT_ARGS[@]}" \
  "${ENV_ARGS[@]}" \
  "$IMAGE_TAG" \
  bash -lc "$INNER_CMD")"

docker start "$CID" >/dev/null
docker logs -f "$CID" 2>&1 | tee "$OUT_DIR/container.log"
EXIT_CODE="$(docker wait "$CID")"
docker inspect "$CID" >"$OUT_DIR/inspect.json"

OOM_KILLED="$(docker inspect -f '{{.State.OOMKilled}}' "$CID")"
STATE_ERROR="$(docker inspect -f '{{.State.Error}}' "$CID")"

{
  echo "container_name=$CONTAINER_NAME"
  echo "container_id=$CID"
  echo "image=$IMAGE_TAG"
  echo "memory=$MEMORY_LIMIT"
  echo "cpus=$CPUS_LIMIT"
  echo "pids=$PIDS_LIMIT"
  echo "device=$DEVICE"
  echo "gpus=${GPUS:-none}"
  echo "hf_cache=$HF_CACHE"
  echo "hf_token_file=${TOKEN_FILE:-}"
  echo "exit_code=$EXIT_CODE"
  echo "oom_killed=$OOM_KILLED"
  echo "state_error=$STATE_ERROR"
  echo "repo_root=$REPO_ROOT"
  echo "out_root=$OUT_ROOT"
  echo "label=$LABEL_SLUG"
  echo "command=$INNER_CMD"
} >"$OUT_DIR/metadata.txt"

echo "docker export run complete"
echo "artifacts: $OUT_DIR"
echo "exit_code: $EXIT_CODE"
echo "oom_killed: $OOM_KILLED"
if [[ -n "$STATE_ERROR" ]]; then
  echo "docker_state_error: $STATE_ERROR"
fi

if [[ "$EXIT_CODE" != "0" ]]; then
  exit "$EXIT_CODE"
fi
