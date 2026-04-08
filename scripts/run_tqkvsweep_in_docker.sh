#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"
OUT_ROOT="$HOME/work/gotreesitter/harness_out/docker"
LABEL=""

CPU_IMAGE_TAG="turboquant/go-cli:local"
CUDA_IMAGE_TAG="turboquant/go-cli-cuda:local"
IMAGE_TAG=""
MEMORY_LIMIT="12g"
CPUS_LIMIT="4"
PIDS_LIMIT="4096"
BUILD_IMAGE=1
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
GO_CACHE_ROOT="$HOME/.cache/turboquant-docker-go"
USE_GPU=0
GPUS=""

usage() {
  cat <<'EOF'
Usage: run_tqkvsweep_in_docker.sh [options] [-- [tqkvsweep args]]

Options:
  --image <tag>          Docker image tag (default: turboquant/go-cli:local)
  --repo-root <path>     Repository/worktree root mounted at /workspace
  --out-root <path>      Artifact output root
                         (default: ~/work/gotreesitter/harness_out/docker)
  --label <name>         Optional run label for container/artifact naming
  --memory <limit>       Container memory limit (default: 12g)
  --cpus <count>         CPU limit passed to Docker (default: 4)
  --pids <count>         PID limit passed to Docker (default: 4096)
  --gpu                  Build and run tqkvsweep with CUDA support and pass
                         the CLI --gpu flag. Uses the CUDA Go image and
                         Docker --gpus all by default.
  --gpus <spec>          Docker GPU request. Default auto-enables `all` when
                         --gpu is set. Use `none` to disable GPU wiring.
  --no-build             Skip docker build step
  -h, --help             Show this help

Environment passthrough (if set):
  GOMAXPROCS
  GOMEMLIMIT
  GOGC

Artifacts are written to <out-root>/<timestamp>[-<label>]/:
  - container.log
  - inspect.json
  - metadata.txt

If no tqkvsweep args are provided, the script runs:
  go run ./cmd/tqkvsweep --input ./captures.json --out ./tqkvsweep-report.json
EOF
}

SWEEP_ARGS=()
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
    --gpu)
      USE_GPU=1
      shift
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
      SWEEP_ARGS=("$@")
      break
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
OUT_ROOT="${OUT_ROOT/#\~/$HOME}"
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "repo root does not exist: $REPO_ROOT" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"
mkdir -p "$GO_CACHE_ROOT/gocache" "$GO_CACHE_ROOT/gomodcache"

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

has_cli_gpu_flag() {
  local arg
  for arg in "${SWEEP_ARGS[@]}"; do
    if [[ "$arg" == "--gpu" ]]; then
      return 0
    fi
  done
  return 1
}

LABEL_SLUG=""
if [[ -n "$LABEL" ]]; then
  LABEL_SLUG="$(sanitize_label "$LABEL")"
fi

DOCKERFILE="$DOCKER_DIR/Dockerfile.go-cli"
if [[ -z "$IMAGE_TAG" ]]; then
  IMAGE_TAG="$CPU_IMAGE_TAG"
fi
if [[ "$USE_GPU" == "1" ]]; then
  if [[ "$IMAGE_TAG" == "$CPU_IMAGE_TAG" ]]; then
    IMAGE_TAG="$CUDA_IMAGE_TAG"
  fi
  DOCKERFILE="$DOCKER_DIR/Dockerfile.go-cli-cuda"
fi

if [[ "$BUILD_IMAGE" == "1" ]]; then
  docker build -t "$IMAGE_TAG" -f "$DOCKERFILE" "$DOCKER_DIR"
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="$OUT_ROOT/$STAMP"
if [[ -n "$LABEL_SLUG" ]]; then
  OUT_DIR="${OUT_DIR}-${LABEL_SLUG}"
fi
mkdir -p "$OUT_DIR"

RUN_CMD=(go run)
if [[ "$USE_GPU" == "1" ]]; then
  RUN_CMD+=(-tags cuda)
fi
RUN_CMD+=(./cmd/tqkvsweep)
if [[ ${#SWEEP_ARGS[@]} -gt 0 ]]; then
  RUN_CMD+=("${SWEEP_ARGS[@]}")
else
  RUN_CMD+=(--input ./captures.json --out ./tqkvsweep-report.json)
fi
if [[ "$USE_GPU" == "1" ]] && ! has_cli_gpu_flag; then
  RUN_CMD+=(--gpu)
fi
INNER_CMD="export PATH=/usr/local/go/bin:\$PATH; cd /workspace && /usr/bin/time -v $(quote_cmd "${RUN_CMD[@]}")"

ENV_ARGS=()
ENV_ARGS+=(
  "-e" "GOCACHE=/cache/go-build"
  "-e" "GOMODCACHE=/cache/go-mod"
  "-e" "CGO_ENABLED=1"
)
for var in GOMAXPROCS GOMEMLIMIT GOGC; do
  if [[ -n "${!var:-}" ]]; then
    ENV_ARGS+=("-e" "$var=${!var}")
  fi
done

GPU_ARGS=()
if [[ "$USE_GPU" == "1" && -z "$GPUS" ]]; then
  GPUS="all"
fi
if [[ -n "$GPUS" && "$GPUS" != "none" ]]; then
  GPU_ARGS=(--gpus "$GPUS")
fi

CONTAINER_NAME="turboquant-sweep-${STAMP,,}"
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
  --mount "type=bind,src=$GO_CACHE_ROOT/gomodcache,dst=/cache/go-mod" \
  --mount "type=bind,src=$GO_CACHE_ROOT/gocache,dst=/cache/go-build" \
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
  echo "gpu_mode=$USE_GPU"
  echo "gpus=${GPUS:-none}"
  echo "memory=$MEMORY_LIMIT"
  echo "cpus=$CPUS_LIMIT"
  echo "pids=$PIDS_LIMIT"
  echo "exit_code=$EXIT_CODE"
  echo "oom_killed=$OOM_KILLED"
  echo "state_error=$STATE_ERROR"
  echo "repo_root=$REPO_ROOT"
  echo "out_root=$OUT_ROOT"
  echo "label=$LABEL_SLUG"
  echo "command=$INNER_CMD"
} >"$OUT_DIR/metadata.txt"

echo "docker sweep run complete"
echo "artifacts: $OUT_DIR"
echo "exit_code: $EXIT_CODE"
echo "oom_killed: $OOM_KILLED"
if [[ -n "$STATE_ERROR" ]]; then
  echo "docker_state_error: $STATE_ERROR"
fi

if [[ "$EXIT_CODE" != "0" ]]; then
  exit "$EXIT_CODE"
fi
