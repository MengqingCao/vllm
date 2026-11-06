#!/usr/bin/env bash
# Run on the GB300 after the user confirms engine, image, and serve config.
# Required: CONTAINER_NAME IMAGE MODEL
# Optional: LOG_DIR SRC VLLM_GIT_URL VLLM_GIT_REF INSTALL_VLLM INSTALL_BUILD_DEPS
#           HF_CACHE_MOUNT HF_HUB_CACHE HF_HUB_OFFLINE GPU_DEVICES
# INSTALL_VLLM=1 only for a confirmed vLLM source install. SGLang stays 0.
# Serve command: $LOG_DIR/run-serve.sh (mounted at /logs/run-serve.sh)
set -euo pipefail

: "${CONTAINER_NAME:?set CONTAINER_NAME}"
: "${IMAGE:?set IMAGE}"
: "${MODEL:?set MODEL}"

LOG_DIR="${LOG_DIR:-${HOME}/deepswe-eval/logs}"
SRC="${SRC:-${HOME}/vllm}"
VLLM_GIT_URL="${VLLM_GIT_URL:-https://github.com/vllm-project/vllm.git}"
VLLM_GIT_REF="${VLLM_GIT_REF:-main}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
INSTALL_BUILD_DEPS="${INSTALL_BUILD_DEPS:-0}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
GPU_DEVICES="${GPU_DEVICES:-all}"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "container $CONTAINER_NAME already exists" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
if [[ ! -f "${LOG_DIR}/run-serve.sh" ]]; then
  echo "missing ${LOG_DIR}/run-serve.sh" >&2
  exit 1
fi
chmod +x "${LOG_DIR}/run-serve.sh"

docker pull "$IMAGE"

mount_args=()
if [[ -n "${HF_CACHE_MOUNT:-}" ]]; then
  mount_args+=(-v "${HF_CACHE_MOUNT}:${HF_CACHE_MOUNT}:ro")
fi

env_args=(-e "PYTHONUNBUFFERED=1" -e "VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-7200}")
if [[ -n "${HF_HUB_CACHE:-}" ]]; then
  env_args+=(-e "HF_HUB_CACHE=${HF_HUB_CACHE}" -e "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}")
fi

if [[ "$INSTALL_VLLM" == "1" ]]; then
  if [[ ! -d "${SRC}/.git" ]]; then
    git clone "$VLLM_GIT_URL" "$SRC"
  fi
  if [[ -n "$(git -C "$SRC" status --porcelain)" ]]; then
    echo "vLLM checkout is dirty: $SRC" >&2
    exit 1
  fi
  git -C "$SRC" fetch origin "$VLLM_GIT_REF"
  git -C "$SRC" checkout --detach FETCH_HEAD
  git -C "$SRC" rev-parse HEAD | tee "${LOG_DIR}/vllm-sha.txt"
  mount_args+=(-v "${SRC}:/opt/vllm")
fi

docker run -d --name "$CONTAINER_NAME" \
  --gpus "$GPU_DEVICES" --ipc=host --network host \
  "${mount_args[@]}" \
  -v "${LOG_DIR}:/logs" \
  "${env_args[@]}" \
  --entrypoint bash \
  "$IMAGE" \
  -lc 'sleep infinity'

if [[ "$INSTALL_VLLM" == "1" ]]; then
  docker exec -e "INSTALL_BUILD_DEPS=${INSTALL_BUILD_DEPS}" "$CONTAINER_NAME" bash -lc '
    set -euo pipefail
    git config --global --add safe.directory /opt/vllm
    if [[ "${INSTALL_BUILD_DEPS}" == "1" ]] && ! dpkg -s libdw-dev >/dev/null 2>&1; then
      apt-get update || {
        sed -i "s#http://ports.ubuntu.com#https://ports.ubuntu.com#g" /etc/apt/sources.list.d/ubuntu.sources
        apt-get update
      }
      apt-get install -y libdw-dev
    fi
    cd /opt/vllm
    python3 tools/use_existing_torch.py
    python3 -m pip install -r requirements/build/cuda.txt
    python3 -m pip install --no-build-isolation -e .
  '
fi

docker exec -d "$CONTAINER_NAME" bash -lc 'exec /logs/run-serve.sh'
echo "started ${CONTAINER_NAME}; log dir ${LOG_DIR}"
