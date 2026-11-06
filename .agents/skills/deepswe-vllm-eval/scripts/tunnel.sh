#!/usr/bin/env bash
# Run on the DeepSWE request machine.
# Forwards 127.0.0.1:LOCAL_PORT to the GB300 vLLM port through the jump host.
# Uses SSH credentials already on this machine. Do not copy a private key here.
set -u
: "${JUMP_HOST:?set JUMP_HOST}"
: "${GPU_HOST:?set GPU_HOST}"
: "${GPU_USER:?set GPU_USER}"

JUMP_USER="${JUMP_USER:-$GPU_USER}"
JUMP_PORT="${JUMP_PORT:-22}"
LOCAL_PORT="${LOCAL_PORT:-18000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"

while true; do
  ssh -N \
    -o BatchMode=yes \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=accept-new \
    -o "ProxyCommand=ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p ${JUMP_PORT} -W %h:%p ${JUMP_USER}@${JUMP_HOST}" \
    -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" \
    "${GPU_USER}@${GPU_HOST}"
  echo "$(date -Is) tunnel exited rc=$?, reconnecting in 5s" >&2
  sleep 5
done
