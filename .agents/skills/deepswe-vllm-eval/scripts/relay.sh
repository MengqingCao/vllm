#!/usr/bin/env bash
# Run on the DeepSWE request machine.
# Task containers have no network and cannot use the host's 127.0.0.1.
# Publish the tunnel on the docker bridge, on port 80 or 443 only.
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-${USER}-deepswe-relay}"
BIND_ADDR="${BIND_ADDR:-172.17.0.1}"
BIND_PORT="${BIND_PORT:-443}"
UPSTREAM_PORT="${UPSTREAM_PORT:-18000}"

if [[ "$BIND_PORT" != "80" && "$BIND_PORT" != "443" ]]; then
  echo "BIND_PORT must be 80 or 443" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "container $CONTAINER_NAME already exists" >&2
  exit 1
fi

if [[ -n "${ADD_LOOPBACK:-}" ]]; then
  docker run --rm --privileged --network host alpine \
    ip addr add "${ADD_LOOPBACK}/32" dev lo || true
fi

docker run -d --name "$CONTAINER_NAME" --network host alpine/socat \
  "TCP-LISTEN:${BIND_PORT},bind=${BIND_ADDR},fork,reuseaddr" \
  "TCP:127.0.0.1:${UPSTREAM_PORT}"

echo "started $CONTAINER_NAME on ${BIND_ADDR}:${BIND_PORT}"
