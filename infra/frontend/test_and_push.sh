#!/usr/bin/env bash
set -euo pipefail
DOCKER_USERNAME="${DOCKER_USERNAME:-athithya324}"
IMAGE_NAME="${IMAGE_NAME:-$DOCKER_USERNAME/frontend-streamlit:v1}"
PUSH_REGISTRY="${PUSH_REGISTRY:-$IMAGE_NAME}"
CONTAINER_NAME="${CONTAINER_NAME:-frontend-test-run}"
APP_PORT="${APP_PORT:-3000}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-60}"
NO_CACHE="${NO_CACHE:-0}"

BUILD_CMD=(docker build -t "$IMAGE_NAME" .)
[ "$NO_CACHE" -eq 1 ] && BUILD_CMD+=(--no-cache)
"${BUILD_CMD[@]}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run --rm -d --name "$CONTAINER_NAME" -p "${APP_PORT}:${APP_PORT}" "$IMAGE_NAME" >/dev/null

echo "Waiting for Streamlit ready message..."
start=$(date +%s)
while true; do
  if docker logs "$CONTAINER_NAME" 2>/dev/null | tail -n 200 | grep -q "You can now view your Streamlit app"; then
    echo "Streamlit ready."
    break
  fi
  sleep 1
  if [ $(( $(date +%s) - start )) -ge "$WAIT_TIMEOUT" ]; then
    echo "WARNING: Streamlit did not report ready in time. Container logs:"
    docker logs "$CONTAINER_NAME" --tail 200 || true
    break
  fi
done

# push
if [ -z "${DOCKER_PASSWORD:-}" ]; then
  echo "ERROR: DOCKER_PASSWORD not set. Set it to push."
  exit 1
fi
echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
docker tag "$IMAGE_NAME" "$PUSH_REGISTRY"
docker push "$PUSH_REGISTRY"
echo "Push complete: $PUSH_REGISTRY"
