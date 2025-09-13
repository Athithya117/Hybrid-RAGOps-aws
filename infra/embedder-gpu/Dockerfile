# Common settings
MODEL_HOST_PATH=/workspace/models
CONTAINER_NAME=embedder-test-run21
APP_PORT=8006

# ---- AMD64 ----
docker pull athithya324/embedder-cpu-inference@sha256:c3c290f22318586071efa3d615d4e142cbf173a66b04f183fcd4e19808914026

docker run --rm -d --name ${CONTAINER_NAME}-amd64 \
  -p ${APP_PORT}:${APP_PORT} \
  --shm-size=1.8g \
  -v ${MODEL_HOST_PATH}:/workspace/models:ro \
  -e PORT=${APP_PORT} \
  athithya324/embedder-cpu-inference@sha256:c3c290f22318586071efa3d615d4e142cbf173a66b04f183fcd4e19808914026



# ---- ARM64 ----
docker pull athithya324/embedder-cpu-inference@sha256:61441535a6436f92a5d41832b5683716b87bf9323ea5defb1142db8bb5297340

docker run --rm -d --name ${CONTAINER_NAME}-arm64 \
  -p ${APP_PORT}:${APP_PORT} \
  --shm-size=1.8g \
  -v ${MODEL_HOST_PATH}:/workspace/models:ro \
  -e PORT=${APP_PORT} \
  athithya324/embedder-cpu-inference@sha256:61441535a6436f92a5d41832b5683716b87bf9323ea5defb1142db8bb5297340


