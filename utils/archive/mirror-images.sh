#!/usr/bin/env bash
set -euo pipefail

# === Config / preflight ===
: "${GIT_PAT:?GIT_PAT must be set (GitHub Personal Access Token)}"
GH_USER="${GH_USER:-Athithya-Sakthivel}"           # exact GitHub login (case-sensitive for docker login)
GH_OWNER="$(echo "$GH_USER" | tr '[:upper:]' '[:lower:]')"  # GHCR namespace must be lowercase

DOCKER_IMAGES="docker.io/qdrant/qdrant:v1.16.0"
if [[ -z "$DOCKER_IMAGES" ]]; then
  echo "ERROR: DOCKER_IMAGES is empty. Set DOCKER_IMAGES and re-run."
  exit 2
fi

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker CLI is required"; exit 2; }

# Login to GHCR (exact username casing)
docker logout ghcr.io >/dev/null 2>&1 || true
rm -f "${HOME}/.docker/config.json" >/dev/null 2>&1 || true
echo "$GIT_PAT" | docker login ghcr.io -u "${GH_USER}" --password-stdin

IFS=',' read -ra IMAGES <<< "$DOCKER_IMAGES"

echo "Starting amd64 mirroring for ${#IMAGES[@]} images..."

for SRC_IMAGE in "${IMAGES[@]}"; do
  SRC_IMAGE="$(echo "$SRC_IMAGE" | xargs)"   # trim whitespace
  if [[ -z "$SRC_IMAGE" ]]; then
    continue
  fi

  # Skip images already hosted in our GHCR namespace
  if [[ "$SRC_IMAGE" == "ghcr.io/${GH_OWNER}/"* ]]; then
    echo "SKIP: already in GHCR namespace -> $SRC_IMAGE"
    echo
    continue
  fi

  NAME="${SRC_IMAGE%%[:@]*}"   # registry/namespace/repo
  REF="${SRC_IMAGE#${NAME}}"   # :tag or @sha256:...
  REPO="$(basename "$NAME")"

  DST_TAG="ghcr.io/${GH_OWNER}/${REPO}:mirror-temp-$(date +%s)"

  echo "Mirroring (amd64):"
  echo "  FROM: ${SRC_IMAGE}"
  echo "  TO:   ${DST_TAG}"

  # 1) Pull the amd64 image locally
  echo "-> docker pull --platform=linux/amd64 ${SRC_IMAGE}"
  docker pull --platform=linux/amd64 "${SRC_IMAGE}"

  # 2) Tag local image to our GHCR temporary tag
  echo "-> docker tag ${SRC_IMAGE} ${DST_TAG}"
  docker tag "${SRC_IMAGE}" "${DST_TAG}"

  # 3) Push the tag to GHCR (this will push only amd64 single-arch image)
  echo "-> docker push ${DST_TAG}"
  docker push "${DST_TAG}"

  # 4) Inspect pushed repo digest (immutable)
  #    docker inspect returns "ghcr.io/owner/repo@sha256:..."
  REPODIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "${DST_TAG}" 2>/dev/null || true)"

  if [[ -z "${REPODIGEST}" ]]; then
    # Try scanning all repo digests if index 0 missing (defensive)
    REPODIGEST="$(docker inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "${DST_TAG}" | grep "ghcr.io/${GH_OWNER}/${REPO}@" | head -n1 || true)"
  fi

  if [[ -z "${REPODIGEST}" ]]; then
    echo "ERROR: couldn't obtain digest for ${DST_TAG}"
    # cleanup temporary tag before exit
    docker rmi "${DST_TAG}" >/dev/null 2>&1 || true
    exit 3
  fi

  # Extract digest-only portion and print full immutable pull reference
  DIGEST="${REPODIGEST##*@}"
  echo "PULL (immutable, amd64):"
  echo "  ghcr.io/${GH_OWNER}/${REPO}@${DIGEST}"
  echo

  # 5) Cleanup temporary local tag and optionally the pulled source image
  docker rmi "${DST_TAG}" >/dev/null 2>&1 || true
  # Optionally remove the source image to free space:
  docker rmi "${SRC_IMAGE}" >/dev/null 2>&1 || true
done

echo "Done."
