#!/usr/bin/env bash

AZURE_STORAGE_CONNECTION_STRING="${AZURE_STORAGE_CONNECTION_STRING:?AZURE_STORAGE_CONNECTION_STRING required}"
AZURE_STORAGE_ACCOUNT_NAME="${AZURE_STORAGE_ACCOUNT_NAME:?AZURE_STORAGE_ACCOUNT_NAME required}"
AZURE_RUNBOOKS_CONTAINER="${AZURE_RUNBOOKS_CONTAINER:-\$web}"
RUNBOOKS_SRC_DIR="${RUNBOOKS_SRC_DIR:-docs/infra/observability/runbooks}"
RUNBOOKS_BUILD_DIR="${RUNBOOKS_BUILD_DIR:-.runbooks_build}"
RUNBOOKS_BASE_URL="${RUNBOOKS_BASE_URL:-https://${AZURE_STORAGE_ACCOUNT_NAME}.z13.web.core.windows.net}"

ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
log(){ echo "ts=$(ts) level=$1 msg=$2 ${3:-}"; }

log INFO runbooks_publish_start "account=${AZURE_STORAGE_ACCOUNT_NAME} container=${AZURE_RUNBOOKS_CONTAINER} src=${RUNBOOKS_SRC_DIR}"

if [ ! -d "${RUNBOOKS_SRC_DIR}" ]; then
  log ERROR source_directory_missing "path=${RUNBOOKS_SRC_DIR}"
  exit 2
fi

for bin in az pandoc curl; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    log ERROR dependency_missing "binary=${bin}"
    exit 3
  fi
done

rm -rf "${RUNBOOKS_BUILD_DIR}"
mkdir -p "${RUNBOOKS_BUILD_DIR}"

log INFO render_markdown_to_html
shopt -s nullglob
for f in "${RUNBOOKS_SRC_DIR}"/*.md; do
  name="$(basename "$f" .md)"
  out="${RUNBOOKS_BUILD_DIR}/${name}.html"
  pandoc "$f" --from markdown --to html5 --standalone --metadata title="${name} runbook" -o "$out"
  if [ $? -ne 0 ]; then
    log ERROR pandoc_failed "file=${f}"
    exit 4
  fi
done
shopt -u nullglob

log INFO ensure_static_website_enabled
az storage blob service-properties update \
  --connection-string "${AZURE_STORAGE_CONNECTION_STRING}" \
  --static-website \
  --index-document index.html \
  --404-document 404.html \
  --only-show-errors >/dev/null 2>&1

enabled="$(az storage blob service-properties show \
  --connection-string "${AZURE_STORAGE_CONNECTION_STRING}" \
  --query "staticWebsite.enabled" -o tsv 2>/dev/null || echo false)"

if [ "${enabled}" != "true" ]; then
  log ERROR static_website_not_enabled "enabled=${enabled}"
  exit 5
fi
log INFO static_website_enabled

log INFO ensure_container_exists "container=${AZURE_RUNBOOKS_CONTAINER}"
az storage container create \
  --name "${AZURE_RUNBOOKS_CONTAINER}" \
  --connection-string "${AZURE_STORAGE_CONNECTION_STRING}" \
  --only-show-errors >/dev/null 2>&1
log INFO container_ready "container=${AZURE_RUNBOOKS_CONTAINER}"

log INFO upload_runbooks_start

declare -A MIME
MIME[html]=text/html
MIME[css]=text/css
MIME[js]=application/javascript
MIME[png]=image/png
MIME[jpg]=image/jpeg
MIME[jpeg]=image/jpeg
MIME[svg]=image/svg+xml
MIME[json]=application/json
MIME[txt]=text/plain

files=()
while IFS= read -r -d $'\0' f; do files+=("$f"); done < <(find "${RUNBOOKS_BUILD_DIR}" -type f -print0)
while IFS= read -r -d $'\0' f; do files+=("$f"); done < <(find "${RUNBOOKS_SRC_DIR}" -maxdepth 1 -type f ! -name "*.md" -print0)

if [ "${#files[@]}" -eq 0 ]; then
  log WARN no_files_to_upload
fi

for file in "${files[@]}"; do
  if [[ "$file" == "${RUNBOOKS_BUILD_DIR}/"* ]]; then
    rel="${file#${RUNBOOKS_BUILD_DIR}/}"
  else
    rel="${file#${RUNBOOKS_SRC_DIR}/}"
  fi
  ext="${rel##*.}"
  ctype="${MIME[$ext]:-application/octet-stream}"
  az storage blob upload \
    --container-name "${AZURE_RUNBOOKS_CONTAINER}" \
    --file "$file" \
    --name "$rel" \
    --content-type "$ctype" \
    --overwrite \
    --connection-string "${AZURE_STORAGE_CONNECTION_STRING}" \
    --only-show-errors >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    log ERROR blob_upload_failed "file=${file}"
    exit 6
  fi
done

log INFO upload_runbooks_complete

first_html="$(ls "${RUNBOOKS_BUILD_DIR}"/*.html 2>/dev/null | head -n 1 || true)"
if [ -z "${first_html}" ]; then
  log WARN no_html_runbooks_found "skip_access_verification"
else
  page="$(basename "${first_html}")"
  log INFO verify_runbook_access "page=${page}"
  curl -sfI "${RUNBOOKS_BASE_URL}/${page}" >/dev/null 2>&1
  if [ $? -ne 0 ]; then
    log ERROR runbook_not_accessible "url=${RUNBOOKS_BASE_URL}/${page}"
    exit 7
  fi
fi

log INFO published_runbooks_list_start
for f in "${RUNBOOKS_BUILD_DIR}"/*.html; do
  [ -e "$f" ] || continue
  echo "ts=$(ts) level=INFO msg=runbook_url url=${RUNBOOKS_BASE_URL}/$(basename "$f")"
done
for f in "${RUNBOOKS_SRC_DIR}"/*; do
  [ -e "$f" ] || continue
  case "$f" in
    *.md) ;;
    *) echo "ts=$(ts) level=INFO msg=static_asset url=${RUNBOOKS_BASE_URL}/$(basename "$f")" ;;
  esac
done
rm -rf .runbooks_build
echo "RUNBOOK_BASE_URL=$RUNBOOK_BASE_URL"
RUNBOOK_BASE_URL="$(az storage account show --name "$AZURE_STORAGE_ACCOUNT_NAME" --query "primaryEndpoints.web" -o tsv | sed 's:/*$::')" && [ -n "$RUNBOOK_BASE_URL" ] && sed -i '/^export RUNBOOK_BASE_URL=/d' ~/.bashrc && echo "export RUNBOOK_BASE_URL=$RUNBOOK_BASE_URL" >> ~/.bashrc
exit 0
