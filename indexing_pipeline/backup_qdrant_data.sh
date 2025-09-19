#!/usr/bin/env bash
set -euo pipefail
S3_BUCKET="${S3_BUCKET:-}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-}"
BACKUP_AND_RESTORE_PATH="${BACKUP_AND_RESTORE_PATH:-/workspace/qdrant/data/}"
BACKUP_FROM_PATH="${BACKUP_FROM_PATH:-/workspace/qdrant/backups/snapshots/}"
BACKUP_TO_PREFIX="${BACKUP_TO_PREFIX:-qdrant/backups}"
MULTIPART_CHUNKSIZE_MB="${MULTIPART_CHUNKSIZE_MB:-100}"
OUTPUT_MANIFEST_KEY="${OUTPUT_MANIFEST_KEY:-latest_qdrant_backup.manifest.json}"
[ -n "$S3_BUCKET" ] || { printf 'error: S3_BUCKET must be set\n' >&2; exit 1; }
if [ -z "$SNAPSHOT_PATH" ]; then SNAPSHOT_PATH="$(ls -t "${BACKUP_FROM_PATH}"/*.snapshot.zst 2>/dev/null | head -n1 || true)"; fi
[ -n "$SNAPSHOT_PATH" ] || { printf 'error: no snapshot found in %s and SNAPSHOT_PATH not set\n' "$BACKUP_FROM_PATH" >&2; exit 1; }
[ -f "$SNAPSHOT_PATH" ] || { printf 'error: snapshot file not found: %s\n' "$SNAPSHOT_PATH" >&2; exit 1; }
VCPUS="${VCPUS:-$(nproc)}"
if [ -n "${BACKUP_MAX_CONCURRENCY:-}" ]; then case "$BACKUP_MAX_CONCURRENCY" in ''|*[!0-9]*) CAND_CONC=$((VCPUS/2)); if [ "$CAND_CONC" -lt 1 ]; then CAND_CONC=1; fi;;*) CAND_CONC="$BACKUP_MAX_CONCURRENCY";;esac; else CAND_CONC=$((VCPUS/2)); if [ "$CAND_CONC" -lt 1 ]; then CAND_CONC=1; fi; fi
SNAPSHOT_BASENAME="$(basename "$SNAPSHOT_PATH")"
DEST_S3="s3://${S3_BUCKET%/}/${BACKUP_TO_PREFIX%/}/$SNAPSHOT_BASENAME"
SIZE_BYTES="$(stat -c%s "$SNAPSHOT_PATH")"
LATEST_SNAPSHOT_SIZE_IN_MB=$(( (SIZE_BYTES + 1048575) / 1048576 ))
START_TS="$(date +%s)"
aws s3 cp --only-show-errors --no-progress "$SNAPSHOT_PATH" "$DEST_S3"
END_TS="$(date +%s)"
UPLOAD_DURATION_SEC=$((END_TS - START_TS))
TMP_MANIFEST="$(mktemp /tmp/qdrant_manifest.XXXX)"
python3 - <<'PY' > "$TMP_MANIFEST"
import json,os,sys
data={
"backup_and_restore_path": os.environ.get("BACKUP_AND_RESTORE_PATH"),
"latest_snapshot": os.path.basename(os.environ.get("SNAPSHOT_PATH")),
"latest_snapshot_from_path": os.path.abspath(os.environ.get("BACKUP_FROM_PATH")),
"latest_snapshot_to_path": os.environ.get("DEST_S3"),
"latest_snapshot_size_in_mb": int(os.environ.get("LATEST_SNAPSHOT_SIZE_IN_MB")),
"multipart_upload_concurrency": int(os.environ.get("CAND_CONC")),
"latest_snapshot_upload_duration_in_seconds": int(os.environ.get("UPLOAD_DURATION_SEC"))
}
print(json.dumps(data,indent=2))
PY
aws s3 cp --only-show-errors --no-progress "$TMP_MANIFEST" "s3://${S3_BUCKET%/}/$OUTPUT_MANIFEST_KEY"
rm -f "$TMP_MANIFEST"
printf "manifest written to s3://%s/%s\n" "$S3_BUCKET" "$OUTPUT_MANIFEST_KEY"
printf "snapshot %s uploaded to %s in %d seconds (size: %d MB, concurrency: %d)\n" "$SNAPSHOT_BASENAME" "$DEST_S3" "$UPLOAD_DURATION_SEC" "$LATEST_SNAPSHOT_SIZE_IN_MB" "$CAND_CONC"