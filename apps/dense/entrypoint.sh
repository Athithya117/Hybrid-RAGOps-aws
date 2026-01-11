#!/bin/sh
set -eu

# Ensure baked files exist
if [ ! -f /app/.baked_dense_model ] || [ ! -f /app/.baked_dense_dim ]; then
  echo "ERROR: baked model/dim files missing from image" >&2
  exit 1
fi

BAKED_MODEL=$(cat /app/.baked_dense_model)
BAKED_DIM=$(cat /app/.baked_dense_dim)

# Current environment values (may be unset)
CUR_MODEL="${DENSE_MODEL_NAME:-}"
CUR_DIM="${DENSE_DIM:-}"

# If runtime variables are empty, treat them as the baked values (common when no env override)
if [ -z "$CUR_MODEL" ]; then CUR_MODEL="$BAKED_MODEL"; fi
if [ -z "$CUR_DIM" ]; then CUR_DIM="$BAKED_DIM"; fi

if [ "$CUR_MODEL" != "$BAKED_MODEL" ] || [ "$CUR_DIM" != "$BAKED_DIM" ]; then
  echo "ERROR: runtime DENSE_MODEL_NAME/DENSE_DIM must match the baked image values" >&2
  echo "  Baked:   DENSE_MODEL_NAME=${BAKED_MODEL}, DENSE_DIM=${BAKED_DIM}" >&2
  echo "  Runtime: DENSE_MODEL_NAME=${CUR_MODEL:-<unset>}, DENSE_DIM=${CUR_DIM:-<unset>}" >&2
  exit 1
fi

# Exec uvicorn with the baked/runtime host/port
exec uvicorn host_dense:app --host "${DENSE_HOST:-0.0.0.0}" --port "${DENSE_PORT:-8200}" --lifespan on
