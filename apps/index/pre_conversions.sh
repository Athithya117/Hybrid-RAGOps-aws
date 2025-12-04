#!/usr/bin/env bash
set -u

# Robust pre-conversion script
# - skips missing audio files (no fatal head-object errors)
# - skips conversions gracefully when tools aren't available
# - warns on failures but continues processing remaining objects

S3_BUCKET="${S3_BUCKET:?S3_BUCKET required}"
S3_RAW_PREFIX="${S3_RAW_PREFIX:-data/raw/}"
S3_RAW_PREFIX="${S3_RAW_PREFIX#/}"      # remove leading slash
S3_RAW_PREFIX="${S3_RAW_PREFIX%/}/"     # ensure trailing slash

AWS_REGION="${AWS_REGION:-}"
[ -n "${AWS_REGION:-}" ] && export AWS_DEFAULT_REGION="$AWS_REGION"

OVERWRITE_ALL_AUDIO_FILES="${OVERWRITE_ALL_AUDIO_FILES:-false}"
OVERWRITE_OTHER_TO_PDF="${OVERWRITE_OTHER_TO_PDF:-true}"
OVERWRITE_SPREADSHEETS_WITH_CSV="${OVERWRITE_SPREADSHEETS_WITH_CSV:-false}"

TMP_DIR="${TMP_DIR:-/tmp/preconv}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
SOFFICE_BIN="${SOFFICE_BIN:-soffice}"
AWSCLI="${AWSCLI:-aws}"
JQ_BIN="${JQ_BIN:-jq}"

# Prepare temp directories
mkdir -p "$TMP_DIR/src" "$TMP_DIR/out"

# Check required tools: aws & jq required; ffmpeg/soffice optional
missing=0
for cmd in "$AWSCLI" "$JQ_BIN"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    printf "[ERROR] required command not found: %s\n" "$cmd" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  printf "[ERROR] missing required commands, aborting\n" >&2
  exit 2
fi

HAVE_FFMPEG=0
if command -v "$FFMPEG_BIN" >/dev/null 2>&1; then
  HAVE_FFMPEG=1
else
  printf "[WARN] ffmpeg not found; audio transcoding will be skipped or limited\n" >&2
fi

HAVE_FFPROBE=0
if command -v ffprobe >/dev/null 2>&1; then
  HAVE_FFPROBE=1
fi

HAVE_SOFFICE=0
if command -v "$SOFFICE_BIN" >/dev/null 2>&1; then
  HAVE_SOFFICE=1
else
  printf "[WARN] soffice not found; document/spreadsheet conversions will be skipped\n" >&2
fi

# Java detection (optional)
if command -v java >/dev/null 2>&1; then
  JAVA_BIN="$(readlink -f "$(which java)" 2>/dev/null || true)"
  [ -n "${JAVA_BIN:-}" ] && export JAVA_HOME="$(dirname "$(dirname "$JAVA_BIN")")"
fi

# Known extensions
audio_exts="mp3 m4a aac wav flac ogg opus webm amr wma aiff aif"
sheet_exts="xls xlsx ods xlsm xlsb"
doc_exts="doc docx"

# Helper: list keys under prefix (safe, returns nothing on empty)
list_keys() {
  "$AWSCLI" s3api list-objects-v2 --bucket "$S3_BUCKET" --prefix "$S3_RAW_PREFIX" --output json \
    2>/dev/null | "$JQ_BIN" -r '.Contents[]?.Key' 2>/dev/null || true
}

# s3 helpers (return codes preserved)
s3_cp_down() {
  local src_key="$1"; local dest="$2"
  "$AWSCLI" s3 cp "s3://$S3_BUCKET/$src_key" "$dest"
}

s3_upload() {
  local src="$1"; local dest_key="$2"
  "$AWSCLI" s3 cp "$src" "s3://$S3_BUCKET/$dest_key"
}

s3_delete() {
  local key="$1"
  "$AWSCLI" s3 rm "s3://$S3_BUCKET/$key" >/dev/null 2>&1 || true
}

# reliable existence check: suppress AWS CLI error output, return 0 if exists else 1
s3_object_exists() {
  local key="$1"
  if [ -n "${AWS_REGION:-}" ]; then
    "$AWSCLI" s3api head-object --bucket "$S3_BUCKET" --key "$key" --region "$AWS_REGION" >/dev/null 2>&1
  else
    "$AWSCLI" s3api head-object --bucket "$S3_BUCKET" --key "$key" >/dev/null 2>&1
  fi
  return $?
}

# small utilities
to_lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }
basename_no_ext() { b="$(basename "$1")"; echo "${b%.*}"; }

# Process audio files: convert to 16k mono WAV unless already in correct format.
process_audio() {
  local key="$1"
  local name; name="$(basename "$key")"
  local ext; ext="$(to_lower "${name##*.}")"

  # only handle known audio exts
  if ! echo " $audio_exts " | grep -q " $ext "; then
    return
  fi

  local s3_target_key
  if [ "$ext" = "wav" ]; then
    s3_target_key="${S3_RAW_PREFIX%/}/audio/${name}"
  else
    s3_target_key="${S3_RAW_PREFIX%/}/audio/$(basename_no_ext "$name").wav"
  fi

  # skip if not overwriting and target equals source or already exists
  if [ "${OVERWRITE_ALL_AUDIO_FILES:-false}" != "true" ]; then
    if [ "$s3_target_key" = "$key" ]; then
      printf "[audio] target equals source, skipping: %s\n" "$key"
      return
    fi
    if s3_object_exists "$s3_target_key"; then
      printf "[audio] target already exists, skipping: %s\n" "$s3_target_key"
      return
    fi
  fi

  local local_src="$TMP_DIR/src/$name"
  local local_out="$TMP_DIR/out/$(basename_no_ext "$name").wav"

  # safe download: if missing, warn and skip (do NOT abort script)
  if ! s3_cp_down "$key" "$local_src" >/dev/null 2>&1; then
    printf "[WARN] failed to download %s; skipping audio processing\n" "$key" >&2
    rm -f "$local_src"
    return
  fi

  # If WAV already and correct sample rate / channels, avoid re-encoding
  if [ "$ext" = "wav" ] && [ "$HAVE_FFPROBE" -eq 1 ]; then
    sr=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate \
         -of default=noprint_wrappers=1:nokey=1 "$local_src" 2>/dev/null || echo "")
    ch=$(ffprobe -v error -select_streams a:0 -show_entries stream=channels \
         -of default=noprint_wrappers=1:nokey=1 "$local_src" 2>/dev/null || echo "")
    if [ "${sr:-}" = "16000" ] && [ "${ch:-}" = "1" ]; then
      cp -f "$local_src" "$local_out"
      if s3_upload "$local_out" "$s3_target_key" >/dev/null 2>&1; then
        [ "${OVERWRITE_ALL_AUDIO_FILES}" = "true" ] && [ "$s3_target_key" != "$key" ] && s3_delete "$key"
      else
        printf "[WARN] upload failed for %s -> %s\n" "$local_out" "$s3_target_key" >&2
      fi
      rm -f "$local_src" "$local_out"
      return
    fi
  fi

  # If ffmpeg not available, attempt raw upload for WAV, otherwise skip
  if [ "$HAVE_FFMPEG" -ne 1 ]; then
    if [ "$ext" = "wav" ]; then
      cp -f "$local_src" "$local_out"
      if s3_upload "$local_out" "$s3_target_key" >/dev/null 2>&1; then
        [ "${OVERWRITE_ALL_AUDIO_FILES}" = "true" ] && [ "$s3_target_key" != "$key" ] && s3_delete "$key"
      else
        printf "[WARN] upload failed for %s -> %s\n" "$local_out" "$s3_target_key" >&2
      fi
    else
      printf "[WARN] ffmpeg missing and source is not WAV — skipping audio: %s\n" "$key" >&2
    fi
    rm -f "$local_src" "$local_out"
    return
  fi

  # Otherwise re-encode to 16k mono WAV
  tmp_out="$(mktemp "$TMP_DIR/out/tmpout.XXXXXX.wav")"
  if "$FFMPEG_BIN" -y -hide_banner -loglevel error -i "$local_src" -ar 16000 -ac 1 -sample_fmt s16 "$tmp_out" >/dev/null 2>&1; then
    mv -f "$tmp_out" "$local_out"
    if s3_upload "$local_out" "$s3_target_key" >/dev/null 2>&1; then
      [ "${OVERWRITE_ALL_AUDIO_FILES}" = "true" ] && [ "$s3_target_key" != "$key" ] && s3_delete "$key"
    else
      printf "[WARN] upload failed for %s -> %s\n" "$local_out" "$s3_target_key" >&2
    fi
  else
    printf "[WARN] ffmpeg failed to transcode %s\n" "$local_src" >&2
    rm -f "$tmp_out"
  fi

  rm -f "$local_src" "$local_out"
}

# Convert docs (doc/docx) to PDF using soffice (if available)
process_doc() {
  local key="$1"
  local name; name="$(basename "$key")"
  local ext; ext="$(to_lower "${name##*.}")"
  if ! echo " $doc_exts " | grep -q " $ext "; then
    return
  fi

  if [ "$HAVE_SOFFICE" -ne 1 ]; then
    printf "[doc] soffice not available — skipping doc conversion for %s\n" "$key"
    return
  fi

  local local_src="$TMP_DIR/src/$name"
  if ! s3_cp_down "$key" "$local_src" >/dev/null 2>&1; then
    printf "[WARN] failed to download %s; skipping doc conversion\n" "$key" >&2
    rm -f "$local_src"
    return
  fi

  "$SOFFICE_BIN" --headless --invisible --nologo --nodefault --nofirststartwizard --nolockcheck \
    --convert-to pdf:writer_pdf_Export --outdir "$TMP_DIR/out" "$local_src" >/dev/null 2>&1 || true

  local out_pdf="$TMP_DIR/out/$(basename_no_ext "$name").pdf"
  # fallback: pick first produced pdf if conversion chooses another name
  if [ ! -f "$out_pdf" ]; then
    for f in "$TMP_DIR/out"/*.pdf; do
      [ -f "$f" ] && { out_pdf="$f"; break; }
    done
  fi

  if [ -f "$out_pdf" ]; then
    s3_target_key="${S3_RAW_PREFIX%/}/pdfs/${name}.pdf"
    if s3_upload "$out_pdf" "$s3_target_key" >/dev/null 2>&1; then
      [ "${OVERWRITE_OTHER_TO_PDF}" = "true" ] && s3_delete "$key"
    else
      printf "[WARN] failed to upload converted PDF for %s\n" "$key" >&2
    fi
  else
    printf "[WARN] conversion produced no PDF for %s\n" "$key" >&2
  fi

  rm -f "$local_src" "$out_pdf"
}

# Convert spreadsheets to CSV(s)
process_sheet() {
  local key="$1"
  local name; name="$(basename "$key")"
  local ext; ext="$(to_lower "${name##*.}")"
  if ! echo " $sheet_exts " | grep -q " $ext "; then
    return
  fi

  if [ "$HAVE_SOFFICE" -ne 1 ]; then
    printf "[sheet] soffice not available — skipping sheet conversion for %s\n" "$key"
    return
  fi

  local local_src="$TMP_DIR/src/$name"
  if ! s3_cp_down "$key" "$local_src" >/dev/null 2>&1; then
    printf "[WARN] failed to download %s; skipping sheet conversion\n" "$key" >&2
    rm -f "$local_src"
    return
  fi

  "$SOFFICE_BIN" --headless --invisible --nologo --nodefault --nofirststartwizard --nolockcheck \
    --convert-to csv --outdir "$TMP_DIR/out" "$local_src" >/dev/null 2>&1 || true

  shopt -s nullglob
  for f in "$TMP_DIR/out"/*.csv; do
    base_csv="$(basename "$f")"
    if ! s3_upload "$f" "${S3_RAW_PREFIX%/}/csvs/${name}.${base_csv}" >/dev/null 2>&1; then
      printf "[WARN] failed to upload CSV %s\n" "$f" >&2
    fi
  done
  shopt -u nullglob

  [ "${OVERWRITE_SPREADSHEETS_WITH_CSV}" = "true" ] && s3_delete "$key"
  rm -f "$local_src" "$TMP_DIR/out"/*.csv
}

# Move remaining objects into categorized subfolders under S3_RAW_PREFIX
group_remaining() {
  list_keys | while IFS= read -r key; do
    [ -z "${key:-}" ] && continue
    [[ "$key" == */ ]] && continue
    [[ "$key" == *.manifest.json ]] && continue

    name="$(basename "$key")"
    ext="$(to_lower "${name##*.}")"

    case "$ext" in
      mp3|m4a|aac|wav|flac|ogg|opus|webm|amr|wma|aiff|aif) sub="audio/";;
      jpg|jpeg|png|webp|tif|tiff|bmp|gif) sub="images/";;
      pdf) sub="pdfs/";;
      doc|docx) sub="docs/";;
      ppt|pptx) sub="ppts/";;
      txt) sub="txts/";;
      csv) sub="csvs/";;
      md) sub="mds/";;
      html) sub="htmls/";;
      jsonl) sub="jsonls/";;
      *) sub="others/";;
    esac

    dst="${S3_RAW_PREFIX}${sub}${name}"
    if [ "$dst" != "$key" ]; then
      if ! "$AWSCLI" s3 mv "s3://$S3_BUCKET/$key" "s3://$S3_BUCKET/$dst" >/dev/null 2>&1; then
        printf "[WARN] failed to move s3://%s/%s -> s3://%s/%s\n" "$S3_BUCKET" "$key" "$S3_BUCKET" "$dst" >&2
      fi
    fi
  done
}

# Cleanup temp dirs
cleanup() { rm -rf "$TMP_DIR/src" "$TMP_DIR/out" || true; }
trap cleanup EXIT

# Main processing loop
main_loop() {
  # iterate safely even if list_keys yields nothing
  list_keys | while IFS= read -r key; do
    [ -z "${key:-}" ] && continue
    [[ "$key" == */ ]] && continue
    [[ "$key" == *.manifest.json ]] && continue

    ext="$(to_lower "${key##*.}")"

    if echo " $audio_exts " | grep -q " $ext "; then
      process_audio "$key"
    elif echo " $doc_exts " | grep -q " $ext "; then
      process_doc "$key"
    elif echo " $sheet_exts " | grep -q " $ext "; then
      process_sheet "$key"
    else
      # do nothing for other files here; group_remaining will organise them
      :
    fi
  done

  group_remaining
}

main_loop
