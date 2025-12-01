#!/usr/bin/env bash

S3_BUCKET="${S3_BUCKET:?S3_BUCKET required}"
S3_RAW_PREFIX="${S3_RAW_PREFIX:-data/raw/}"
S3_RAW_PREFIX="${S3_RAW_PREFIX#/}"
S3_RAW_PREFIX="${S3_RAW_PREFIX%/}/"
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
mkdir -p "$TMP_DIR/src" "$TMP_DIR/out"
for cmd in "$AWSCLI" "$JQ_BIN" "$FFMPEG_BIN" "$SOFFICE_BIN"; do command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd not found" >&2; exit 2; }; done
if command -v java >/dev/null 2>&1; then JAVA_BIN="$(readlink -f "$(which java)" 2>/dev/null || true)"; [ -n "${JAVA_BIN:-}" ] && export JAVA_HOME="$(dirname "$(dirname "$JAVA_BIN")")"; fi
command -v javaldx >/dev/null 2>&1 || true
audio_exts="mp3 m4a aac wav flac ogg opus webm amr wma aiff aif"
sheet_exts="xls xlsx ods xlsm xlsb"
doc_exts="doc docx"
list_keys(){ "$AWSCLI" s3api list-objects-v2 --bucket "$S3_BUCKET" --prefix "$S3_RAW_PREFIX" --output json | "$JQ_BIN" -r '.Contents[]?.Key' || true; }
s3_cp_down(){ "$AWSCLI" s3 cp "s3://$S3_BUCKET/$1" "$2"; }
s3_upload(){ "$AWSCLI" s3 cp "$1" "s3://$S3_BUCKET/$2"; }
s3_delete(){ "$AWSCLI" s3 rm "s3://$S3_BUCKET/$1"; }
to_lower(){ echo "$1" | tr '[:upper:]' '[:lower:]'; }
basename_no_ext(){ b="$(basename "$1")"; echo "${b%.*}"; }
s3_object_exists(){ key="$1"; if [ -n "${AWS_REGION:-}" ]; then "$AWSCLI" s3api head-object --bucket "$S3_BUCKET" --key "$key" --region "$AWS_REGION" >/dev/null 2>&1; else "$AWSCLI" s3api head-object --bucket "$S3_BUCKET" --key "$key" >/dev/null 2>&1; fi; }
process_audio(){ key="$1"; name="$(basename "$key")"; ext="$(to_lower "${name##*.}")"; [ -z "$(echo " $audio_exts " | grep " $ext ")" ] && return; if [ "$ext" = "wav" ]; then s3_target_key="${S3_RAW_PREFIX%/}/audio/${name}"; else s3_target_key="${S3_RAW_PREFIX%/}/audio/$(basename_no_ext "$name").wav"; fi; if [ "${OVERWRITE_ALL_AUDIO_FILES:-false}" != "true" ]; then if [ "$s3_target_key" = "$key" ]; then return; fi; if s3_object_exists "$s3_target_key"; then return; fi; fi; local_src="$TMP_DIR/src/$name"; local_out="$TMP_DIR/out/$(basename_no_ext "$name").wav"; s3_cp_down "$key" "$local_src"; if [ "$ext" = "wav" ]; then if command -v ffprobe >/dev/null 2>&1; then sr=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "$local_src" 2>/dev/null || echo ""); ch=$(ffprobe -v error -select_streams a:0 -show_entries stream=channels -of default=noprint_wrappers=1:nokey=1 "$local_src" 2>/dev/null || echo ""); fi; if [ "${sr:-}" = "16000" ] && [ "${ch:-}" = "1" ]; then cp -f "$local_src" "$local_out"; s3_upload "$local_out" "$s3_target_key"; [ "${OVERWRITE_ALL_AUDIO_FILES}" = "true" ] && [ "$s3_target_key" != "$key" ] && s3_delete "$key"; rm -f "$local_src" "$local_out"; return; fi; fi; tmp_out="$(mktemp "$TMP_DIR/out/tmpout.XXXXXX.wav")"; "$FFMPEG_BIN" -y -hide_banner -loglevel error -i "$local_src" -ar 16000 -ac 1 -sample_fmt s16 "$tmp_out"; mv -f "$tmp_out" "$local_out"; s3_upload "$local_out" "$s3_target_key"; [ "${OVERWRITE_ALL_AUDIO_FILES}" = "true" ] && [ "$s3_target_key" != "$key" ] && s3_delete "$key"; rm -f "$local_src" "$local_out"; }
process_doc(){ key="$1"; name="$(basename "$key")"; ext="$(to_lower "${name##*.}")"; [ -z "$(echo " $doc_exts " | grep " $ext ")" ] && return; local_src="$TMP_DIR/src/$name"; s3_cp_down "$key" "$local_src"; "$SOFFICE_BIN" --headless --invisible --nologo --nodefault --nofirststartwizard --nolockcheck --convert-to pdf:writer_pdf_Export --outdir "$TMP_DIR/out" "$local_src"; out_pdf="$TMP_DIR/out/$(basename_no_ext "$name").pdf"; [ ! -f "$out_pdf" ] && for f in "$TMP_DIR/out"/*.pdf; do [ -f "$f" ] && out_pdf="$f" && break; done; if [ -f "$out_pdf" ]; then s3_target_key="${S3_RAW_PREFIX%/}/pdfs/${name}.pdf"; s3_upload "$out_pdf" "$s3_target_key"; [ "${OVERWRITE_OTHER_TO_PDF}" = "true" ] && s3_delete "$key"; fi; rm -f "$local_src" "$out_pdf"; }
process_sheet(){ key="$1"; name="$(basename "$key")"; ext="$(to_lower "${name##*.}")"; [ -z "$(echo " $sheet_exts " | grep " $ext ")" ] && return; local_src="$TMP_DIR/src/$name"; s3_cp_down "$key" "$local_src"; "$SOFFICE_BIN" --headless --invisible --nologo --nodefault --nofirststartwizard --nolockcheck --convert-to csv --outdir "$TMP_DIR/out" "$local_src"; shopt -s nullglob; for f in "$TMP_DIR/out"/*.csv; do base_csv="$(basename "$f")"; s3_upload "$f" "${S3_RAW_PREFIX%/}/csvs/${name}.${base_csv}"; done; [ "${OVERWRITE_SPREADSHEETS_WITH_CSV}" = "true" ] && s3_delete "$key"; rm -f "$local_src" "$TMP_DIR/out"/*.csv; }
group_remaining(){ list_keys | while IFS= read -r key; do [[ -z "$key" ]] && continue; [[ "$key" == */ ]] && continue; [[ "$key" == *.manifest.json ]] && continue; name="$(basename "$key")"; ext="$(to_lower "${name##*.}")"; case "$ext" in mp3|m4a|aac|wav|flac|ogg|opus|webm|amr|wma|aiff|aif) sub="audio/";; jpg|jpeg|png|webp|tif|tiff|bmp|gif) sub="images/";; pdf) sub="pdfs/";; doc|docx) sub="docs/";; ppt|pptx) sub="ppts/";; txt) sub="txts/";; csv) sub="csvs/";; md) sub="mds/";; html) sub="htmls/";; jsonl) sub="jsonls/";; *) sub="others/";; esac; dst="${S3_RAW_PREFIX}${sub}${name}"; [ "$dst" != "$key" ] && "$AWSCLI" s3 mv "s3://$S3_BUCKET/$key" "s3://$S3_BUCKET/$dst" || true; done; }
cleanup(){ rm -rf "$TMP_DIR/src" "$TMP_DIR/out" || true; }
trap cleanup EXIT
main_loop(){ list_keys | while IFS= read -r key; do [[ -z "$key" ]] && continue; [[ "$key" == */ ]] && continue; [[ "$key" == *.manifest.json ]] && continue; ext="$(to_lower "${key##*.}")"; if echo " $audio_exts " | grep -q " $ext "; then process_audio "$key"; elif echo " $doc_exts " | grep -q " $ext "; then process_doc "$key"; elif echo " $sheet_exts " | grep -q " $ext "; then process_sheet "$key"; fi; done; group_remaining; }
main_loop
