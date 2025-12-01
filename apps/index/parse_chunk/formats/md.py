import os
import sys
import json
import time
import logging
import hashlib
import tempfile
import unicodedata
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import boto3
from botocore.exceptions import ClientError
from markdown_it import MarkdownIt
try:
    import colorama
    colorama.init()
except Exception:
    pass
try:
    import tiktoken
except Exception:
    tiktoken = None
RESET = "\033[0m"
COLORS = {logging.DEBUG: "\033[90m", logging.INFO: "\033[37m", logging.WARNING: "\033[33m", logging.ERROR: "\033[31m", logging.CRITICAL: "\033[1;41m"}
class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = COLORS.get(record.levelno, RESET)
        message = super().format(record)
        return f"{color}{message}{RESET}"
log = logging.getLogger("md_parser")
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))
log.handlers[:] = [handler]
REQUIRED = ["S3_BUCKET", "S3_RAW_PREFIX", "S3_CHUNKED_PREFIX", "CHUNK_FORMAT"]
missing = [v for v in REQUIRED if os.getenv(v) is None]
if missing:
    sys.exit(f"ERROR: Missing env vars: {', '.join(missing)}")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT", "json").lower()
assert CHUNK_FORMAT in ("json", "jsonl")
MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
MIN_TOKENS_PER_CHUNK = int(os.getenv("MIN_TOKENS_PER_CHUNK", "100"))
DEFAULT_OVERLAP = max(1, int(MAX_TOKENS_PER_CHUNK * 0.1))
OVERLAP_TOKENS = int(os.getenv("OVERLAP_TOKENS", str(DEFAULT_OVERLAP)))
if OVERLAP_TOKENS >= MAX_TOKENS_PER_CHUNK:
    OVERLAP_TOKENS = max(1, MAX_TOKENS_PER_CHUNK - 1)
ENC_NAME = os.getenv("TOKEN_ENCODER", "cl100k_base")
PARSER_VERSION = os.getenv("PARSER_VERSION_MD", "markdown-it-py-v1")
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "false").lower() == "true"
SAVE_SNAPSHOT = os.getenv("SAVE_SNAPSHOT", "false").lower() == "true"
S3_PUT_RETRIES = int(os.getenv("S3_PUT_RETRIES", "3"))
S3_PUT_BACKOFF = float(os.getenv("S3_PUT_BACKOFF", "0.3"))
s3 = boto3.client("s3")
try:
    enc = tiktoken.get_encoding(ENC_NAME)
except Exception:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
md_parser = MarkdownIt()
def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def canonicalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r'[ \t]+$', '', ln) for ln in s.split("\n")]
    return "\n".join(lines).strip()
def try_decode_bytes(b: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return b.decode(encoding)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")
def token_count_for(text: str) -> int:
    if not text:
        return 0
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text.split())
def s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False
def s3_put_object_with_retries(key: str, body: bytes) -> None:
    for attempt in range(1, S3_PUT_RETRIES + 1):
        try:
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=body, ContentType="application/json")
            return
        except Exception:
            if attempt < S3_PUT_RETRIES:
                time.sleep(S3_PUT_BACKOFF * attempt)
    raise Exception("s3 put failed after retries")
def _is_rootish(h: Any) -> bool:
    if h is None:
        return True
    try:
        return str(h).strip().lower() in ("", "root")
    except Exception:
        return False
def build_header_sections(raw_text: str) -> List[Dict[str, Any]]:
    lines = raw_text.splitlines(keepends=True)
    try:
        tokens = md_parser.parse(raw_text)
    except Exception:
        return [{"heading_path": [], "heading": "", "level": 0, "start_line": 0, "end_line": len(lines), "lines": lines}]
    stack = [{"heading_path": [], "heading": "", "level": 0, "start_line": None, "end_line": None}]
    sections_out: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        ttype = getattr(tok, "type", "")
        if ttype == "heading_open":
            tag = getattr(tok, "tag", "h1")
            try:
                level = int(tag[1])
            except Exception:
                level = 1
            map_tuple = getattr(tok, "map", None)
            heading_text = ""
            if i + 1 < len(tokens) and getattr(tokens[i + 1], "type", "") == "inline":
                heading_text = getattr(tokens[i + 1], "content", "").strip()
            while stack and stack[-1]["level"] >= level:
                completed = stack.pop()
                if completed.get("start_line") is not None:
                    sections_out.append(completed)
            parent_path = [p for p in (stack[-1]["heading_path"][:] if stack else []) if not _is_rootish(p)]
            new_path = parent_path + ([] if _is_rootish(heading_text) else [heading_text])
            sec = {"heading_path": new_path, "heading": "" if _is_rootish(heading_text) else heading_text, "level": level, "start_line": None, "end_line": None}
            if map_tuple:
                sec["start_line"] = map_tuple[0]
                sec["end_line"] = map_tuple[1]
            stack.append(sec)
            i += 1
            continue
        map_tuple = getattr(tok, "map", None)
        if map_tuple:
            sline, eline = map_tuple[0], map_tuple[1]
            top = stack[-1]
            if top.get("start_line") is None or sline < top["start_line"]:
                top["start_line"] = sline
            if top.get("end_line") is None or eline > top["end_line"]:
                top["end_line"] = eline
        i += 1
    while stack:
        completed = stack.pop()
        if completed.get("start_line") is not None:
            sections_out.append(completed)
    total_lines = len(lines)
    normalized_sections = []
    for sec in sections_out:
        s = sec.get("start_line")
        e = sec.get("end_line")
        if s is None:
            continue
        s = max(0, s)
        e = min(total_lines, e)
        if s >= e and s < total_lines:
            e = s + 1
        heading_path = [h for h in sec.get("heading_path", []) if not _is_rootish(h)]
        heading = "" if _is_rootish(sec.get("heading", "")) else sec.get("heading", "")
        sec_copy = {"heading_path": heading_path, "heading": heading, "level": sec.get("level", 0), "start_line": s, "end_line": e, "lines": lines[s:e]}
        normalized_sections.append(sec_copy)
    normalized_sections_sorted = sorted(normalized_sections, key=lambda x: (x["start_line"], x["end_line"]))
    merged: List[Dict[str, Any]] = []
    last_end = 0
    if normalized_sections_sorted:
        first_start = normalized_sections_sorted[0]["start_line"]
        if first_start > 0:
            merged.append({"heading_path": [], "heading": "", "level": 0, "start_line": 0, "end_line": first_start, "lines": lines[0:first_start]})
    for sec in normalized_sections_sorted:
        if sec["start_line"] > last_end:
            gap_start = last_end
            gap_end = sec["start_line"]
            if gap_end > gap_start:
                merged.append({"heading_path": [], "heading": "", "level": 0, "start_line": gap_start, "end_line": gap_end, "lines": lines[gap_start:gap_end]})
        merged.append(sec)
        last_end = max(last_end, sec["end_line"])
    if last_end < total_lines:
        merged.append({"heading_path": [], "heading": "", "level": 0, "start_line": last_end, "end_line": total_lines, "lines": lines[last_end:total_lines]})
    return merged
def merge_small_sections(sections: List[Dict[str, Any]], merge_threshold: int, max_tokens: int, line_token_cache: Dict[int, int], prevent_merge_across_level: bool = False) -> List[Dict[str, Any]]:
    merged = []
    i = 0
    n = len(sections)
    while i < n:
        sec = sections[i]
        start_line = sec["start_line"]
        end_line = sec["end_line"]
        lines_acc = list(sec.get("lines", []))
        headings_acc = [] if _is_rootish(sec.get("heading", "")) else [sec.get("heading", "")]
        heading_path = [h for h in (sec.get("heading_path", []) or []) if not _is_rootish(h)]
        level = sec.get("level", 0)
        token_sum = 0
        for idx, l in enumerate(lines_acc):
            abs_idx = start_line + idx
            if abs_idx in line_token_cache:
                cnt = line_token_cache[abs_idx]
            else:
                try:
                    cnt = len(enc.encode(l)) if enc else len(l.split())
                except Exception:
                    cnt = len(l.split())
                line_token_cache[abs_idx] = cnt
            token_sum += cnt
        if token_sum >= merge_threshold:
            merged.append({"heading_path": heading_path, "headings": [h for h in headings_acc if not _is_rootish(h)], "level": level, "start_line": start_line, "end_line": end_line, "lines": lines_acc, "token_count": token_sum})
            i += 1
            continue
        if merged:
            prev = merged[-1]
            if not (prevent_merge_across_level and level <= prev.get("level", 0)):
                if prev.get("token_count", 0) + token_sum <= max_tokens:
                    prev["lines"].extend(lines_acc)
                    prev["end_line"] = end_line
                    if not _is_rootish(sec.get("heading", "")):
                        prev_headings = prev.get("headings", [])
                        prev_headings.append(sec.get("heading", ""))
                        prev["headings"] = [h for h in prev_headings if not _is_rootish(h)]
                    prev["token_count"] = prev.get("token_count", 0) + token_sum
                    i += 1
                    continue
        j = i + 1
        while j < n:
            next_sec = sections[j]
            if prevent_merge_across_level and next_sec.get("level", 0) <= level:
                break
            next_start = next_sec["start_line"]
            next_lines = next_sec.get("lines", [])
            next_tokens = 0
            for idx, l in enumerate(next_lines):
                abs_idx = next_start + idx
                if abs_idx in line_token_cache:
                    cnt = line_token_cache[abs_idx]
                else:
                    try:
                        cnt = len(enc.encode(l)) if enc else len(l.split())
                    except Exception:
                        cnt = len(l.split())
                    line_token_cache[abs_idx] = cnt
                next_tokens += cnt
            if token_sum + next_tokens > max_tokens:
                break
            token_sum += next_tokens
            lines_acc = lines_acc + next_lines
            nh = next_sec.get("heading", "")
            if not _is_rootish(nh):
                headings_acc.append(nh)
            end_line = next_sec["end_line"]
            j += 1
            if token_sum >= merge_threshold:
                break
        merged.append({"heading_path": heading_path, "headings": [h for h in headings_acc if not _is_rootish(h)], "level": level, "start_line": start_line, "end_line": end_line, "lines": lines_acc, "token_count": token_sum})
        i = max(j, i + 1)
    return merged
def split_long_line_into_char_windows(line: str, max_tokens: int, overlap_tokens: int) -> List[Dict[str, Any]]:
    pieces = []
    approx_char_per_token = max(1, len(line) // max(1, token_count_for(line)))
    window_chars = max(200, approx_char_per_token * max_tokens)
    step_chars = max(1, window_chars - approx_char_per_token * overlap_tokens)
    start = 0
    idx = 1
    cap = 1000
    while start < len(line) and idx <= cap:
        end = min(len(line), start + window_chars)
        piece = line[start:end]
        pieces.append({"text": piece, "token_count": token_count_for(piece), "subchunk_index": idx})
        idx += 1
        if end >= len(line):
            break
        start = start + step_chars
    return pieces
def split_section_by_tokens_lines(section: Dict[str, Any], overlap_tokens: int, max_tokens: int, line_token_cache: Dict[int, int]) -> List[Dict[str, Any]]:
    lines = section["lines"]
    base_start_line = section["start_line"]
    token_counts = []
    for idx, l in enumerate(lines):
        abs_idx = base_start_line + idx
        if abs_idx in line_token_cache:
            token_counts.append(line_token_cache[abs_idx])
        else:
            try:
                cnt = len(enc.encode(l)) if enc else len(l.split())
            except Exception:
                cnt = len(l.split())
            line_token_cache[abs_idx] = cnt
            token_counts.append(cnt)
    n = len(lines)
    chunks = []
    ptr = 0
    sub_idx = 1
    while ptr < n:
        current_tokens = 0
        j = ptr
        while j < n:
            next_tokens = token_counts[j]
            if current_tokens + next_tokens > max_tokens and current_tokens > 0:
                break
            current_tokens += next_tokens
            j += 1
        if j == ptr:
            line_idx = ptr
            line_text = lines[line_idx]
            long_pieces = split_long_line_into_char_windows(line_text, max_tokens, overlap_tokens)
            for p in long_pieces:
                chunk_start_line = base_start_line + line_idx
                chunk_end_line = chunk_start_line + 1
                chunk_text = p["text"]
                chunks.append({"text": canonicalize_text(chunk_text), "token_count": token_count_for(chunk_text), "start_line": chunk_start_line, "end_line": chunk_end_line, "subchunk_index": p["subchunk_index"]})
                sub_idx += 1
            ptr = ptr + 1
            continue
        chunk_start_line = base_start_line + ptr
        chunk_end_line = base_start_line + j
        chunk_text = "".join(lines[ptr:j]).strip()
        chunks.append({"text": canonicalize_text(chunk_text), "token_count": current_tokens, "start_line": chunk_start_line, "end_line": chunk_end_line, "subchunk_index": sub_idx})
        sub_idx += 1
        if overlap_tokens <= 0:
            next_ptr = j
        else:
            back_sum = 0
            back_idx = j - 1
            min_back_idx = ptr
            while back_idx >= min_back_idx and back_sum < overlap_tokens:
                back_sum += token_counts[back_idx]
                back_idx -= 1
            overlap_start = max(ptr, back_idx + 1)
            next_ptr = overlap_start
            if next_ptr <= ptr:
                next_ptr = j
        ptr = next_ptr
    return chunks
class S3DocWriter:
    def __init__(self, doc_id: str, s3_path: str, ext: str, content_type: str = "application/json"):
        self.doc_id = doc_id
        self.s3_path = s3_path or ""
        self.ext = ext
        self.content_type = content_type
        self.temp = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=f".{ext}", dir="/tmp")
        self.count = 0
        self._first = True
        if self.ext == "json":
            self.temp.write(b"[\n")
            self.temp.flush()
    def write_payload(self, payload: Dict[str, Any]) -> int:
        self.count += 1
        if self.ext == "jsonl":
            line = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            self.temp.write(line)
        else:
            pretty = json.dumps(payload, ensure_ascii=False, indent=2)
            indented = ("\n".join("  " + ln for ln in pretty.splitlines()) + "\n").encode("utf-8")
            if not self._first:
                self.temp.write(b",\n")
            self.temp.write(indented)
            self._first = False
        self.temp.flush()
        return 1
    def finalize_and_upload(self, out_key: str) -> Tuple[int, str]:
        if self.ext == "json":
            self.temp.write(b"]\n")
        self.temp.flush()
        self.temp.close()
        try:
            s3.upload_file(self.temp.name, S3_BUCKET, out_key, ExtraArgs={"ContentType": self.content_type})
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            return self.count, out_key
        except Exception:
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            raise
def sanitize_payload_for_weaviate(payload: Dict[str, Any]) -> None:
    for k in list(payload.keys()):
        v = payload.get(k)
        if isinstance(v, (int, float, bool)):
            continue
        if k == "tags":
            if v is None:
                payload[k] = []
            elif isinstance(v, (list, tuple)):
                payload[k] = [str(x) for x in v]
            else:
                payload[k] = [str(v)]
            continue
        if v is None:
            payload.pop(k, None)
            continue
        if isinstance(v, (list, tuple, dict)):
            try:
                payload[k] = json.dumps(v, ensure_ascii=False)
            except Exception:
                payload[k] = str(v)
            continue
        if not isinstance(v, str):
            payload[k] = str(v)
def parse_file(s3_key: str, manifest: dict) -> dict:
    start_all = time.perf_counter()
    try:
        head_obj = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        log.error("Could not HEAD S3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0, "skipped": True, "error": str(e)}
    last_modified = head_obj.get("LastModified", "")
    etag = head_obj.get("ETag", "")
    if isinstance(etag, str):
        etag = etag.strip('"')
    content_len = head_obj.get("ContentLength", 0) or 0
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        doc_id = manifest.get("file_hash")
    else:
        if etag:
            doc_id = sha256_hex(s3_key + str(etag))
        else:
            doc_id = sha256_hex(s3_key + str(last_modified or ""))
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("Skipping entire file because chunked file exists: %s", out_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    if content_len == 0:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("Skipping empty object %s (zero bytes).", s3_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception as e:
        log.error("Could not GET S3 object %s: %s", s3_key, e)
        return {"saved_chunks": 0, "total_parse_duration_ms": 0, "skipped": True, "error": str(e)}
    raw_body = obj["Body"].read()
    raw_text = try_decode_bytes(raw_body)
    doc_id = manifest.get("file_hash") or doc_id
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    snapshot_path = ""
    if SAVE_SNAPSHOT:
        try:
            key = f"{S3_CHUNKED_PREFIX}{doc_id}.snapshot.md"
            s3.put_object(Bucket=S3_BUCKET, Key=key, Body=raw_text.encode("utf-8"), ContentType="text/markdown")
            snapshot_path = f"s3://{S3_BUCKET}/{key}"
        except Exception:
            snapshot_path = ""
    canonical_full = canonicalize_text(raw_text)
    sections = build_header_sections(canonical_full)
    line_token_cache: Dict[int, int] = {}
    merged_sections = merge_small_sections(sections, MIN_TOKENS_PER_CHUNK, MAX_TOKENS_PER_CHUNK, line_token_cache)
    saved = 0
    chunk_index = 1
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("Skipping entire file because chunked file exists (post-download): %s", out_key)
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True}
    writer = S3DocWriter(doc_id=doc_id, s3_path=s3_path, ext=ext)
    file_name = os.path.basename(s3_key)
    try:
        for sec in merged_sections:
            sec_lines = sec.get("lines", [])
            if not sec_lines:
                continue
            sec_text = "".join(sec_lines).strip()
            sec_token_count = sec.get("token_count", token_count_for(sec_text))
            heading_path = [h for h in (sec.get("heading_path", []) or []) if not _is_rootish(h)]
            headings_raw = sec.get("headings") or []
            headings = [h for h in headings_raw if not _is_rootish(h)]
            if not headings and heading_path:
                headings = list(heading_path)
            sec_start_line = sec.get("start_line", 0)
            sec_end_line = sec.get("end_line", sec_start_line)
            start_line_1b = sec_start_line + 1
            end_line_1b = sec_end_line
            if sec_token_count <= MAX_TOKENS_PER_CHUNK:
                chunk_id = f"{doc_id}_{chunk_index}"
                chunk_index += 1
                payload = {
                    "document_id": doc_id or "",
                    "file_name": file_name,
                    "chunk_id": chunk_id or "",
                    "chunk_type": "md_section",
                    "text": canonicalize_text(sec_text) or "",
                    "token_count": int(sec_token_count or 0),
                    "figures": "[]",
                    "embedding": None,
                    "file_type": "text/markdown",
                    "source_url": s3_path,
                    "page_number": None,
                    "slide_range": None,
                    "row_range": None,
                    "token_range": None,
                    "audio_range": None,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "parser_version": PARSER_VERSION or "",
                    "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                    "layout_tags": [],
                    "used_ocr": False,
                    "heading_path": heading_path or [],
                    "headings": headings or [],
                    "line_range": [int(start_line_1b), int(end_line_1b)] if start_line_1b and end_line_1b is not None else None,
                }
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                saved += 1
                log.info("Buffered chunk %s", payload["chunk_id"])
            else:
                subchunks = split_section_by_tokens_lines(sec, OVERLAP_TOKENS, MAX_TOKENS_PER_CHUNK, line_token_cache)
                for sub in subchunks:
                    chunk_text = sub.get("text", "")
                    token_ct = int(sub.get("token_count", 0))
                    sline = sub.get("start_line", 0)
                    eline = sub.get("end_line", sline)
                    chunk_id = f"{doc_id}_{chunk_index}"
                    chunk_index += 1
                    start_line_sub = sline + 1
                    end_line_sub = eline
                    payload = {
                        "document_id": doc_id or "",
                        "file_name": file_name,
                        "chunk_id": chunk_id or "",
                        "chunk_type": "md_subchunk",
                        "text": canonicalize_text(chunk_text) or "",
                        "token_count": token_ct,
                        "figures": "[]",
                        "embedding": None,
                        "file_type": "text/markdown",
                        "source_url": s3_path,
                        "page_number": None,
                        "slide_range": None,
                        "row_range": None,
                        "token_range": None,
                        "audio_range": None,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "parser_version": PARSER_VERSION or "",
                        "tags": manifest.get("tags", []) if isinstance(manifest, dict) else [],
                        "layout_tags": [],
                        "used_ocr": False,
                        "heading_path": heading_path or [],
                        "headings": headings or [],
                        "line_range": [int(start_line_sub), int(end_line_sub)] if start_line_sub and end_line_sub is not None else None,
                    }
                    sanitize_payload_for_weaviate(payload)
                    writer.write_payload(payload)
                    saved += 1
                    log.info("Buffered subchunk %s (lines %d-%d)", payload["chunk_id"], start_line_sub, end_line_sub)
    except Exception as e:
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.exception("Error while buffering chunks for %s: %s", s3_key, str(e))
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e)}
    try:
        if saved == 0:
            try:
                if writer and writer.temp:
                    os.unlink(writer.temp.name)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            log.info("No chunks produced for %s", s3_key)
            return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": False}
        count, uploaded_key = writer.finalize_and_upload(out_key)
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.info("Wrote %d chunks for %s → %s (%d ms)", count, s3_key, uploaded_key, total_ms)
        return {"saved_chunks": count, "total_parse_duration_ms": total_ms, "skipped": False}
    except Exception as e_up:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        log.error("Failed to upload chunked file for %s error=%s", s3_key, str(e_up))
        try:
            if writer and writer.temp:
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        return {"saved_chunks": 0, "total_parse_duration_ms": total_ms, "skipped": True, "error": str(e_up)}
if __name__ == "__main__":
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.lower().endswith(".md") or key.lower().endswith(".markdown")):
                continue
            log.info("Routing parse_file for s3://%s/%s", S3_BUCKET, key)
            manifest_key = key + ".manifest.json"
            try:
                mf = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key)
                manifest = json.load(mf["Body"])
            except Exception:
                manifest = {}
            try:
                parse_file(key, manifest)
            except Exception as e:
                log.exception("Failed to parse %s: %s", key, e)
