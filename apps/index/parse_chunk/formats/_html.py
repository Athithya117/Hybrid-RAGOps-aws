#!/usr/bin/env python3
import os, sys, json, time, logging, hashlib, tempfile, re, unicodedata
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Iterator, Tuple
from botocore.exceptions import ClientError
RESET="\033[0m"
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
log=logging.getLogger("html_trafilatura")
S3_BUCKET=os.getenv("S3_BUCKET")
S3_RAW_PREFIX=os.getenv("S3_RAW_PREFIX","data/raw/").rstrip("/")+"/"
S3_CHUNKED_PREFIX=os.getenv("S3_CHUNKED_PREFIX","data/chunked/").rstrip("/")+"/"
PARSER_VERSION=os.getenv("PARSER_VERSION_HTML","trafilatura-only-v2")
FORCE_OVERWRITE=os.getenv("FORCE_OVERWRITE","false").lower()=="true"
SAVE_SNAPSHOT=os.getenv("SAVE_SNAPSHOT","false").lower()=="true"
ENC_NAME=os.getenv("TOKEN_ENCODER","cl100k_base")
REQUEST_TIMEOUT=int(os.getenv("REQUEST_TIMEOUT","15"))
FETCH_RETRIES=int(os.getenv("FETCH_RETRIES","3"))
FETCH_BACKOFF=float(os.getenv("FETCH_BACKOFF","0.5"))
MAX_TOKENS_PER_CHUNK=int(os.getenv("MAX_TOKENS_PER_CHUNK","512"))
MIN_TOKENS_PER_CHUNK=int(os.getenv("MIN_TOKENS_PER_CHUNK","100"))
NUMBER_OF_OVERLAPPING_SENTENCES=int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES","2"))
if not S3_BUCKET:
    sys.exit("ERROR: S3_BUCKET not set")
try:
    import boto3
    s3=boto3.client("s3")
except Exception as e:
    log.error("boto3 required: %s",e); raise
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    log.error("pyarrow required (parquet-only pipeline)"); sys.exit(1)
try:
    import requests
except Exception:
    log.error("requests required"); sys.exit(1)
try:
    import trafilatura
except Exception:
    trafilatura=None
try:
    import tiktoken
    ENCODER=None
    try:
        ENCODER=tiktoken.get_encoding(ENC_NAME)
    except Exception:
        ENCODER=None
except Exception:
    ENCODER=None
try:
    import spacy
    from spacy.pipeline import Sentencizer
except Exception:
    spacy=None
    Sentencizer=None
def sha256_hex_bytes(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()
def sha256_hex_str(s:str)->str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def canonicalize_text(s:Any)->str:
    if not isinstance(s,str):
        s=str(s or "")
    s=unicodedata.normalize("NFKC",s)
    s=s.replace("\r\n","\n").replace("\r","\n")
    s=re.sub(r'\s+',' ',s).strip()
    return s
def s3_object_exists(key:str)->bool:
    try:
        s3.head_object(Bucket=S3_BUCKET,Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False
def fetch_html_with_retries(url:str,timeout:int=REQUEST_TIMEOUT,retries:int=FETCH_RETRIES,backoff:float=FETCH_BACKOFF)->str:
    last=None
    for attempt in range(1,retries+1):
        try:
            r=requests.get(url,timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            last=e
            if attempt<retries:
                time.sleep(backoff*attempt)
    raise last
def upload_snapshot_to_s3(snapshot_html:str,doc_id:str)->Optional[str]:
    if not SAVE_SNAPSHOT:
        return None
    key=f"{S3_CHUNKED_PREFIX}{doc_id}.snapshot.html"
    s3.put_object(Bucket=S3_BUCKET,Key=key,Body=snapshot_html.encode("utf-8"),ContentType="text/html")
    return f"s3://{S3_BUCKET}/{key}"
def trafilatura_extract_markdown(html_text:str)->Tuple[Optional[str],Dict[str,Any]]:
    if trafilatura is None:
        return None,{}
    try:
        md=trafilatura.extract(html_text,output_format="markdown",with_metadata=True)
    except Exception:
        md=None
    parsed={}
    try:
        json_doc=trafilatura.extract(html_text,output_format="json",with_metadata=True)
        if json_doc:
            parsed=json.loads(json_doc)
    except Exception:
        parsed={}
    return md,parsed
def _make_sentencizer():
    if spacy is None:
        return None
    try:
        nlp=spacy.blank("en")
        try:
            nlp.add_pipe("sentencizer")
        except Exception:
            if Sentencizer is not None:
                nlp.add_pipe(Sentencizer())
            else:
                nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        return None
_NLP_SENTENCIZER=_make_sentencizer()
def _regex_sentences_with_offsets(text:str):
    spans=[]
    pattern=re.compile(r'(.+?[\.\?\!]["\']?\s+)|(.+?$)',re.DOTALL)
    cursor=0
    for m in pattern.finditer(text):
        s=(m.group(1) or m.group(2) or "").strip()
        if not s:
            continue
        start=text.find(s,cursor)
        if start==-1:
            start=cursor
        end=start+len(s)
        spans.append((s,start,end))
        cursor=end
    return spans
def _sentences_with_offsets(text:str):
    if _NLP_SENTENCIZER is not None:
        doc=_NLP_SENTENCIZER(text)
        return [(sent.text.strip(),int(sent.start_char),int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
    return _regex_sentences_with_offsets(text)
def _make_encoder_clients():
    if ENCODER is not None:
        return (lambda txt: ENCODER.encode(txt), lambda toks: ENCODER.decode(toks), "tiktoken")
    return (lambda txt: txt.split(), lambda toks: " ".join(toks), "whitespace")
_ENCODER_ENCODE,_ENCODER_DECODE,_ENCODER_BACKEND=_make_encoder_clients()
def split_into_token_windows(text:str,max_tokens:int=MAX_TOKENS_PER_CHUNK,min_tokens:int=MIN_TOKENS_PER_CHUNK,overlap_sentences:int=NUMBER_OF_OVERLAPPING_SENTENCES)->Iterator[Dict[str,Any]]:
    if not text:
        yield {"window_index":0,"text":"","token_count":0,"token_start":0,"token_end":0}
        return
    text=canonicalize_text(text)
    sentences=_sentences_with_offsets(text)
    sent_items=[]
    for s,sc,ec in sentences:
        toks=_ENCODER_ENCODE(s)
        tok_len=len(toks)
        sent_items.append({"text":s,"start_char":sc,"end_char":ec,"token_len":tok_len,"tokens":toks})
    if not sent_items:
        all_toks=_ENCODER_ENCODE(text)
        yield {"window_index":0,"text":text,"token_count":len(all_toks),"token_start":0,"token_end":len(all_toks)}
        return
    token_cursor=0
    for si in sent_items:
        si["token_start_idx"]=token_cursor
        si["token_end_idx"]=token_cursor+si["token_len"]
        token_cursor=si["token_end_idx"]
    windows=[]
    i=0
    window_index=0
    while i<len(sent_items):
        cur_token_count=0
        chunk_sent_texts=[]
        chunk_token_start=sent_items[i]["token_start_idx"]
        chunk_token_end=chunk_token_start
        is_truncated_sentence=False
        start_i=i
        while i<len(sent_items):
            sent=sent_items[i]
            sent_tok_len=sent["token_len"]
            if cur_token_count+sent_tok_len>max_tokens:
                if not chunk_sent_texts:
                    if sent_tok_len>0:
                        if _ENCODER_BACKEND=="tiktoken":
                            prefix_tok_ids=sent["tokens"][:max_tokens]
                            prefix_text=_ENCODER_DECODE(prefix_tok_ids)
                            chunk_sent_texts.append(prefix_text)
                            cur_token_count=len(prefix_tok_ids)
                            is_truncated_sentence=True
                            remainder_tok_ids=sent["tokens"][max_tokens:]
                            if remainder_tok_ids:
                                remainder_text=_ENCODER_DECODE(remainder_tok_ids)
                                sent_items[i]={"text":remainder_text,"start_char":None,"end_char":None,"token_len":len(remainder_tok_ids),"tokens":remainder_tok_ids,"token_start_idx":None,"token_end_idx":None}
                            else:
                                i+=1
                            chunk_token_end=chunk_token_start+cur_token_count
                            break
                        else:
                            tokens=sent["tokens"]
                            prefix=tokens[:max_tokens]
                            prefix_text=" ".join(prefix)
                            chunk_sent_texts.append(prefix_text)
                            cur_token_count=len(prefix)
                            is_truncated_sentence=True
                            remainder=tokens[max_tokens:]
                            if remainder:
                                remainder_text=" ".join(remainder)
                                sent_items[i]={"text":remainder_text,"start_char":None,"end_char":None,"token_len":len(remainder),"tokens":remainder,"token_start_idx":None,"token_end_idx":None}
                            else:
                                i+=1
                            chunk_token_end=chunk_token_start+cur_token_count
                            break
                    else:
                        i+=1
                        break
                else:
                    break
            else:
                chunk_sent_texts.append(sent["text"])
                cur_token_count+=sent_tok_len
                chunk_token_end=sent.get("token_end_idx",chunk_token_start+cur_token_count)
                i+=1
        if not chunk_sent_texts:
            i+=1
            continue
        chunk_text=" ".join(chunk_sent_texts).strip()
        chunk_meta={"window_index":window_index,"text":chunk_text,"token_count":cur_token_count,"token_start":chunk_token_start,"token_end":chunk_token_end,"start_sentence_idx":start_i,"end_sentence_idx":i,"is_truncated_sentence":is_truncated_sentence}
        window_index+=1
        new_start=max(start_i+1,chunk_meta["end_sentence_idx"]-overlap_sentences)
        if windows and chunk_meta["token_count"]<min_tokens:
            prev=windows[-1]
            prev["text"]=prev["text"]+" "+chunk_meta["text"]
            prev["token_count"]=prev["token_count"]+chunk_meta["token_count"]
            prev["token_end"]=chunk_meta["token_end"]
            prev["end_sentence_idx"]=chunk_meta["end_sentence_idx"]
            prev["is_truncated_sentence"]=prev.get("is_truncated_sentence",False) or chunk_meta.get("is_truncated_sentence",False)
        else:
            windows.append(chunk_meta)
        i=new_start
    for w in windows:
        yield w
def s3_upload_file_atomic(local_path:str,bucket:str,key:str,content_type:str="application/octet-stream")->None:
    tmp_key=f"{key}.tmp.{os.getpid()}.{int(time.time())}"
    retries=int(os.getenv("S3_PUT_RETRIES","3"))
    backoff=float(os.getenv("S3_PUT_BACKOFF","0.3"))
    for attempt in range(1,retries+1):
        try:
            s3.upload_file(local_path,bucket,tmp_key,ExtraArgs={"ContentType":content_type})
            copy_source={"Bucket":bucket,"Key":tmp_key}
            s3.copy_object(CopySource=copy_source,Bucket=bucket,Key=key)
            s3.delete_object(Bucket=bucket,Key=tmp_key)
            return
        except Exception as e:
            log.warning("s3 upload attempt %d failed for %s: %s",attempt,key,e)
            time.sleep(backoff*attempt)
    raise Exception(f"s3 atomic upload failed for {key}")
class S3ParquetWriter:
    def __init__(self,doc_id:str):
        self.doc_id=doc_id
        self._rows:List[Dict[str,Any]]=[]
    def _normalize(self,payload:Dict[str,Any])->Dict[str,Any]:
        fields={}
        fields["document_id"]=payload.get("document_id") or ""
        fields["file_name"]=payload.get("file_name") or ""
        fields["chunk_id"]=payload.get("chunk_id") or ""
        fields["chunk_type"]=payload.get("chunk_type") or ""
        fields["text"]=payload.get("text") or ""
        try:
            fields["token_count"]=int(payload.get("token_count") or 0)
        except Exception:
            fields["token_count"]=0
        for k in ("figures","tags","layout_tags","heading_path","headings"):
            v=payload.get(k,None)
            try:
                fields[k]=json.dumps(v,ensure_ascii=False,sort_keys=True) if v is not None else "[]"
            except Exception:
                fields[k]="[]"
        fields["file_type"]=payload.get("file_type") or ""
        fields["source_url"]=payload.get("source_url") or ""
        try:
            ts=payload.get("timestamp") or ""
            fields["timestamp"]=ts
        except Exception:
            fields["timestamp"]=""
        fields["parser_version"]=payload.get("parser_version") or PARSER_VERSION
        fields["used_ocr"]=bool(payload.get("used_ocr",False))
        try:
            tr=payload.get("token_range")
            if isinstance(tr,(list,tuple)) and len(tr)>=2:
                fields["token_start"]=int(tr[0])
                fields["token_end"]=int(tr[1])
            else:
                fields["token_start"]=0
                fields["token_end"]=0
        except Exception:
            fields["token_start"]=0; fields["token_end"]=0
        return fields
    def write_payload(self,payload:Dict[str,Any])->int:
        self._rows.append(self._normalize(payload))
        return 1
    def finalize_and_upload(self,out_basename:str)->Tuple[int,str,str,int]:
        if not self._rows:
            return 0,"","",0
        schema=pa.schema([
            pa.field("document_id",pa.string()),pa.field("file_name",pa.string()),pa.field("chunk_id",pa.string()),pa.field("chunk_type",pa.string()),pa.field("text",pa.string()),pa.field("token_count",pa.int64()),pa.field("figures",pa.string()),pa.field("tags",pa.string()),pa.field("layout_tags",pa.string()),pa.field("heading_path",pa.string()),pa.field("headings",pa.string()),pa.field("file_type",pa.string()),pa.field("source_url",pa.string()),pa.field("token_start",pa.int64()),pa.field("token_end",pa.int64()),pa.field("timestamp",pa.string()),pa.field("parser_version",pa.string()),pa.field("used_ocr",pa.bool_())
        ])
        cols={name:[] for name in [f.name for f in schema]}
        for r in self._rows:
            for name in cols:
                cols[name].append(r.get(name) if name in r else None)
        table=pa.Table.from_pydict(cols,schema=schema)
        existing_md=table.schema.metadata or {}
        new_md=dict(existing_md)
        new_md.update({b"schema_version":CHUNKED_SCHEMA_VERSION.encode("utf-8") if (CHUNKED_SCHEMA_VERSION:=os.getenv("CHUNKED_SCHEMA_VERSION","chunked_v1")) else b"chunked_v1",b"parser_version":PARSER_VERSION.encode("utf-8"),b"producer":b"html_trafilatura",b"created_at":datetime.utcnow().isoformat().encode("utf-8")})
        table=table.replace_schema_metadata(new_md)
        tmpfile=tempfile.NamedTemporaryFile(mode="wb",delete=False,suffix=".parquet",dir="/tmp")
        tmpfile.close()
        pq.write_table(table,tmpfile.name,compression="zstd",flavor="spark")
        with open(tmpfile.name,"rb") as fh:
            b=fh.read()
        sha=sha256_hex_bytes(b)
        size=os.path.getsize(tmpfile.name)
        parquet_key=out_basename+".parquet"
        s3_upload_file_atomic(tmpfile.name,S3_BUCKET,S3_CHUNKED_PREFIX+parquet_key,content_type="application/octet-stream")
        try:
            os.unlink(tmpfile.name)
        except Exception:
            pass
        return len(self._rows),S3_CHUNKED_PREFIX+parquet_key,sha,size
def sanitize_payload_for_raw_manifest(doc_id:str,s3_raw_key:str,chunked_s3_key:str,rows:int,sha:str,size:int)->Dict[str,Any]:
    return {"raw_key":s3_raw_key,"doc_id":doc_id,"chunked_key":chunked_s3_key,"rows":rows,"sha256":sha,"size_bytes":size,"schema_version":os.getenv("CHUNKED_SCHEMA_VERSION","chunked_v1"),"parser_version":PARSER_VERSION,"created_at":datetime.utcnow().isoformat()+"Z"}
def _derive_file_name_from_source(source:Optional[str],s3_key:str)->str:
    if source:
        try:
            base=source.split("?")[0].rstrip("/")
            base_name=os.path.basename(base)
            if base_name:
                return base_name
        except Exception:
            pass
    return os.path.basename(s3_key)
def sanitize_payload_for_weaviate(payload:Dict[str,Any])->None:
    for k in list(payload.keys()):
        v=payload.get(k)
        if k=="tags":
            if v is None:
                payload[k]=[]
            elif isinstance(v,(list,tuple)):
                payload[k]=[str(x) for x in v]
            else:
                payload[k]=[str(v)]
            continue
        if v is None:
            payload.pop(k,None)
            continue
        if isinstance(v,(list,tuple,dict)):
            try:
                payload[k]=json.dumps(v)
            except Exception:
                payload[k]=str(v)
            continue
        if not isinstance(v,(str,int,float,bool)):
            payload[k]=str(v)
def parse_file(s3_key:str,manifest:Dict[str,Any])->Dict[str,Any]:
    start_all=time.perf_counter()
    try:
        head=s3.head_object(Bucket=S3_BUCKET,Key=s3_key)
    except Exception as e:
        log.error("Could not head S3 object %s: %s",s3_key,e)
        return {"saved_chunks":0,"total_parse_duration_ms":0}
    last_modified=head.get("LastModified","")
    doc_id=manifest.get("file_hash") or sha256_hex_str(s3_key+str(last_modified or ""))
    out_basename=f"{doc_id}"
    raw_manifest_key=s3_key+".manifest.json"
    if not FORCE_OVERWRITE and s3_object_exists(raw_manifest_key):
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.info("Skipping because raw manifest exists: %s",raw_manifest_key)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True}
    if not FORCE_OVERWRITE and s3_object_exists(S3_CHUNKED_PREFIX+out_basename+".parquet"):
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.info("Skipping because parquet exists: %s",out_basename+".parquet")
        try:
            if not s3_object_exists(raw_manifest_key):
                head2=s3.head_object(Bucket=S3_BUCKET,Key=S3_CHUNKED_PREFIX+out_basename+".parquet")
                etag=head2.get("ETag","")
                if isinstance(etag,str):
                    etag=etag.strip('"')
                size=head2.get("ContentLength",0)
                raw_manifest=sanitize_payload_for_raw_manifest(doc_id,s3_key,S3_CHUNKED_PREFIX+out_basename+".parquet",0,etag,size)
                s3.put_object(Bucket=S3_BUCKET,Key=raw_manifest_key,Body=json.dumps(raw_manifest).encode("utf-8"),ContentType="application/json")
        except Exception:
            pass
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True}
    try:
        obj=s3.get_object(Bucket=S3_BUCKET,Key=s3_key)
    except Exception as e:
        log.error("Could not get S3 object %s: %s",s3_key,e)
        return {"saved_chunks":0,"total_parse_duration_ms":0}
    raw_body=obj["Body"].read()
    try:
        raw_text=raw_body.decode("utf-8",errors="replace")
    except Exception:
        raw_text=raw_body.decode("latin-1",errors="replace")
    s3_path=f"s3://{S3_BUCKET}/{s3_key}"
    stripped=raw_text.strip()
    use_remote_fetch=False
    remote_url=None
    if stripped.startswith("http://") or stripped.startswith("https://"):
        use_remote_fetch=True
        remote_url=stripped.splitlines()[0].strip()
    if use_remote_fetch:
        try:
            html_text=fetch_html_with_retries(remote_url)
            source_url=remote_url
        except Exception:
            html_text=raw_text
            source_url=s3_path
    else:
        html_text=raw_text
        source_url=s3_path
    try:
        _=upload_snapshot_to_s3(html_text,doc_id)
    except Exception:
        pass
    t0=time.perf_counter()
    md,parsed=trafilatura_extract_markdown(html_text)
    extract_duration_ms=int((time.perf_counter()-t0)*1000)
    if not md:
        fallback=re.sub(r'(?is)<(script|style).*?>.*?</\1>','',html_text)
        fallback=re.sub(r'(?is)<.*?>',' ',fallback)
        md=re.sub(r'\s+',' ',fallback).strip()
    title=parsed.get("title") if isinstance(parsed,dict) else None
    canonical_full=canonicalize_text(md or "")
    token_ct=len(_ENCODER_ENCODE(canonical_full)) if canonical_full else 0
    saved=0
    writer=S3ParquetWriter(doc_id=doc_id)
    file_name=_derive_file_name_from_source(source_url,s3_key)
    try:
        windows=list(split_into_token_windows(canonical_full))
        if len(windows)<=1:
            chunk_index=1
            chunk_id=f"{doc_id}_{chunk_index}"
            payload={"document_id":doc_id,"file_name":file_name,"chunk_id":chunk_id,"chunk_type":"page","text":canonical_full,"token_count":int(token_ct),"figures":"[]","embedding":None,"file_type":"text/html","source_url":source_url,"token_range":[0,token_ct],"timestamp":datetime.utcnow().isoformat()+"Z","parser_version":PARSER_VERSION,"tags":manifest.get("tags",[]) if isinstance(manifest,dict) else [],"layout_tags":["page"],"used_ocr":False,"heading_path":[],"headings":[title] if title else [],"line_range":None}
            sanitize_payload_for_weaviate(payload)
            writer.write_payload(payload)
            saved+=1
        else:
            for w in windows:
                window_idx=int(w.get("window_index",0))
                chunk_index=window_idx+1
                chunk_id=f"{doc_id}_{chunk_index}"
                wtext=w.get("text","")
                w_token_count=int(w.get("token_count",0))
                token_range=[int(w.get("token_start",0)),int(w.get("token_end",0))]
                payload={"document_id":doc_id,"file_name":file_name,"chunk_id":chunk_id,"chunk_type":"token_window","text":wtext,"token_count":int(w_token_count),"figures":"[]","embedding":None,"file_type":"text/html","source_url":source_url,"token_range":token_range,"timestamp":datetime.utcnow().isoformat()+"Z","parser_version":PARSER_VERSION,"tags":manifest.get("tags",[]) if isinstance(manifest,dict) else [],"layout_tags":["page"],"used_ocr":False,"heading_path":[],"headings":[title] if title else [],"line_range":None}
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                saved+=1
    except Exception as e:
        try:
            if writer and getattr(writer,"_rows",None):
                pass
        except Exception:
            pass
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.exception("Error while buffering chunks for %s: %s",s3_key,str(e))
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e)}
    try:
        if saved==0:
            total_ms=int((time.perf_counter()-start_all)*1000)
            log.info("No chunks produced for %s",s3_key)
            return {"saved_chunks":0,"total_parse_duration_ms":total_ms}
        count,uploaded_s3_key,sha,size=writer.finalize_and_upload(out_basename)
        total_ms=int((time.perf_counter()-start_all)*1000)
        try:
            raw_manifest=sanitize_payload_for_raw_manifest(doc_id,s3_key,uploaded_s3_key,count,sha,size)
            s3.put_object(Bucket=S3_BUCKET,Key=raw_manifest_key,Body=json.dumps(raw_manifest).encode("utf-8"),ContentType="application/json")
        except Exception:
            log.warning("Failed to write raw manifest for %s",s3_key)
        log.info("Wrote %d chunks for %s → %s (%d ms)",count,s3_key,uploaded_s3_key,total_ms)
        return {"saved_chunks":count,"total_parse_duration_ms":total_ms,"skipped":False}
    except Exception as e_up:
        total_ms=int((time.perf_counter()-start_all)*1000)
        try:
            pass
        except Exception:
            pass
        log.error("Failed to upload chunked file for %s error=%s",s3_key,str(e_up))
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e_up)}
if __name__=="__main__":
    paginator=s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET,Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents",[]):
            key=obj["Key"]
            if not (key.lower().endswith(".html") or key.lower().endswith(".htm")):
                continue
            log.info("Routing parse_file for s3://%s/%s",S3_BUCKET,key)
            manifest_key=key+".manifest.json"
            try:
                mf_obj=s3.get_object(Bucket=S3_BUCKET,Key=manifest_key)
                manifest=json.load(mf_obj["Body"])
            except Exception:
                manifest={}
            try:
                res=parse_file(key,manifest)
                log.info("Result for %s: %s",key,res)
            except Exception as e:
                log.exception("Failed to parse %s: %s",key,e)
