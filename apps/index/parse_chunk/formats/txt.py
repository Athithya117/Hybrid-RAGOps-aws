import os,sys,json,time,logging,hashlib,boto3,unicodedata,re,tempfile
from datetime import datetime
from botocore.exceptions import ClientError
from typing import List,Dict,Any,Tuple,Optional,Generator
try:
    import colorama
    colorama.init()
except Exception:
    pass
RESET="\033[0m"
COLORS={logging.DEBUG:"\033[90m",logging.INFO:"\033[37m",logging.WARNING:"\033[33m",logging.ERROR:"\033[31m",logging.CRITICAL:"\033[1;41m"}
class ColorFormatter(logging.Formatter):
    def format(self,record):
        color=COLORS.get(record.levelno,RESET)
        message=super().format(record)
        return f"{color}{message}{RESET}"
log=logging.getLogger("txt_parser")
log.setLevel(logging.INFO)
handler=logging.StreamHandler(sys.stdout)
handler.setFormatter(ColorFormatter("%(asctime)s %(levelname)s %(message)s"))
log.handlers[:]=[handler]
REQUIRED=["S3_BUCKET","S3_RAW_PREFIX","S3_CHUNKED_PREFIX","CHUNK_FORMAT"]
missing=[v for v in REQUIRED if os.getenv(v) is None]
if missing:
    sys.exit(f"ERROR: Missing env vars: {', '.join(missing)}")
S3_BUCKET=os.getenv("S3_BUCKET")
S3_RAW_PREFIX=os.getenv("S3_RAW_PREFIX").rstrip("/")+"/"
S3_CHUNKED_PREFIX=os.getenv("S3_CHUNKED_PREFIX").rstrip("/")+"/"
CHUNK_FORMAT=os.getenv("CHUNK_FORMAT","json").lower()
assert CHUNK_FORMAT in ("json","jsonl")
ENC_NAME=os.getenv("TOKEN_ENCODER","cl100k_base")
PARSER_VERSION=os.getenv("PARSER_VERSION_TXT","plain-txt-v1")
FORCE_OVERWRITE=os.getenv("FORCE_OVERWRITE","false").lower()=="true"
S3_PUT_RETRIES=int(os.getenv("S3_PUT_RETRIES","3"))
S3_PUT_BACKOFF=float(os.getenv("S3_PUT_BACKOFF","0.3"))
s3=boto3.client("s3")
try:
    import tiktoken
except Exception:
    tiktoken=None
def sha256_hex(s:str)->str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def canonicalize_text(s:str)->str:
    if not isinstance(s,str):
        s=str(s or "")
    s=unicodedata.normalize("NFKC",s)
    s=s.replace("\r\n","\n").replace("\r","\n")
    lines=[re.sub(r'[ \t]+$','',ln) for ln in s.split("\n")]
    return "\n".join(lines).strip()
def try_decode_bytes(b:bytes)->str:
    for encoding in ("utf-8","utf-8-sig","latin-1"):
        try:
            return b.decode(encoding)
        except Exception:
            continue
    return b.decode("utf-8",errors="replace")
def token_count_for(text:str)->int:
    if not text:
        return 0
    try:
        enc=None
        if tiktoken is not None:
            try:
                enc=tiktoken.get_encoding(ENC_NAME)
            except Exception:
                try:
                    enc=tiktoken.encoding_for_model("gpt2")
                except Exception:
                    enc=None
        if enc is not None:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
    except Exception:
        pass
    return len(text.split())
def s3_object_exists(key:str)->bool:
    try:
        s3.head_object(Bucket=S3_BUCKET,Key=key)
        return True
    except ClientError:
        return False
    except Exception:
        return False
def s3_put_object_with_retries(key:str,body:bytes,content_type:str="application/json")->None:
    for attempt in range(1,S3_PUT_RETRIES+1):
        try:
            s3.put_object(Bucket=S3_BUCKET,Key=key,Body=body,ContentType=content_type)
            return
        except Exception:
            if attempt < S3_PUT_RETRIES:
                time.sleep(S3_PUT_BACKOFF*attempt)
    raise Exception("s3 put failed after retries")
def _derive_source_key_from_path(s3_path:str)->str:
    prefix=f"s3://{S3_BUCKET}/"
    if s3_path.startswith(prefix):
        return s3_path[len(prefix):]
    return ""
class S3DocWriter:
    def __init__(self,doc_id:str,s3_path:str,ext:str,content_type:str="application/json"):
        self.doc_id=doc_id
        self.s3_path=s3_path or ""
        self.ext=ext
        self.content_type=content_type
        self.temp=tempfile.NamedTemporaryFile(mode="wb",delete=False,suffix=f".{ext}",dir="/tmp")
        self.count=0
        self._first=True
        if self.ext=="json":
            self.temp.write(b"[\n")
            self.temp.flush()
    def write_payload(self,payload:Dict[str,Any])->int:
        self.count+=1
        if self.ext=="jsonl":
            line=(json.dumps(payload,ensure_ascii=False)+"\n").encode("utf-8")
            self.temp.write(line)
        else:
            pretty=json.dumps(payload,ensure_ascii=False,indent=2)
            indented=("\n".join("  "+ln for ln in pretty.splitlines())+"\n").encode("utf-8")
            if not self._first:
                self.temp.write(b",\n")
            self.temp.write(indented)
            self._first=False
        self.temp.flush()
        return 1
    def finalize_and_upload(self,out_key:str)->Tuple[int,str]:
        if self.ext=="json":
            self.temp.write(b"]\n")
        self.temp.flush()
        self.temp.close()
        try:
            s3.upload_file(self.temp.name,S3_BUCKET,out_key,ExtraArgs={"ContentType":self.content_type})
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            return self.count,out_key
        except Exception:
            try:
                os.unlink(self.temp.name)
            except Exception:
                pass
            raise
def _derive_doc_id_from_head(s3_key:str,head_obj:dict,manifest:dict)->str:
    if isinstance(manifest,dict) and manifest.get("file_hash"):
        return manifest.get("file_hash")
    etag=head_obj.get("ETag","")
    if isinstance(etag,str):
        etag=etag.strip('"')
    if etag:
        return sha256_hex(s3_key+str(etag))
    lm=head_obj.get("LastModified","")
    if lm:
        return sha256_hex(s3_key+str(lm))
    base=os.path.basename(s3_key)
    if base:
        return base
    return sha256_hex(s3_key)
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
try:
    import spacy
    from spacy.pipeline import Sentencizer
except Exception:
    spacy=None
    Sentencizer=None
class TokenEncoder:
    def __init__(self,model_name:str="gpt2",enc_name:Optional[str]=None):
        self.model_name=model_name
        self.enc_name=enc_name or os.getenv("TOKEN_ENCODER",ENC_NAME)
        self.backend="whitespace"
        self._enc=None
        try:
            if tiktoken is None:
                raise RuntimeError("tiktoken not installed")
            try:
                self._enc=tiktoken.get_encoding(self.enc_name)
            except Exception:
                try:
                    self._enc=tiktoken.encoding_for_model(self.model_name)
                except Exception:
                    self._enc=None
            if self._enc is not None:
                self.encode=lambda txt:self._enc.encode(txt)
                self.decode=lambda toks:self._enc.decode(toks)
                self.backend="tiktoken"
                log.info("Using tiktoken encoder: enc_name=%s model=%s",self.enc_name,self.model_name)
            else:
                raise RuntimeError("tiktoken imported but no encoder available")
        except Exception as e:
            log.warning("tiktoken unavailable or failed to load encoder (%s); falling back to whitespace tokeniser",str(e))
            self.encode=lambda txt:txt.split()
            self.decode=lambda toks:" ".join(toks)
class SentenceChunker:
    def __init__(self,max_tokens_per_chunk:Optional[int]=None,overlap_sentences:Optional[int]=None,token_model:str="gpt2",min_tokens_per_chunk:Optional[int]=None):
        self.max_tokens_per_chunk=int(os.getenv("MAX_TOKENS_PER_CHUNK","512")) if max_tokens_per_chunk is None else int(max_tokens_per_chunk)
        self.overlap_sentences=int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES","2")) if overlap_sentences is None else int(overlap_sentences)
        if self.overlap_sentences<0:
            raise ValueError("overlap_sentences must be >= 0")
        self.min_tokens_per_chunk=int(os.getenv("MIN_TOKENS_PER_CHUNK","100")) if min_tokens_per_chunk is None else int(min_tokens_per_chunk)
        if self.min_tokens_per_chunk<0:
            raise ValueError("min_tokens_per_chunk must be >= 0")
        self.encoder=TokenEncoder(model_name=token_model,enc_name=os.getenv("TOKEN_ENCODER",ENC_NAME))
        self.nlp=self._make_sentencizer()
        log.info("SentenceChunker initialized: max_tokens=%d,min_tokens=%d,overlap_sentences=%d,token_backend=%s",self.max_tokens_per_chunk,self.min_tokens_per_chunk,self.overlap_sentences,getattr(self.encoder,"backend","unknown"))
    def _make_sentencizer(self):
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
            log.warning("spaCy sentencizer unavailable")
            return None
    def _sentences_with_offsets_spacy(self,text:str):
        doc=self.nlp(text)
        return [(sent.text.strip(),int(sent.start_char),int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
    def _sentences_with_offsets_regex(self,text:str):
        pattern=re.compile(r'(?s).*?[\.\!\?]["\']?\s+|.+$')
        items=[]
        pos=0
        for m in pattern.finditer(text):
            s=m.group(0)
            if not s or s.strip()=="":
                pos=m.end()
                continue
            start=pos
            end=pos+len(s)
            items.append((s.strip(),start,end))
            pos=m.end()
        if not items:
            if text.strip():
                items=[(text.strip(),0,len(text))]
        return items
    def _sentences_with_offsets(self,text:str):
        if self.nlp is not None:
            try:
                return self._sentences_with_offsets_spacy(text)
            except Exception:
                pass
        return self._sentences_with_offsets_regex(text)
    def chunk_document(self,text:str)->Generator[Dict,None,None]:
        sentences=self._sentences_with_offsets(text)
        sent_items=[{"text":s,"start_char":sc,"end_char":ec,"orig_idx":i,"is_remainder":False} for i,(s,sc,ec) in enumerate(sentences)]
        i=0
        n=len(sent_items)
        prev_chunk=None
        while i<n:
            cur_token_count=0
            chunk_sent_texts=[]
            chunk_start_idx=i
            chunk_start_char=sent_items[i]["start_char"] if i<n else None
            chunk_end_char=None
            is_truncated_sentence=False
            while i<n:
                sent_text=sent_items[i]["text"]
                tok_ids=self.encoder.encode(sent_text)
                sent_tok_len=len(tok_ids)
                if cur_token_count+sent_tok_len>self.max_tokens_per_chunk:
                    if not chunk_sent_texts:
                        prefix_tok_ids=tok_ids[:self.max_tokens_per_chunk]
                        try:
                            prefix_text=self.encoder.decode(prefix_tok_ids)
                        except Exception:
                            prefix_text=" ".join(str(x) for x in prefix_tok_ids)
                        chunk_sent_texts.append(prefix_text)
                        cur_token_count=len(prefix_tok_ids)
                        is_truncated_sentence=True
                        remainder_tok_ids=tok_ids[self.max_tokens_per_chunk:]
                        if remainder_tok_ids:
                            try:
                                remainder_text=self.encoder.decode(remainder_tok_ids)
                            except Exception:
                                remainder_text=" ".join(str(x) for x in remainder_tok_ids)
                            sent_items[i]={"text":remainder_text,"start_char":None,"end_char":None,"orig_idx":sent_items[i]["orig_idx"],"is_remainder":True}
                        else:
                            i+=1
                        break
                    else:
                        break
                else:
                    chunk_sent_texts.append(sent_text)
                    cur_token_count+=sent_tok_len
                    chunk_end_char=sent_items[i]["end_char"]
                    i+=1
            if not chunk_sent_texts:
                log.warning("Empty chunk at idx %d; advancing one sentence to avoid infinite loop",i)
                i+=1
                continue
            chunk_text=" ".join(chunk_sent_texts).strip()
            chunk_meta={"text":chunk_text,"token_count":cur_token_count,"start_sentence_idx":chunk_start_idx,"end_sentence_idx":i,"start_char":chunk_start_char,"end_char":chunk_end_char,"is_truncated_sentence":is_truncated_sentence}
            new_start=max(chunk_start_idx+1,chunk_meta["end_sentence_idx"]-self.overlap_sentences)
            if prev_chunk is None:
                prev_chunk=chunk_meta
            else:
                if chunk_meta["token_count"]<self.min_tokens_per_chunk:
                    prev_chunk["text"]=prev_chunk["text"]+" "+chunk_meta["text"]
                    prev_chunk["token_count"]=prev_chunk["token_count"]+chunk_meta["token_count"]
                    prev_chunk["end_sentence_idx"]=chunk_meta["end_sentence_idx"]
                    prev_chunk["end_char"]=chunk_meta["end_char"]
                    prev_chunk["is_truncated_sentence"]=prev_chunk["is_truncated_sentence"] or chunk_meta["is_truncated_sentence"]
                else:
                    yield prev_chunk
                    prev_chunk=chunk_meta
            i=new_start
            n=len(sent_items)
        if prev_chunk is not None:
            yield prev_chunk
def _charpos_to_line_range(start_char:Optional[int],end_char:Optional[int],lines:List[str])->Tuple[int,int]:
    if start_char is None or end_char is None or not lines:
        return 1,max(1,len(lines))
    cum=[0]
    for ln in lines:
        cum.append(cum[-1]+len(ln))
    start_line=1
    end_line=len(lines)
    for i in range(len(cum)-1):
        if cum[i]<=start_char<cum[i+1]:
            start_line=i+1
            break
    for j in range(len(cum)-1,-1,-1):
        if j==0:
            end_line=1
            break
        if cum[j-1]<=max(0,end_char-1)<cum[j]:
            end_line=j
            break
    return start_line,end_line
def parse_file(s3_key:str,manifest:dict)->dict:
    start_all=time.perf_counter()
    try:
        head_obj=s3.head_object(Bucket=S3_BUCKET,Key=s3_key)
    except Exception as e:
        log.warning("HEAD failed for %s: %s -- will try GET",s3_key,e)
        head_obj={}
    doc_id=_derive_doc_id_from_head(s3_key,head_obj or {},manifest or {})
    ext="jsonl" if CHUNK_FORMAT=="jsonl" else "json"
    out_key=f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.info("Skipping entire file because chunked file exists: %s",out_key)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True}
    try:
        obj=s3.get_object(Bucket=S3_BUCKET,Key=s3_key)
    except Exception as e:
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.error("Could not read S3 object %s: %s",s3_key,e)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e)}
    raw_body=obj["Body"].read()
    raw_text=try_decode_bytes(raw_body)
    if isinstance(manifest,dict) and manifest.get("file_hash"):
        doc_id=manifest.get("file_hash")
        out_key=f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.info("Skipping entire file because chunked file exists (post-download): %s",out_key)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True}
    s3_path=f"s3://{S3_BUCKET}/{s3_key}"
    canonical_full=canonicalize_text(raw_text)
    lines=[ln+("\n" if not ln.endswith("\n") else "") for ln in canonical_full.split("\n")]
    max_tokens=int(os.getenv("MAX_TOKENS_PER_CHUNK","512"))
    min_tokens=int(os.getenv("MIN_TOKENS_PER_CHUNK","100"))
    overlap_sentences=int(os.getenv("NUMBER_OF_OVERLAPPING_SENTENCES","2"))
    full_token_count=token_count_for(canonical_full)
    saved=0
    chunk_index=1
    writer=S3DocWriter(doc_id=doc_id,s3_path=s3_path,ext=ext)
    file_name=os.path.basename(s3_key)
    try:
        if full_token_count<=max_tokens:
            chunk_id=f"{doc_id}_{chunk_index}"
            chunk_index+=1
            payload={"document_id":doc_id or "","file_name":file_name,"chunk_id":chunk_id or "","chunk_type":"txt_subchunk","text":canonical_full or "","token_count":int(full_token_count or 0),"figures":"[]","embedding":None,"figures":[],"file_type":"text/plain","source_url":s3_path,"page_number":None,"slide_range":None,"row_range":None,"token_range":None,"audio_range":None,"timestamp":datetime.utcnow().isoformat()+"Z","parser_version":PARSER_VERSION,"tags":manifest.get("tags",[]) if isinstance(manifest,dict) else [],"layout_tags":[],"used_ocr":False,"heading_path":[],"headings":[],"line_range":[1,len(lines)]}
            sanitize_payload_for_weaviate(payload)
            writer.write_payload(payload)
            log.info("Buffered single chunk %s",payload["chunk_id"])
            saved+=1
        else:
            chunker=SentenceChunker(max_tokens_per_chunk=max_tokens,overlap_sentences=overlap_sentences,token_model="gpt2",min_tokens_per_chunk=min_tokens)
            subchunks=[]
            for ch in chunker.chunk_document(canonical_full):
                subchunks.append(ch)
            sub_idx=1
            for sub in subchunks:
                chunk_text=sub.get("text","")
                token_ct=sub.get("token_count",0)
                schar=sub.get("start_char")
                echar=sub.get("end_char")
                start_line,end_line=_charpos_to_line_range(schar,echar,lines)
                chunk_id=f"{doc_id}_{chunk_index}"
                chunk_index+=1
                payload={"document_id":doc_id or "","file_name":file_name,"chunk_id":chunk_id or "","chunk_type":"txt_subchunk","text":chunk_text or "","token_count":int(token_ct or 0),"figures":"[]","embedding":None,"figures":[],"file_type":"text/plain","source_url":s3_path,"page_number":None,"slide_range":None,"row_range":None,"token_range":None,"audio_range":None,"timestamp":datetime.utcnow().isoformat()+"Z","parser_version":PARSER_VERSION,"tags":manifest.get("tags",[]) if isinstance(manifest,dict) else [],"layout_tags":[],"used_ocr":False,"heading_path":[],"headings":[],"line_range":[int(start_line),int(end_line)]}
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                log.info("Buffered subchunk %s (lines %d-%d)",payload["chunk_id"],start_line,end_line)
                saved+=1
                sub_idx+=1
    except Exception as e:
        try:
            if writer and getattr(writer,"temp",None):
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.exception("Error while buffering chunks for %s: %s",s3_key,str(e))
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e)}
    try:
        if saved==0:
            try:
                if writer and getattr(writer,"temp",None):
                    os.unlink(writer.temp.name)
            except Exception:
                pass
            total_ms=int((time.perf_counter()-start_all)*1000)
            log.info("No chunks produced for %s",s3_key)
            return {"saved_chunks":0,"total_parse_duration_ms":total_ms}
        count,uploaded_key=writer.finalize_and_upload(out_key)
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.info("Wrote %d chunks for %s → %s (%d ms)",count,s3_key,uploaded_key,total_ms)
        return {"saved_chunks":count,"total_parse_duration_ms":total_ms,"skipped":False}
    except Exception as e_up:
        total_ms=int((time.perf_counter()-start_all)*1000)
        log.error("Failed to upload chunked file for %s error=%s",s3_key,str(e_up))
        try:
            if writer and getattr(writer,"temp",None):
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e_up)}
if __name__=="__main__":
    log.info("TOKEN_ENCODER env: %s tiktoken_present: %s",os.getenv("TOKEN_ENCODER",ENC_NAME), "yes" if tiktoken is not None else "no")
    paginator=s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET,Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents",[]):
            key=obj["Key"]
            if not key.lower().endswith(".txt"):
                continue
            log.info("Routing parse_file for s3://%s/%s",S3_BUCKET,key)
            manifest_key=key+".manifest.json"
            try:
                mf=s3.get_object(Bucket=S3_BUCKET,Key=manifest_key)
                manifest=json.load(mf["Body"])
            except Exception:
                manifest={}
            try:
                parse_file(key,manifest)
            except Exception as e:
                log.exception("Failed to parse %s: %s",key,e)
