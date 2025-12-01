from __future__ import annotations
import os,io,sys,json,time,logging,hashlib,boto3,unicodedata,re,tempfile,importlib
from datetime import datetime
from botocore.exceptions import ClientError
from typing import List,Tuple,Dict,Generator,Optional,Any
from contextlib import contextmanager
from PIL import Image
import spacy
try:
    from spacy.pipeline import Sentencizer
except Exception:
    Sentencizer = None
def env_or_fail(name: str, default=None, mandatory: bool = False):
    val = os.getenv(name, default)
    if mandatory and val is None:
        print(f"ERROR: Required env var '{name}' not set", file=sys.stderr)
        sys.exit(1)
    return val
logger = logging.getLogger("images_parser")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.handlers[:] = [handler]
def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        logger.warning("Invalid env var for %s: %r; falling back to %d", name, v, default)
        return default
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = v.strip().lower()
    if s in ("1","true","yes","y","t"):
        return True
    if s in ("0","false","no","n","f"):
        return False
    return default
S3_BUCKET = env_or_fail("S3_BUCKET", None, mandatory=True)
S3_RAW_PREFIX = os.getenv("S3_RAW_PREFIX","data/raw/").rstrip("/") + "/"
S3_CHUNKED_PREFIX = os.getenv("S3_CHUNKED_PREFIX","data/chunked/").rstrip("/") + "/"
CHUNK_FORMAT = os.getenv("CHUNK_FORMAT","json").lower()
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE","false").lower() == "true"
IMAGE_DISABLE_OCR = _env_bool("IMAGE_DISABLE_OCR",False)
IMAGE_FORCE_OCR = _env_bool("IMAGE_FORCE_OCR",False)
IMAGE_OCR_ENGINE = os.getenv("IMAGE_OCR_ENGINE","auto").lower()
IMAGE_TESSERACT_LANG = os.getenv("IMAGE_TESSERACT_LANG","eng")
IMAGE_MIN_IMG_SIZE_BYTES = _env_int("IMAGE_MIN_IMG_SIZE_BYTES",512)
IMAGE_RENDER_DPI = _env_int("IMAGE_RENDER_DPI",300)
IMAGE_UPSCALE_FACTOR = float(os.getenv("IMAGE_UPSCALE_FACTOR","1.0"))
IMAGE_ENABLE_WORDSEGMENT = _env_bool("IMAGE_ENABLE_WORDSEGMENT",False)
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG","--oem 3 --psm 6")
MAX_TOKENS_PER_CHUNK = _env_int("MAX_TOKENS_PER_CHUNK",512)
MIN_TOKENS_PER_CHUNK = _env_int("MIN_TOKENS_PER_CHUNK",100)
NUMBER_OF_OVERLAPPING_SENTENCES = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES",2)
PARSER_VERSION_IMAGE = os.getenv("PARSER_VERSION_IMAGE","image-v1")
S3_PUT_RETRIES = _env_int("S3_PUT_RETRIES",3)
S3_PUT_BACKOFF = float(os.getenv("S3_PUT_BACKOFF","0.3"))
ENC_NAME = os.getenv("TOKEN_ENCODER","cl100k_base")
s3 = boto3.client("s3")
def _load_tiktoken(encoding_name: str):
    try:
        import tiktoken
        if hasattr(tiktoken, "encoding_for_model"):
            return tiktoken.encoding_for_model(encoding_name)
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return None
enc = _load_tiktoken(ENC_NAME)
def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
def local_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
def canonicalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s or "")
    s = unicodedata.normalize("NFKC",s)
    s = s.replace("\r\n","\n").replace("\r","\n")
    lines = [re.sub(r'[ \t]+$','',ln) for ln in s.split("\n")]
    return "\n".join(lines).strip()
def try_decode_bytes(b: bytes) -> str:
    for encoding in ("utf-8","utf-8-sig","latin-1"):
        try:
            return b.decode(encoding)
        except Exception:
            continue
    return b.decode("utf-8",errors="replace")
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
        s3.head_object(Bucket=S3_BUCKET,Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404","NoSuchKey","NotFound"):
            return False
        return False
    except Exception:
        return False
def s3_put_object_with_retries(key: str, body: bytes, content_type: str = "application/json") -> None:
    for attempt in range(1,S3_PUT_RETRIES+1):
        try:
            s3.put_object(Bucket=S3_BUCKET,Key=key,Body=body,ContentType=content_type)
            return
        except Exception:
            if attempt < S3_PUT_RETRIES:
                time.sleep(S3_PUT_BACKOFF * attempt)
    raise Exception("s3 put failed after retries")
def _derive_doc_id_from_head(s3_key: str, head_obj: dict, manifest: dict) -> str:
    if isinstance(manifest, dict) and manifest.get("file_hash"):
        return manifest.get("file_hash")
    etag = head_obj.get("ETag","") if isinstance(head_obj, dict) else ""
    if isinstance(etag, str):
        etag = etag.strip('"')
    if etag:
        return sha256_hex(s3_key + str(etag))
    lm = head_obj.get("LastModified","") if isinstance(head_obj, dict) else ""
    if lm:
        return sha256_hex(s3_key + str(lm))
    base = os.path.basename(s3_key)
    if base:
        return base
    return sha256_hex(s3_key)
def sanitize_payload_for_weaviate(payload: Dict[str,Any]) -> None:
    for k in list(payload.keys()):
        v = payload.get(k)
        if k == "tags":
            if v is None:
                payload[k] = []
            elif isinstance(v,(list,tuple)):
                payload[k] = [str(x) for x in v]
            else:
                payload[k] = [str(v)]
            continue
        if v is None:
            payload.pop(k,None)
            continue
        if isinstance(v,(list,tuple,dict)):
            try:
                payload[k] = json.dumps(v)
            except Exception:
                payload[k] = str(v)
            continue
        if not isinstance(v,(str,int,float,bool)):
            payload[k] = str(v)
class S3DocWriter:
    def __init__(self,doc_id: str,s3_path: str,ext: str,content_type: str="application/json"):
        self.doc_id = doc_id
        self.s3_path = s3_path or ""
        self.ext = ext
        self.content_type = content_type
        tmpdir = os.getenv("TMPDIR") or None
        self.temp = tempfile.NamedTemporaryFile(mode="wb",delete=False,suffix=f".{ext}",dir=tmpdir)
        self.count = 0
        self._first = True
        if self.ext == "json":
            self.temp.write(b"[\n")
            self.temp.flush()
    def write_payload(self,payload: Dict[str,Any]) -> int:
        self.count += 1
        if self.ext == "jsonl":
            line = (json.dumps(payload,ensure_ascii=False) + "\n").encode("utf-8")
            self.temp.write(line)
        else:
            pretty = json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True)
            indented = ("\n".join("  "+ln for ln in pretty.splitlines()) + "\n").encode("utf-8")
            if not self._first:
                self.temp.write(b",\n")
            self.temp.write(indented)
            self._first = False
        self.temp.flush()
        return 1
    def finalize_and_upload(self,out_key: str) -> Tuple[int,str]:
        if self.ext == "json":
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
@contextmanager
def without_cwd_on_syspath():
    saved = list(sys.path)
    try:
        cwd = os.getcwd()
        sys.path = [p for p in sys.path if p not in ("",cwd)]
        yield
    finally:
        sys.path[:] = saved
class TokenEncoder:
    def __init__(self,model_name: str="gpt2"):
        self.model_name = model_name
        self.backend = "whitespace"
        try:
            import tiktoken
            inc = getattr(tiktoken, "encoding_for_model", None)
            enc_local = inc(model_name) if inc is not None else tiktoken.get_encoding(model_name)
            self.encode = lambda txt: enc_local.encode(txt)
            self.decode = lambda toks: enc_local.decode(toks)
            self.backend = "tiktoken"
        except Exception:
            self.encode = lambda txt: txt.split()
            self.decode = lambda toks: " ".join(toks)
def split_long_sentence_by_words(sent_text: str,max_tokens: int,encoder: TokenEncoder) -> List[str]:
    words = sent_text.split()
    pieces: List[str] = []
    cur_words: List[str] = []
    cur_tok = 0
    for w in words:
        toks = encoder.encode(w)
        l = len(toks)
        if cur_tok + l > max_tokens:
            if cur_words:
                pieces.append(" ".join(cur_words))
                cur_words = []
                cur_tok = 0
            if l > max_tokens:
                tok_ids = encoder.encode(w)
                i = 0
                while i < len(tok_ids):
                    chunk_ids = tok_ids[i:i+max_tokens]
                    pieces.append(encoder.decode(chunk_ids))
                    i += max_tokens
                continue
        cur_words.append(w)
        cur_tok += l
    if cur_words:
        pieces.append(" ".join(cur_words))
    return pieces
class SentenceChunker:
    def __init__(self,max_tokens_per_chunk: Optional[int]=None,overlap_sentences: Optional[int]=None,token_model: str="gpt2",nlp=None,min_tokens_per_chunk: Optional[int]=None):
        self.max_tokens_per_chunk = _env_int("MAX_TOKENS_PER_CHUNK",MAX_TOKENS_PER_CHUNK) if max_tokens_per_chunk is None else int(max_tokens_per_chunk)
        self.overlap_sentences = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES",NUMBER_OF_OVERLAPPING_SENTENCES) if overlap_sentences is None else int(overlap_sentences)
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.min_tokens_per_chunk = _env_int("MIN_TOKENS_PER_CHUNK",MIN_TOKENS_PER_CHUNK) if min_tokens_per_chunk is None else int(min_tokens_per_chunk)
        if self.min_tokens_per_chunk < 0:
            raise ValueError("min_tokens_per_chunk must be >= 0")
        self.encoder = TokenEncoder(model_name=token_model)
        self.nlp = nlp or self._make_sentencizer()
    @staticmethod
    def _make_sentencizer():
        try:
            return spacy.load("en_core_web_sm")
        except Exception:
            nlp = spacy.blank("en")
            try:
                if Sentencizer is not None:
                    nlp.add_pipe("sentencizer")
                else:
                    nlp.add_pipe("sentencizer")
            except Exception:
                try:
                    nlp.add_pipe(Sentencizer())
                except Exception:
                    raise RuntimeError("Failed to add Sentencizer to spaCy pipeline.")
            return nlp
    def _sentences_with_offsets(self,text: str) -> List[Tuple[str,int,int]]:
        doc = self.nlp(text)
        sents = [(sent.text.strip(),int(sent.start_char),int(sent.end_char)) for sent in doc.sents if sent.text.strip()]
        return sents
    def chunk_document(self,text: str) -> Generator[Dict,None,None]:
        sentences = self._sentences_with_offsets(text)
        sent_items: List[Dict] = [{"text":s,"start_char":sc,"end_char":ec,"orig_idx":i,"is_remainder":False} for i,(s,sc,ec) in enumerate(sentences)]
        i = 0
        n = len(sent_items)
        prev_chunk = None
        while i < n:
            cur_token_count = 0
            chunk_sent_texts: List[str] = []
            chunk_start_idx = i
            chunk_start_char: Optional[int] = sent_items[i]["start_char"] if i < n else None
            chunk_end_char: Optional[int] = None
            is_truncated_sentence = False
            while i < n:
                sent_text = sent_items[i]["text"]
                tok_ids = self.encoder.encode(sent_text)
                sent_tok_len = len(tok_ids)
                if sent_tok_len > self.max_tokens_per_chunk:
                    pieces = split_long_sentence_by_words(sent_text,self.max_tokens_per_chunk,self.encoder)
                    if not pieces:
                        pieces = [sent_text[:1000]]
                    sent_items[i]["text"] = pieces[0]
                    for j,rem in enumerate(pieces[1:],start=1):
                        insert_idx = i + j
                        sent_items.insert(insert_idx,{"text":rem,"start_char":None,"end_char":None,"orig_idx":sent_items[i]["orig_idx"],"is_remainder":True})
                    n = len(sent_items)
                    tok_ids = self.encoder.encode(sent_items[i]["text"])
                    sent_tok_len = len(tok_ids)
                if cur_token_count + sent_tok_len > self.max_tokens_per_chunk:
                    if not chunk_sent_texts:
                        prefix_tok_ids = tok_ids[: self.max_tokens_per_chunk]
                        prefix_text = self.encoder.decode(prefix_tok_ids)
                        chunk_sent_texts.append(prefix_text)
                        cur_token_count = len(prefix_tok_ids)
                        is_truncated_sentence = True
                        remainder_tok_ids = tok_ids[self.max_tokens_per_chunk :]
                        if remainder_tok_ids:
                            remainder_text = self.encoder.decode(remainder_tok_ids)
                            sent_items[i] = {"text":remainder_text,"start_char":None,"end_char":None,"orig_idx":sent_items[i]["orig_idx"],"is_remainder":True}
                        else:
                            i += 1
                        break
                    else:
                        break
                else:
                    chunk_sent_texts.append(sent_text)
                    cur_token_count += sent_tok_len
                    chunk_end_char = sent_items[i]["end_char"]
                    i += 1
            if not chunk_sent_texts:
                i += 1
                continue
            chunk_text = " ".join(chunk_sent_texts).strip()
            chunk_meta = {"text":chunk_text,"token_count":cur_token_count,"start_sentence_idx":chunk_start_idx,"end_sentence_idx":i,"start_char":chunk_start_char,"end_char":chunk_end_char,"is_truncated_sentence":is_truncated_sentence}
            new_start = max(chunk_start_idx + 1, chunk_meta["end_sentence_idx"] - self.overlap_sentences)
            if prev_chunk is None:
                prev_chunk = chunk_meta
            else:
                if chunk_meta["token_count"] < self.min_tokens_per_chunk:
                    prev_chunk["text"] = prev_chunk["text"] + " " + chunk_meta["text"]
                    prev_chunk["token_count"] = prev_chunk["token_count"] + chunk_meta["token_count"]
                    prev_chunk["end_sentence_idx"] = chunk_meta["end_sentence_idx"]
                    prev_chunk["end_char"] = chunk_meta["end_char"]
                    prev_chunk["is_truncated_sentence"] = prev_chunk["is_truncated_sentence"] or chunk_meta["is_truncated_sentence"]
                else:
                    yield prev_chunk
                    prev_chunk = chunk_meta
            i = new_start
            n = len(sent_items)
        if prev_chunk is not None:
            yield prev_chunk
    @classmethod
    def from_env(cls,**kwargs):
        max_tokens = _env_int("MAX_TOKENS_PER_CHUNK",MAX_TOKENS_PER_CHUNK)
        overlap = _env_int("NUMBER_OF_OVERLAPPING_SENTENCES",NUMBER_OF_OVERLAPPING_SENTENCES)
        min_tokens = _env_int("MIN_TOKENS_PER_CHUNK",MIN_TOKENS_PER_CHUNK)
        token_model = os.getenv("TOKEN_ENCODER_MODEL",os.getenv("TOKEN_ENCODER","gpt2"))
        return cls(max_tokens_per_chunk=max_tokens,overlap_sentences=overlap,token_model=token_model,nlp=None,min_tokens_per_chunk=min_tokens)
def run_ocr_on_pil_image(engine_name: str,engine_obj,pil_img: Image.Image) -> str:
    if engine_name == "rapidocr" and engine_obj is not None:
        try:
            import numpy as np,cv2
            img_arr = None
            if isinstance(pil_img,Image.Image):
                img_arr = np.array(pil_img.convert("RGB"))[:,:,::-1].copy()
            elif isinstance(pil_img,(bytes,bytearray)):
                nparr = np.frombuffer(pil_img,np.uint8)
                img_arr = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
            else:
                try:
                    img_arr = np.asarray(pil_img)
                except Exception:
                    img_arr = None
            if img_arr is None:
                logger.error("RapidOCR input image conversion failed")
                return ""
            res = engine_obj(img_arr)
            if isinstance(res,tuple) and len(res) >= 1:
                ocr_result = res[0]
            else:
                ocr_result = res
            lines: List[str] = []
            if isinstance(ocr_result,list):
                for item in ocr_result:
                    if isinstance(item,dict) and "text" in item:
                        txt = item.get("text") or item.get("rec") or ""
                        if txt:
                            lines.append(str(txt))
                            continue
                    if isinstance(item,(list,tuple)):
                        found = False
                        for element in item:
                            if isinstance(element,str) and element.strip():
                                lines.append(element.strip())
                                found = True
                                break
                            if isinstance(element,(list,tuple)) and element and isinstance(element[0],str):
                                lines.append(element[0].strip())
                                found = True
                                break
                        if found:
                            continue
                        try:
                            joined = " ".join([str(x) for x in item if x is not None])
                            if joined.strip():
                                lines.append(joined.strip())
                                continue
                        except Exception:
                            pass
                    try:
                        s = str(item)
                        if s and s.strip():
                            lines.append(s.strip())
                    except Exception:
                        pass
            else:
                try:
                    s = str(ocr_result)
                    if s and s.strip():
                        lines.append(s.strip())
                except Exception:
                    pass
            return "\n".join([ln for ln in lines if ln])
        except Exception:
            logger.exception("RapidOCR failed to OCR image")
            return ""
    if engine_name == "tesseract" and engine_obj is not None:
        try:
            pytesseract = engine_obj
            return pytesseract.image_to_string(pil_img,lang=IMAGE_TESSERACT_LANG,config=TESSERACT_CONFIG)
        except Exception:
            logger.exception("Tesseract OCR failed to OCR image")
            return ""
    return ""
def _create_rapidocr_engine(model_dir: str|None = None):
    models_path = model_dir or os.getenv("RAPIDOCR_MODEL_DIR","/opt/models/rapidocr")
    tried = []
    last_exc = None
    candidates = ("rapidocr_onnxruntime","rapidocr")
    for module_name in candidates:
        try:
            with without_cwd_on_syspath():
                mod = importlib.import_module(module_name)
            RapidOCR = getattr(mod,"RapidOCR",None)
            if RapidOCR is None:
                raise ImportError(f"module {module_name} does not expose RapidOCR")
            try:
                eng = RapidOCR(model_dir=models_path)
            except TypeError:
                eng = RapidOCR(models_path)
            return eng
        except Exception as e:
            tried.append((module_name,repr(e)))
            last_exc = e
    raise ImportError("RapidOCR import failed; tried: " + "; ".join(f"{m}:{err}" for m,err in tried)) from last_exc
def get_image_ocr_engine():
    if IMAGE_DISABLE_OCR and not IMAGE_FORCE_OCR:
        logger.info("IMAGE_DISABLE_OCR=true and IMAGE_FORCE_OCR=false -> skipping OCR")
        return "none",None
    choice = (IMAGE_OCR_ENGINE or "auto").lower()
    if choice == "rapidocr":
        try:
            eng = _create_rapidocr_engine()
            logger.info("Using RapidOCR model_dir=%s",os.getenv("RAPIDOCR_MODEL_DIR","/opt/models/rapidocr"))
            return "rapidocr",eng
        except Exception as e:
            logger.exception("Requested RapidOCR but import/create failed: %s",e)
            return "none",None
    if choice == "tesseract":
        try:
            with without_cwd_on_syspath():
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD","tesseract")
                logger.info("Using Tesseract OCR")
                return "tesseract",pytesseract
        except Exception as e:
            logger.exception("Requested Tesseract but import failed: %s",e)
            return "none",None
    if choice == "auto":
        try:
            eng = _create_rapidocr_engine()
            logger.info("Auto-selected RapidOCR model_dir=%s",os.getenv("RAPIDOCR_MODEL_DIR","/opt/models/rapidocr"))
            return "rapidocr",eng
        except Exception as e_rapid:
            logger.warning("RapidOCR auto-select failed: %s",repr(e_rapid))
            try:
                with without_cwd_on_syspath():
                    import pytesseract
                    pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD","tesseract")
                    logger.info("Auto-selected Tesseract")
                    return "tesseract",pytesseract
            except Exception as e_tess:
                logger.warning("Tesseract auto-select failed: %s",repr(e_tess))
                logger.error("No OCR engine available. OCR will be skipped.")
                return "none",None
    try:
        eng = _create_rapidocr_engine()
        logger.info("Fallback RapidOCR")
        return "rapidocr",eng
    except Exception as e:
        logger.exception("Fallback RapidOCR failed: %s",e)
        return "none",None
def reflow_and_clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'[\x00-\x1F]+',' ',text)
    text = text.replace('\r\n','\n').replace('\r','\n')
    text = re.sub(r'\n{2,}','\n\n',text)
    text = re.sub(r'(?<!\n)\n(?!\n)',' ',text)
    text = re.sub(r'\s+',' ',text).strip()
    return text
def download_s3_object_to_temp(s3_key: str,ext: str) -> str:
    suffix = f".{ext}" if not ext.startswith(".") else ext
    tmpdir = os.getenv("TMPDIR") or None
    tf = tempfile.NamedTemporaryFile(delete=False,suffix=suffix,dir=tmpdir)
    try:
        s3.download_fileobj(Bucket=S3_BUCKET,Key=s3_key,Fileobj=tf)
    finally:
        tf.flush();tf.close()
    return tf.name
def _mime_type_for_ext(ext: str) -> str:
    e = ext.lower().lstrip(".")
    mapping = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp","tif":"image/tiff","tiff":"image/tiff","bmp":"image/bmp","gif":"image/gif"}
    return mapping.get(e,"application/octet-stream")
def preprocess_for_ocr(pil_img: Image.Image,target_dpi: int|None = None,upscale_factor: float|None = None) -> Image.Image:
    import numpy as np,cv2
    img = pil_img.convert("RGB")
    arr = np.array(img)[:,:,::-1].copy()
    h,w = arr.shape[:2]
    if target_dpi:
        scale = max(1.0,float(target_dpi)/150.0)
    elif upscale_factor:
        scale = float(upscale_factor)
    else:
        if IMAGE_UPSCALE_FACTOR and IMAGE_UPSCALE_FACTOR > 1.0:
            scale = IMAGE_UPSCALE_FACTOR
        elif max(w,h) < 800:
            scale = 4.0
        elif max(w,h) < 1500:
            scale = 2.5
        else:
            scale = 1.0
    if scale != 1.0:
        new_w = max(32,int(w*scale)); new_h = max(32,int(h*scale))
        arr = cv2.resize(arr,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray,(0,0),sigmaX=1.0)
    sharpen = cv2.addWeighted(gray,1.5,blur,-0.5,0)
    try:
        th0 = cv2.threshold(sharpen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        coords = cv2.findNonZero(cv2.bitwise_not(th0))
        angle = 0.0
        if coords is not None:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
        if abs(angle) > 0.1:
            (h2,w2) = sharpen.shape[:2]
            M = cv2.getRotationMatrix2D((w2//2,h2//2),angle,1.0)
            sharpen = cv2.warpAffine(sharpen,M,(w2,h2),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass
    th = cv2.adaptiveThreshold(sharpen,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    th = cv2.morphologyEx(th,cv2.MORPH_CLOSE,kernel,iterations=1)
    th = cv2.medianBlur(th,3)
    out = cv2.cvtColor(th,cv2.COLOR_GRAY2RGB)
    return Image.fromarray(out[:,:,::-1])
def postprocess_ocr_text(text: str) -> str:
    text = canonicalize_text(text)
    text = reflow_and_clean_text(text)
    if not IMAGE_ENABLE_WORDSEGMENT:
        return text
    try:
        from wordsegment import load,segment
        load()
        tokens = []
        for tok in text.split():
            if len(tok) > 8 and " " not in tok and tok.isalpha():
                seg = segment(tok)
                if seg and len(" ".join(seg)) < len(tok) + 5:
                    tokens.append(" ".join(seg))
                    continue
            tokens.append(tok)
        return " ".join(tokens)
    except Exception:
        return text
def process_image_s3_object(s3_key: str,manifest: dict) -> dict:
    start_all = time.perf_counter()
    try:
        head_obj = s3.head_object(Bucket=S3_BUCKET,Key=s3_key)
    except Exception:
        head_obj = {}
    doc_id = _derive_doc_id_from_head(s3_key,head_obj or {},manifest or {})
    ext = "jsonl" if CHUNK_FORMAT == "jsonl" else "json"
    out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("Skipping because chunked file exists: %s",out_key)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True}
    lower = s3_key.lower()
    allowed = (".jpg",".jpeg",".png",".webp",".tif",".tiff",".bmp",".gif")
    if not any(lower.endswith(a) for a in allowed):
        total_ms = int((time.perf_counter() - start_all) * 1000)
        msg = "Unsupported file extension for images parser"
        logger.error(msg + ": %s",s3_key)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":msg}
    ext_on_disk = os.path.splitext(s3_key)[1] or ".img"
    try:
        local_img = download_s3_object_to_temp(s3_key,ext_on_disk)
    except Exception as e:
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.error("Could not download S3 object %s: %s",s3_key,e)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e)}
    if isinstance(manifest,dict) and manifest.get("file_hash"):
        doc_id = manifest.get("file_hash")
        out_key = f"{S3_CHUNKED_PREFIX}{doc_id}.{ext}"
    if not FORCE_OVERWRITE and s3_object_exists(out_key):
        try:
            os.unlink(local_img)
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("Skipping because chunked file exists (post-download): %s",out_key)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True}
    img_ocr_name,img_ocr_obj = get_image_ocr_engine()
    chunker = SentenceChunker.from_env()
    try:
        writer = S3DocWriter(doc_id=doc_id,s3_path=f"s3://{S3_BUCKET}/{s3_key}",ext=ext)
        saved = 0
        try:
            im = Image.open(local_img)
        except Exception as e:
            try:
                os.unlink(local_img)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.exception("PIL failed to open image %s: %s",s3_key,e)
            return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e)}
        n_frames = getattr(im,"n_frames",1)
        for frame_idx in range(n_frames):
            frame_start = time.perf_counter()
            try:
                if n_frames > 1:
                    im.seek(frame_idx)
                frame = im.convert("RGB")
            except Exception:
                frame = im.convert("RGB")
            try:
                frame = preprocess_for_ocr(frame,target_dpi=IMAGE_RENDER_DPI)
            except Exception:
                pass
            buf = io.BytesIO()
            frame.save(buf,format="PNG")
            img_bytes = buf.getvalue()
            used_ocr = False
            ocr_text = ""
            if img_bytes and len(img_bytes) >= IMAGE_MIN_IMG_SIZE_BYTES and img_ocr_name != "none":
                ocr_text = run_ocr_on_pil_image(img_ocr_name,img_ocr_obj,frame)
                if ocr_text and ocr_text.strip():
                    used_ocr = True
                    ocr_text = postprocess_ocr_text(ocr_text)
            if not ocr_text:
                chunk_id = f"{doc_id}_f{frame_idx+1}_0"
                payload = {"document_id":doc_id,"file_name":os.path.basename(s3_key),"chunk_id":chunk_id,"chunk_type":"image_frame_chunk","text":"","token_count":0,"embedding":None,"figures":[],"file_type":_mime_type_for_ext(ext_on_disk),"source_url":f"s3://{S3_BUCKET}/{s3_key}","page_number":frame_idx+1,"timestamp":datetime.utcnow().isoformat()+"Z","parser_version":PARSER_VERSION_IMAGE,"tags":manifest.get("tags",[]) if isinstance(manifest,dict) else [],"layout_tags":[],"used_ocr":used_ocr,"heading_path":[],"headings":[],"line_range":None,"layout_bbox":None}
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                saved += 1
                logger.info("Buffered empty frame chunk %s",chunk_id)
                continue
            for idx,chunk in enumerate(chunker.chunk_document(ocr_text)):
                chunk_id = f"{doc_id}_f{frame_idx+1}_{idx}"
                payload = {"document_id":doc_id,"file_name":os.path.basename(s3_key),"chunk_id":chunk_id,"chunk_type":"image_frame_chunk","text":chunk["text"],"token_count":int(chunk["token_count"]),"embedding":None,"figures":[],"file_type":_mime_type_for_ext(ext_on_disk),"source_url":f"s3://{S3_BUCKET}/{s3_key}","page_number":frame_idx+1,"timestamp":datetime.utcnow().isoformat()+"Z","parser_version":PARSER_VERSION_IMAGE,"tags":manifest.get("tags",[]) if isinstance(manifest,dict) else [],"layout_tags":[],"used_ocr":used_ocr,"heading_path":[],"headings":[],"line_range":None,"layout_bbox":None}
                sanitize_payload_for_weaviate(payload)
                writer.write_payload(payload)
                saved += 1
            frame_ms = int((time.perf_counter() - frame_start) * 1000)
            logger.info("Processed frame %d/%d (%d ms) chunks so far %d",frame_idx+1,n_frames,frame_ms,saved)
        if saved == 0:
            try:
                if writer and getattr(writer,"temp",None):
                    try:
                        os.unlink(writer.temp.name)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                os.unlink(local_img)
            except Exception:
                pass
            total_ms = int((time.perf_counter() - start_all) * 1000)
            logger.info("No chunks produced for %s",s3_key)
            return {"saved_chunks":0,"total_parse_duration_ms":total_ms}
        count,uploaded_key = writer.finalize_and_upload(out_key)
        try:
            os.unlink(local_img)
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.info("Wrote %d chunks for %s → %s (%d ms)",count,s3_key,uploaded_key,total_ms)
        return {"saved_chunks":count,"total_parse_duration_ms":total_ms,"skipped":False}
    except Exception as e:
        try:
            if writer and getattr(writer,"temp",None):
                try:
                    os.unlink(writer.temp.name)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            os.unlink(local_img)
        except Exception:
            pass
        total_ms = int((time.perf_counter() - start_all) * 1000)
        logger.exception("Error while processing %s: %s",s3_key,str(e))
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e)}
def parse_file(s3_key: str,manifest: dict) -> dict:
    start = time.perf_counter()
    if S3_BUCKET is None:
        raise RuntimeError("S3_BUCKET must be set in environment")
    try:
        result = process_image_s3_object(s3_key,manifest or {})
        return result
    except Exception as e:
        total_ms = int((time.perf_counter() - start) * 1000)
        logger.exception("parse_file error for %s: %s",s3_key,e)
        return {"saved_chunks":0,"total_parse_duration_ms":total_ms,"skipped":True,"error":str(e)}
if __name__ == "__main__":
    engine_name,engine_obj = get_image_ocr_engine()
    logger.info("Engine result: %s %s",engine_name,"object_loaded" if engine_obj else "none")
