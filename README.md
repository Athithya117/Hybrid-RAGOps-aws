

# STEP 2/3 - indexing_pipeline

#### **NVIDIA (June 2025)** : Page-level chunking is the baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

> “Page-level chunking is the overall winner: Our experiments clearly show that page-level chunking achieved the highest average accuracy (0.648) across all datasets and the lowest standard deviation (0.107), showing more consistent performance across different content types. Page-level chunking demonstrated superior overall performance when compared to both token-based chunking and section-level chunking.” 

#### RAG8s implements page-wise chunking and similar chunking for scalability without losing accuracy
<details>
  <summary> View chunk stratergies and chunk schema (Click the triangle)</summary>

```sh

```

| **Component**           | **Tool(s)**                                                                      | **Exact Chunking Strategy**                                                                                                                                                                                          | **Why Chosen for Scalability**                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **PDF Parsing + OCR**   | PyMuPDF, pdfplumber, Tesseract / RapidOCR / IndicOCR, OpenCV, pytesseract, boto3 | - 1 PDF page = 1 chunk (strict)  <br> - Native text extracted via pdfplumber  <br> - OCR fallback: full-page and image-specific (inline)  <br> - Tables are extracted and injected both as structured and plain text | - Page-wise chunking simplifies parallelism (Ray/multiprocessing)  <br> - OCR is conditional  <br> - Robust on scans/multilingual |
| **DOC/DOCX Conversion** | LibreOffice CLI (headless), boto3, tempfile, subprocess                          | - Convert `.doc` → `.docx` → `.pdf` using LibreOffice  <br> - Then parsed page-by-page as PDF                                                                                                                        | - Avoids needing custom Word parser  <br> - Ensures 1:1 page fidelity  <br> - Leverages existing PDF pipeline                     |
| **HTML Parsing**        | extractous, BeautifulSoup                                                        | - Parse HTML DOM tree  <br> - Chunk by headings (`<h1>`–`<h6>`), paragraphs, and sections                                                                                                                            | - Lightweight, preserves semantic structure  <br> - Works on both web pages and embedded HTML                                     |
| **CSV Chunking**        | ray.data.read\_csv(), `.window()`                                                | - Stream rows  <br> - Chunk based on size heuristics (`max(5120, avg_row_len * ROWS_PER_CHUNK)`)                                                                                                                     | - Efficient streaming for large files  <br> - Memory-safe, scalable via Ray                                                       |
| **JSON/JSONL**          | ray.data.read\_json(), `.window()`                                               | - JSONL: each line = record  <br> - For nested JSON: flatten → explode arrays → chunk by size/depth                                                                                                                  | - Handles deeply nested or irregular structures  <br> - Flexible chunk size based on token count                                  |
| **Audio Transcription** | faster-whisper (CTranslate2), pydub, ffmpeg-python                               | - Audio sliced into 20–30s segments via silence detection (`pydub.silence`)  <br> - Each segment transcribed individually                                                                                            | - Faster-Whisper is GPU/CPU efficient  <br> - Segmentation makes long audio scalable and parallelizable                           |
| **Markdown**            | markdown-it-py, mistune, regex                                                   | - Chunk by heading levels, paragraphs, and code blocks  <br> - Fallback to fixed-token or sentence-based slicing                                                                                                     | - Preserves Markdown structure  <br> - Compatible with LLM indexing and embeddings                                                |
| **PPTX (PowerPoint)**   | python-pptx, Pillow (optional OCR)                                               | - 1 slide = 1 chunk  <br> - Extract text, speaker notes, images  <br> - OCR fallback on image slides                                                                                                                 | - Natural chunking by slide  <br> - Works well with educational or slide-heavy documents                                          |
| **EPUB/eBooks**         | ebooklib, BeautifulSoup, html5lib                                                | - Chunk by chapters/headings from EPUB metadata  <br> - Paragraph or heading-based segmentation within chapters                                                                                                      | - Structure-aware  <br> - Works with long-form content like books                                                                 |
| **Images (Scans)**      | OpenCV, PIL/Pillow, Tesseract or RapidOCR                                        | - 1 image = 1 chunk  <br> - OCR applied to entire image or regions (if detected)                                                                                                                                     | - Useful for form scans, handwritten notes, flyers  <br> - Preserves visual layout                                                |
| **ZIP Archives**        | zipfile, tarfile, custom dispatcher                                              | - Files extracted, routed to correct parsers based on extension (pdf, docx, txt, etc.)                                                                                                                               | - Allows batch ingestion  <br> - Enables unified multi-file upload experience                                                     |
| **Plaintext Files**     | open(), re, nltk, tiktoken (optional)                                            | - Chunk by paragraph, newline gaps (`\n\n`), or fixed line/token window                                                                                                                                              | - Extremely lightweight  <br> - Works well with logs, scraped data, or long articles                                              |

</details>


### Export the neccessary configs.

```sh

export S3_BLOCK_PUBLIC_ACCESS=false
export S3_BUCKET=e2e-rag-system      # Give a complex name


export S3_RAW_PREFIX=data/raw/         
export S3_CHUNKED_PREFIX=data/chunked/   
export CHUNK_FORMAT=json               # (OR) 'jsonl' for faster read and storage efficiency for headless use(but not readable)
export DISABLE_OCR=false               # (OR) true = disable ocr and the text in images of docs will not be extracted(but very fast)
export OCR_ENGINE=tesseract            # (OR) `rapidocr` for complex english (OR) tesseract for fast or multilingual (OR) `indicocr` for indian languages
export FORCE_OCR=false                 # (OR) true = always OCR; false = skip if text exists(false recommended)
export OCR_RENDER_DPI=300              # higher dpi = high quality image extraction = higher cost and higher chance of extracting tiny texts
export MIN_IMG_SIZE_BYTES=3072         # Filter out tiny images under 3 KB (often unneccessary black empty images)
export IS_MULTILINGUAL=false           # (OR) true. if false, TESSERACT_LANG will be ignored
export TESSERACT_LANG=eng              #  for both `indicocr` and `tesseract` (but only one lang to avoid noise). refer the mapping table below. 
export OVERWRITE_DOC_DOCX_TO_PDF=true  # (OR) false if dont want to delete original .doc and .docx files in data/raw/


export HF_TOKEN=
export EMBEDDING_EL_DEVICE=cpu      # or gpu for indexing with embedding and entity linking models
export EMBED_MODEL="elastic/multilingual-e5-small-optimized" # or View recommendations
export LOAD_IN=int8




```

<details>
  <summary> View languages table, embedding and EL models recommendations(Click the triangle)</summary>


| Language(Abbr)        | Language(Abbr)            | Language(Abbr)  | Language(Abbr)              | Language(Abbr)       | Language(Abbr)             | Language(Abbr)  |
| --------------------- | ------------------------- | --------------- | --------------------------- | -------------------- | -------------------------- | --------------- |
| Amharic(amh)          | French(fra)               | Oriya(ori)      | Serbian(srp)                | Urdu(urd)            | Azerbaijani(aze)           | Dutch(nld)      |
| Assamese(asm)         | French-Fraktur(fra\_frak) | Pashto(pus)     | Serbian-Cyrillic(srp\_cyrl) | Uzbek(uzb)           | Azerbaijani-Cyr(aze\_cyrl) | Esperanto(epo)  |
| Bosnian(bos)          | Galician(glg)             | Persian(fas)    | Serbian-Latin(srp\_latn)    | Uzbek-Cyr(uzb\_cyrl) | Finnish(fin)               | Latin(lat)      |
| Catalan(cat)          | German(deu)               | Polish(pol)     | Sinhala(sin)                | Vietnamese(vie)      | Tamil(tam)                 | Hebrew(heb)     |
| Cebuano(ceb)          | Greek(ell)                | Portuguese(por) | Slovak(slk)                 | Welsh(cym)           | Telugu(tel)                | Hindi(hin)      |
| Chinese-Sim(chi\_sim) | Gujarati(guj)             | Punjabi(pan)    | Slovenian(slv)              | Yiddish(yid)         | Thai(tha)                  | Hungarian(hun)  |
| Chinese-Tra(chi\_tra) | Haitian(hat)              | Romanian(ron)   | Spanish(spa)                | Yoruba(yor)          | Tibetan(bod)               | Icelandic(isl)  |
| Croatian(hrv)         | Hausa(hau)\*              | Russian(rus)    | Swahili(swa)                | Kurdish(kur)         | Japanese(jpn)              | Indonesian(ind) |
| Czech(ces)            | Khmer(khm)                | Lao(lao)        | Lithuanian(lit)             | Kannada(kan)         | Javanese(jav)              | Italian(ita)    |
| Danish(dan)           | Korean(kor)               | Latvian(lav)    | Nepali(nep)                 | Sinhala(sin)         | Oriya(ori)                 | Malay(may)\*    |
| English(eng)          | Tigrinya(tir)             | Estonian(est)   | Tagalog(tgl)                | Turkish(tur)         | Ukrainian(ukr)             | Uyghur(uig)     |


## Recommendations:
  ### 1. The same embedding and EL model should be used in both indexing pipeline and in inference pipeline(deployment) so choose wisely.
  ### 2. Smaller models performs closer to larger models, so even if you need slightly higher accuracy, choose smaller models as the inference will have graph based multi hop retreival also. The fusion of multiple smaller and int8 versions is better than fewer large models. 
  ### 3. `elastic/multilingual-e5-small-optimized` is a highly efficient multilingual model supporting 90+ languages but supports dense(similarity) retreival only. 
  ### 4. `Alibaba-NLP/gte-multilingual-base(mGTE-TRM)` have long‑context support and improved multilingual retrieval. It supports both sparse(keyword) and dense(similarity) retreival but there isn't an english only version of `mGTE-TRM`
  ### 5. Use `Alibaba‑NLP/gte-modernbert-base` or `intfloat/e5-small` if the data is english only. If sparse retreival also needed, choose `mGTE‑TRM`
  ### 6. For the env variable `EMBED_MODEL`, kindly choose only the models in these tables as they were tested in RAG8s.

---

### Recommeded models:
| **Model**                                   | **MRR @10 / MTEB**                                  | **Params** | **Size (float32)**    | **Embed Dim** | **Max Tokens** | **VRAM (fp32)** | **VRAM (fp16)** | **VRAM (int8)** |
| ------------------------------------------- | --------------------------------------------------- | ---------- | --------------------- | ------------- | -------------- | --------------- | --------------- | --------------- |
| **Alibaba-NLP/gte-multilingual-base**    | \~ 68–71 MRR\@10 (MIRE M) / \~ 71 nDCG\@10 (MIRACL) | \~ 304 M   | \~ 1.2 GB (est.)      | 768–1024      | 8192           | \~ 5–7 GB       | \~ 3–4 GB       | \~ 1.8–2.2 GB   |
| **elastic/multilingual‑e5‑small‑optimized** | \~ 64.4 MRR\@10 (average)                           | \~ 110 M   | – (int8 quant)        | 384           | 512            | \~ 1–1.5 GB     | n/a             | \~ 1 GB         |
| **Alibaba‑NLP/gte-modernbert-base**                   | \~ 64.38 avg                                        | \~ 149 M   | \~ 0.67 GB (≈ 670 MB) | 768           | 8192           | \~ 5–6 GB       | \~ 3–4 GB       | \~ 2–2.5 GB     |

---

### Other models to compare: 
| **Model**                                   | **MRR\@10 / MTEB**                          | **Params** | **Size (float32)** | **Embed Dim** | **Max Tokens** | **VRAM (fp32)** | **VRAM (fp16)** | **VRAM (int8)** |
| ------------------------------------------- | ------------------------------------------- | ---------- | ------------------ | ------------- | -------------- | --------------- | --------------- | --------------- |
| **elastic/multilingual-e5-small**           | 64.4 MRR\@10 (average)     | \~ 110 M   | \~ 440 MB          | 384           | 512            | \~ 2–3 GB       | \~ 1.5–2 GB     | \~ 1–1.2 GB     |
| **elastic/multilingual-e5-base**            | 65.9 MRR\@10 (average)     | \~ 260 M   | \~ 1.0 GB          | 768           | 512            | \~ 4–6 GB       | \~ 2.5–3.5 GB   | \~ 1.5–2 GB     |
| **elastic/multilingual-e5-large**           | n/a (not published)                         | \~ 500 M   | \~ 2.0 GB          | 1024          | 512            | \~ 8–10 GB      | \~ 4.5–6 GB     | \~ 2.5–3.5 GB   |
| **intfloat/e5-small**                       | 64.4 MRR\@10 (average)     | \~ 110 M   | \~ 440 MB          | 384           | 512            | \~ 2–3 GB       | \~ 1.5–2 GB     | \~ 1–1.2 GB     |
| **intfloat/e5-base**                        | 65.9 MRR\@10 (average)     | \~ 260 M   | \~ 1.0 GB          | 768           | 512            | \~ 4–6 GB       | \~ 2.5–3.5 GB   | \~ 1.5–2 GB     |
| **intfloat/e5-large**                       | n/a (not published)                         | \~ 500 M   | \~ 2.0 GB          | 1024          | 512            | \~ 8–10 GB      | \~ 4.5–6 GB     | \~ 2.5–3.5 GB   |
| **gte‑base‑en‑v1.5**  | \~ 62.39 avg           | \~ 137 M   | \~ 0.22 GB (≈ 220 MB) | 768           | 8192           | \~ 2.5–3.5 GB   | \~ 1.5–2.5 GB   | \~ 1 GB         |
| **gte‑large‑en‑v1.5** | \~ 63.13 avg           | \~ 434 M   | \~ 0.67 GB (≈ 670 MB) | 1024          | 8192           | \~ 5–7 GB       | \~ 3–4 GB       | \~ 2–2.5 GB     |

</details>



---

