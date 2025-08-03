

# STEP 2/3 - indexing_pipeline

#### **NVIDIA (June 2025)** : Page-level chunking is the baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

> “Page-level chunking is the overall winner: Our experiments clearly show that page-level chunking achieved the highest average accuracy (0.648) across all datasets and the lowest standard deviation (0.107), showing more consistent performance across different content types. Page-level chunking demonstrated superior overall performance when compared to both token-based chunking and section-level chunking.” 

#### RAG8s implements page-wise chunking and similar chunking for scalability without losing accuracy
<details>
  <summary> View chunk stratergies and chunk schema (Click the triangle)</summary>

```sh

```



| **Component**           | **Tool(s)**                                                                                                          | **Exact Chunking Strategy**                                                                                                                                                                                                                                                                                                                                                          | **Why Chosen for Scalability**                                                                                                                                                                                             |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **PDF Parsing + OCR**   | `PyMuPDF`, `pdfplumber`, `pytesseract`, `RapidOCR`, `IndicOCR`, `OpenCV`, `boto3`, `PIL`, `uuid`, `tiktoken`, `json` | - **1 PDF page = 1 chunk** (strict page granularity) <br> - Try native text via **pdfplumber**, fallback to OCR (Tesseract, RapidOCR, IndicOCR) <br> - Per-page OCR is conditional and hybridized with image-based OCR (OpenCV crops) <br> - Tables are extracted and injected as both: **structured JSON** and **inlined text** <br> - Figures/diagrams parsed as `"figures"` array | - Page-based chunking ensures **parallelism**, simplicity, and caching <br> - Multiple OCR backends improve multilingual and noisy scan robustness <br> - Preserving layout fidelity improves table/graph capture accuracy |
| **DOC/DOCX Conversion** | `LibreOffice (headless)`, `boto3`, `tempfile`, `subprocess`, `os`, `pathlib`                                         | - Convert `.doc` → `.docx` → `.pdf` using LibreOffice CLI <br> - Parse resulting PDF **per-page** using same pipeline as native PDF <br> - Page-to-page correspondence preserved across conversions                                                                                                                                                                                  | - Avoids fragile `.doc/.docx` parsing with Python libraries <br> - Ensures **visual consistency** across platforms <br> - Seamless reuse of existing PDF+OCR pipeline                                                      |
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
export OCR_ENGINE=tesseract            # (OR) `rapidocr` for complex english 
export FORCE_OCR=false                 # (OR) true = always OCR; false = skip if text exists(false recommended)
export OCR_RENDER_DPI=300              # higher dpi = high quality image extraction = higher cost and higher chance of extracting tiny texts
export MIN_IMG_SIZE_BYTES=3072         # Filter out tiny images under 3 KB (often unneccessary black empty images)
export OVERWRITE_DOC_DOCX_TO_PDF=true  # (OR) false if dont want to delete original .doc and .docx files in data/raw/


export HF_TOKEN=



```sh
export TOP_K_CHUNKS=                # number of batches will be calculated accordingly based on tokens in chunk and max tokens of reranker model
```

```sh
+------------------+
|   User Query     |
+------------------+
         |
         v
+-------------------------------+
| gte-modernbert-base Encoder   |
+-------------------------------+
         |
         v
+-------------------------------+
| Dense Top-K Document Retrieval|
+-------------------------------+
         |
         v
+-----------------------------+        +--------------------+
|  Retrieved Documents (Text) | -----> |  ReFinED Entity     |
+-----------------------------+        |  Linking & Linking |
                                        +--------------------+
                                                  |
                                                  v
                                +---------------------------------+
                                | Graph Nodes:                    |
                                | - Docs (with embeddings)        |
                                | - Entities (Wikidata QIDs)      |
                                +---------------------------------+
                                                  |
                                                  v
                                  +------------------------------+
                                  | Inserted into ArangoDB       |
                                  | (Doc <-> Entity edges)       |
                                  +------------------------------+
                                                  |
                                                  v
                              +--------------------------------------+
                              |   AQL Traversal (1..N ANY hops)     |
                              |   Expand to multi-hop neighbors     |
                              +--------------------------------------+
                                                  |
                                                  v
                          +-----------------------------------------+
                          | gte-reranker-modernbert-base            |
                          | Cross-encodes (Query, Doc) pairs        |
                          | Scores and reranks results              |
                          +-----------------------------------------+
                                                  |
                                                  v
                            +-------------------------------+
                            | LLM Generator (e.g., GPT-4)   |
                            | Answer synthesis or summarizer|
                            +-------------------------------+
```