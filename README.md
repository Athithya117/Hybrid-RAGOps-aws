

# STEP 2/3 - indexing_pipeline

#### **NVIDIA (June 2025)** : Page-level chunking is the baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/

> “Page-level chunking is the overall winner: Our experiments clearly show that page-level chunking achieved the highest average accuracy (0.648) across all datasets and the lowest standard deviation (0.107), showing more consistent performance across different content types. Page-level chunking demonstrated superior overall performance when compared to both token-based chunking and section-level chunking.” 

#### RAG8s implements page-wise chunking and similar chunking for scalability without losing accuracy
<details>
  <summary> View chunk stratergies and chunk schema (Click the triangle)</summary>

```sh

{
  "chunk_id": "a3f5be12c9d47e09_5",               // Unique chunk ID: <document_hash>_<chunk_index> (1-based)
  "document_id": "a3f5be12c9d47e09",              // Unique document ID (128-bit hash of file path + size)
  "chunk_type": "page",                           // Type of content: "paragraph", "heading", "section", "table", "page", etc.

  "text": "Cleaned markdown-formatted content of the chunk.",  // Final parsed content (Markdown)
  "embedding": null,                              // Optional: vector embedding (array of floats); null if not yet computed

  "source": {
    "file_type": "application/pdf",                  // MIME type preferred (e.g., "application/pdf", "text/html", "audio/mpeg")
    "source_path": "s3://bucket/data/raw/file.pdf",  // Full s3 path to original source
    "page_number": 3,                                // For paged formats like PDF/ePub; null otherwise
    "time": [null, null],                            // [start_time, end_time] in seconds for audio/video; nulls otherwise
    "line_range": null,                              // For plain/tabular text: [start_line, end_line]; null otherwise
    "bbox": null                                     // For visual formats: [x0, y0, x1, y1] in pixel coordinates; null otherwise
  },

  "graph": {
    "graph_node_id": "a3f5be12c9d47e09_5",       // Same as chunk_id (recommended)
    "parent_id": "a3f5be12c9d47e09_page3",       // Parent node ID (e.g., page, section, table)
    "previous_id": "a3f5be12c9d47e09_4",         // Optional: previous chunk
    "next_id": "a3f5be12c9d47e09_6"              // Optional: next chunk
  },

  "metadata": {
    "timestamp": "2025-08-03T12:00:00Z",         // UTC ISO timestamp of chunk creation/parsing
    "tags": ["invoice", "header"],               // High-level content tags (semantic or manual)
    "layout_tags": ["paragraph"],                // Structural tags (e.g., "heading", "table", etc.)
    "entities": ["Q123", "Q456"]                 // Optional: Linked entity IDs (Wikidata, etc.) or null if not yet computed
  }
}

```


## Component-Level Parsing & Chunking Strategy

| Component            | Tool(s) Used                                                                                                  | Chunking Strategy                                                                                                         | Rationale for Scalability & Fidelity                                                                 |
|----------------------|---------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **PDF Parsing + OCR** | `PyMuPDF`, `pdfplumber`, `pytesseract`, `RapidOCR` | - **1 page = 1 chunk** (primary granularity)<br>- Extract native text via `pdfplumber`, fallback to OCR if needed<br>- Use `OpenCV` for layout-aware OCR (cropping regions, columnar reading)<br>- Tables extracted as both **structured JSON** and **inline Markdown**<br>- Figures/diagrams extracted to a separate `figures` array | - Page-level chunking aligns with layout and enables **parallelism**<br>- Hybrid OCR improves coverage of low-quality scans<br>- Layout fidelity helps preserve **tables, headers, visual order**                             |
| **DOC/DOCX Conversion** | `LibreOffice` (headless mode), `subprocess`, `tempfile`, `boto3`, `pathlib`, `os`                             | - Convert `.doc` → `.docx` → `.pdf` using `LibreOffice CLI`<br>- Apply **same PDF+OCR pipeline per page** on output PDF<br>- Page alignment maintained between original and converted formats                        | - Avoids unreliable native `.docx` parsing<br>- Ensures **visual and semantic consistency** across systems<br>- Helps in tracing                                                   |
| **Text/HTML Parsing** | `BeautifulSoup`, `html2text`, custom chunkers                                                                 | - Segment by structural tags: `h1–h6`, `p`, `li`, `div`, `table`<br>- Normalize output into **Markdown chunks**<br>- Chunk IDs and parent-child relations inferred from tag hierarchy                         | - Converts semi-structured content into **RAG-ready Markdown**<br>- Retains **hierarchy and inline metadata** for graph linkage<br>- Works well with multi-format source ingestion                      |                                            |
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

export RAY_DASHBOARD_PORT=8265

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
                            | LLM Generator (Qwen3)         |
                            | Answer synthesis or summarizer|
                            +-------------------------------+
```

---

### **Model Overview Table**

| Model                             | Language         | Params | Max Tokens                       | Efficiency           |
| --------------------------------- | ---------------- | ------ | -------------------------------- | -------------------- |
| gte-modernbert-base               | English          | 149M   | 8,192                            | High (CPU/ONNX)      |
| gte-reranker-modernbert-base      | English (rerank) | 149M   | 8,192                            | Very high (CPU/ONNX) |
| ReFinED (entity linker)           | English          | \~125M | \~3,000+                         | Very high (CPU/ONNX) |
| unsloth/Qwen3-0.6B-bnb-4bit       | Multilingual     | 600M   | 32,768                           | High (Dynamic 4‑bit) |
| unsloth/Qwen3-1.7B-bnb-4bit       | Multilingual     | 1.7B   | 32,768                           | High (Dynamic 4‑bit) |
| unsloth/Qwen3-4B-unsloth-bnb-4bit | Multilingual     | 4B     | 32,768 (native)                  | High (Dynamic 4‑bit) |
| unsloth/Qwen3-8B-bnb-4bit         | Multilingual     | 8B     | 32,768 native / 131,072 via YaRN | High (Dynamic 4‑bit) |
| unsloth/Qwen3-14B-bnb-4bit        | Multilingual     | 14B    | 32,768 native / 131,072 via YaRN | High (Dynamic 4‑bit) |

---

### 🔗 **References & Specialties**

**\[1] gte-modernbert-base**
- Embedding-only model optimized for dense retrieval
- CPU-efficient, ONNX-compatible
🔗 [https://huggingface.co/Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)

**\[2] gte-reranker-modernbert-base**
- Reranker model for hybrid/vector search pipelines
- High-quality ranking, very fast on CPU
🔗 [https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)

**\[3] ReFinED**
- Fast and scalable entity linking system
- Claims \~60× speed improvement over traditional EL
🔗 [https://www.amazon.science/code-and-datasets/refined](https://www.amazon.science/code-and-datasets/refined)

**\[4] Qwen3‑4B‑Instruct**
- Versatile multitask instruction-following model
- Supports **32k context length** (vLLM-compatible)
- Compact yet highly performant

> “Even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct.”
> — [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)

🔗 [https://huggingface.co/Qwen/Qwen3-4B-Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct)

---
