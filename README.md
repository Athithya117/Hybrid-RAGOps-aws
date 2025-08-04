

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




# Models overview
---
| Model                        | Language     | Params | Max Tokens              | Efficiency           | Triplet μ-F1 | size (FP32)  | size (INT8 / W4A16) | RAM needed (INT8) | VRAM needed (W4A16) |
| ---------------------------- | ------------ | ------ | ----------------------- | -------------------- | ------------ | ------------ | ------------------- | ----------------- | ------------------- |
| gte-modernbert-base          | English      | 149M   | 8,192                   | High (CPU/ONNX)      | —            | 596 MiB      | 149 MiB (INT8)      | 1.0 GiB           | —                   |
| gte-reranker-modernbert-base | English      | 149M   | 8,192                   | Very high (CPU/ONNX) | —            | 598 MiB      | 149 MiB (INT8)      | 1.0 GiB           | —                   |
| relik-cie-tiny               | English      | 174M   | \~3,000                 | Very high (CPU/ONNX) | 73.8%        | 663.8 MiB    | 165.9 MiB (INT8)    | 0.94 GiB          | —                   |
| relik-cie-small              | English      | 216M   | \~3,000                 | Very high (CPU/ONNX) | 74.3%        | 824.0 MiB    | 206.0 MiB (INT8)    | 1.40 GiB          | —                   |
| relik-cie-large              | English      | 467M   | \~3,000                 | High (CPU/ONNX)      | 75.6%        | 1,781.5 MiB  | 445.4 MiB (INT8)    | 2.43 GiB          | —                   |
| Qwen3-0.6B-quantized.w4a16   | Multilingual | 600M   | 32,768                  | High (W4A16 AWQ)     | —            | \~2.4 GiB\*  | 860 MiB (W4A16)     | —                 | \~1.1 GiB           |
| Qwen3-1.7B-quantized.w4a16   | Multilingual | 1.7B   | 32,768                  | High (W4A16 AWQ)     | —            | \~6.8 GiB\*  | 2.0 GiB (W4A16)     | —                 | \~2.7 GiB           |
| Qwen3-4B-quantized.w4a16     | Multilingual | 4B     | 32,768 (native)         | High (W4A16 AWQ)     | —            | \~16.0 GiB\* | 3.43 GiB (W4A16)    | —                 | \~5.6 GiB           |
| Qwen3-8B-quantized.w4a16     | Multilingual | 8.2B   | 32,768 / 131,072 (YaRN) | High (W4A16 AWQ)     | —            | \~32.8 GiB\* | \~6.5 GiB (W4A16)†  | —                 | \~11 GiB            |
| Qwen3-14B-quantized.w4a16    | Multilingual | 14.8B  | 32,768 / 131,072 (YaRN) | High (W4A16 AWQ)     | —            | \~59.2 GiB\* | \~10.9 GiB (W4A16)† | —                 | \~18 GiB            |

---


## 🔗 **References & specialties of the default models in RAG8s**

---

### 🔹 **\[1] gte-modernbert-base**

* Embedding-only model for dense retrieval in RAG pipelines
* Long-context support: up to **8192 tokens**
* Based on **ModernBERT** (FlashAttention 2, RoPE, no position embeddings)
* Optimized for ONNX export and CPU‑inference
* Embedding dimension: **768**
* Parameter size: **149M**
  🔗 [https://huggingface.co/Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)

> **Use case**: Recommended for fast, CPU-efficient **semantic retrieval** in low-latency RAG pipelines with long context document embeddings.

---

### 🔹 **\[2] gte-reranker-modernbert-base**

* **Cross-encoder reranker** for re-ranking retrieved docs
* High BEIR benchmark score (**nDCG\@10 ≈ 90.7%**)
* Same architecture & size as embedding model (149M), supports **8192 tokens**
* Extremely fast CPU inference with ONNX (FlashAttention 2)
  🔗 [https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)

> **Use case**: Ideal for **re-ranking top-k retrieved passages** after dense retrieval to improve precision in RAG answer selection.

---

### 🔹 **\[3] ReLiK-CIE-Tiny**

A compact and efficient **entity + relation extraction** model designed for **Graph-RAG pipelines**. Unlike fast entity-only models (e.g., SpEL, ReFinED), `relik-cie-tiny` can extract both **named entities** and **semantic triplets** (`(head, relation, tail)`), enabling direct construction of **knowledge subgraphs** from raw text.

* Extracts **entities and triplets** in a single pass
* Suitable for **CPU inference** via ONNX
* Balanced for **accuracy and runtime performance**
  🔗 [Model card on HuggingFace](https://huggingface.co/tau/relik-cie-tiny)

> **Use case**: Ideal for production Graph-RAG inference pipelines where lightweight models must still generate structured knowledge without relying on large LLM backends.

---

### 🔹 **\[4] RedHatAI/Qwen3-4B-W4A16**

A compact, high-throughput **instruction-tuned LLM** quantized using **W4A16** (4-bit weights + FP16 activations). Built on **Qwen3-4B**, this variant supports **32,768-token context** natively and achieves performance comparable to models 10× its size (e.g., Qwen2.5-72B). Optimized for **vLLM inference**, it balances **speed, memory efficiency, and accuracy**, running seamlessly on GPUs like A10G, L4, and L40S.

* Architecture: **Transformer** (Qwen3 series, multilingual)
* Context Length: **32k tokens** (vLLM-native)
* Quantization: **W4A16 (AWQ)** — 4-bit weights, FP16 activations
* VRAM Usage: **\~4.8–5.2 GiB** (fits on 24 GiB GPUs with headroom)

🔗 [RedHatAI Qwen3-4B-W4A16](https://huggingface.co/RedHatAI/Qwen3-4B-quantized.w4a16)

> “Even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct.”
> — [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
> — [Thinking-mode](https://qwenlm.github.io/blog/qwen3/#key-features)

> **Use case**: Ideal for production-grade **chat, reasoning, and instruction-following** tasks where **low latency** and **high throughput** are required on **commodity GPUs**.

---
