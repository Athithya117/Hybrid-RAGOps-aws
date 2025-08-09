

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
    "layout_tags": ["paragraph"]                 // Structural tags (e.g., "heading", "table", etc.)
  },

  "entities": ["Q123", "Q456"],                  // Optional: Linked entity IDs (Wikidata, etc.) or null if not yet computed

  "triplets": [                                  // Extracted subject-predicate-object relations
    {
      "subject": "Invoice",
      "predicate": "hasDate",
      "object": "2025-08-01"
    },
    {
      "subject": "Invoice",
      "predicate": "issuedBy",
      "object": "ACME Corp"
    },
    {
      "subject": "ACME Corp",
      "predicate": "locatedIn",
      "object": "New York"
    }
  ]
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

export S3_BUCKET=e2e-rag-system        # Give a complex name
export S3_RAW_PREFIX=data/raw/         
export S3_CHUNKED_PREFIX=data/chunked/   
export CHUNK_FORMAT=json               # (OR) 'jsonl' for faster read and storage efficiency for headless use(but not readable)
export DISABLE_OCR=false               # (OR) true = disable ocr and the text in images of docs will not be extracted(but very fast)
export OCR_ENGINE=tesseract            # (OR) `rapidocr` for complex english but slower
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
# 1. vector search
vec_hits = vec_index.query(q_embedding, top_k=20)

# 2. alias fast-match
aliases = alias_index.lookup(query_text)
if aliases.confidence < threshold:
    # 3. lightweight EL
    el_entities = relik_inference(query_text)
else:
    el_entities = aliases

# 4. subgraph expansion
subgraph = arangodb.traverse(start_nodes=vec_hits ∪ el_entities, hops=2, filters=...)

# 5. assemble context + LLM prompt (include provenance)

---

 

# Models overview

---

| Model                        | Language     | Params | Max Tokens              | Efficiency           | Triplet μ-F1 | size (FP32)  | size (INT8 / W4A16) | RAM needed (INT8) | VRAM needed (W4A16) |
| ---------------------------- | ------------ | ------ | ----------------------- | -------------------- | ------------ | ------------ | ------------------- | ----------------- | ------------------- |
| gte-modernbert-base          | English      | 149M   | 8,192                   | High (CPU/ONNX)      | —            | 596 MiB      | 149 MiB (INT8)      | 1.0 GiB           | —                   |
| gte-reranker-modernbert-base | English      | 149M   | 8,192                   | Very high (CPU/ONNX) | —            | 598 MiB      | 149 MiB (INT8)      | 1.0 GiB           | —                   |
| relik-cie-tiny               | English      | 174M   | \~3,000                 | High (CPU)           | 73.8%        | 663.8 MiB    | 165.9 MiB (INT8)    | 0.94 GiB          | —                   |
| relik-cie-small              | English      | 216M   | \~3,000                 | high (CPU)           | 74.3%        | 824.0 MiB    | 206.0 MiB (INT8)    | 1.40 GiB          | —                   |
| relik-cie-large              | English      | 467M   | \~3,000                 | Moderate (CPU)       | 75.6%        | 1,781.5 MiB  | 445.4 MiB (INT8)    | 2.43 GiB          | —                   |
| Qwen3-0.6B-quantized.w4a16   | Multilingual | 600M   | 32,768                  | High (W4A16 AWQ)     | —            | \~2.4 GiB\*  | 860 MiB (W4A16)     | —                 | \~1.1 GiB           |
| Qwen3-1.7B-quantized.w4a16   | Multilingual | 1.7B   | 32,768                  | High (W4A16 AWQ)     | —            | \~6.8 GiB\*  | 2.0 GiB (W4A16)     | —                 | \~2.7 GiB           |
| Qwen3-4B-quantized.w4a16     | Multilingual | 4B     | 32,768 (native)         | High (W4A16 AWQ)     | —            | \~16.0 GiB\* | 3.43 GiB (W4A16)    | —                 | \~5.6 GiB           |
| Qwen3-8B-quantized.w4a16     | Multilingual | 8.2B   | 32,768 / 131,072 (YaRN) | High (W4A16 AWQ)     | —            | \~32.8 GiB\* | \~6.5 GiB (W4A16)†  | —                 | \~11 GiB            |
| Qwen3-14B-quantized.w4a16    | Multilingual | 14.8B  | 32,768 / 131,072 (YaRN) | High (W4A16 AWQ)     | —            | \~59.2 GiB\* | \~10.9 GiB (W4A16)† | —                 | \~18 GiB            |

---

## 🔗 **References & specialties of the default models in RAG8s**

---

### 🔹 **\[1] gte-modernbert-base**

* Embedding-only model for dense retrieval in RAG pipelines
* Long-context support: up to **8192 tokens** (Sufficient for page wise chunking)
* Based on **ModernBERT** (FlashAttention 2, RoPE, no position embeddings)
* Optimized for ONNX export and CPU-inference
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
  🔗 [relik-ie/relik-cie-tiny](https://huggingface.co/relik-ie/relik-cie-tiny)

> **Use case**: Ideal for production Graph-RAG inference pipelines where lightweight models must still generate structured knowledge without relying on large LLM backends.

---

### 🔹 **\[4] RedHatAI/Qwen3-4B-W4A16**

A compact, high-throughput **instruction-tuned LLM** quantized using **W4A16** (4-bit weights + FP16 activations). Built on **Qwen3-4B**, this variant supports **32,768-token context** natively and achieves performance comparable to models 10× its size (e.g., Qwen2.5-72B). Optimized for **SGLang inference**, it balances **speed, memory efficiency, and accuracy**, running seamlessly on GPUs like A10G, L4, and L40S.

* Architecture: **Transformer** (Qwen3 series, multilingual)
* Context Length: **32k tokens** (SGLang-native)
* Quantization: **W4A16 (AWQ)** — 4-bit weights, FP16 activations
* VRAM Usage: **\~4.8–5.2 GiB** (fits on 24 GiB GPUs with headroom)

🔗 [RedHatAI Qwen3-4B-W4A16](https://huggingface.co/RedHatAI/Qwen3-4B-quantized.w4a16)

> “Even a tiny model like Qwen3-4B can rival the performance of Qwen2.5-72B-Instruct.”
> — [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
> — [Thinking-mode](https://qwenlm.github.io/blog/qwen3/#key-features)

> **Use case**: Smaller models (e.g., Qwen3-4B-W4A16 or 8B) fit on a single VM, making them better suited for data-parallel engines like **SGLang**, rather than tensor-parallel systems like **vLLM**.

---

<details>
 <summary>EC2 (Click the triangle)</summary>

## EC2 instances for spot with fallback pre warmed on-demand scaling with karpenter-gpu

#### **G6f (Fractional L4 GPU) Sizes**

| Instance        | vCPUs | Memory  | GPU Fraction | GPU Memory |
| --------------- | ----- | ------- | ------------ | ---------- |
| **g6f.large**   | 2     | 8 GiB   | 0.125 × L4   | \~3 GiB    |
| **g6f.xlarge**  | 4     | 16 GiB  | 0.25 × L4    | \~6 GiB    |
| **g6f.2xlarge** | 8     | 32 GiB  | 0.5 × L4     | \~12 GiB   |
| **g6f.4xlarge** | 16    | 64 GiB  | 1 × L4       | 24 GiB     |
| **g6f.8xlarge** | 32    | 128 GiB | 2 × L4       | 48 GiB     |

---

#### **G6e (Full L40S GPU) Sizes**

| Instance         | vCPUs | Memory   | GPUs | GPU Memory                               |
| ---------------- | ----- | -------- | ---- | ---------------------------------------- |
| **g6e.xlarge**   | 4     | 32 GiB   | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.2xlarge**  | 8     | 64 GiB   | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.4xlarge**  | 16    | 128 GiB  | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.8xlarge**  | 32    | 256 GiB  | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.12xlarge** | 48    | 384 GiB  | 4    | 178 GiB (4×44)  ([AWS Documentation][1]) |
| **g6e.16xlarge** | 64    | 512 GiB  | 1    | 44 GiB (1×44)   ([AWS Documentation][1]) |
| **g6e.24xlarge** | 96    | 768 GiB  | 4    | 178 GiB (4×44)  ([AWS Documentation][1]) |
| **g6e.48xlarge** | 192   | 1536 GiB | 8    | 357 GiB (8×44)  ([AWS Documentation][1]) |

---
# Local NVMe based EC2s for hosting vector dbs like qdrant, arrangodb as statefulsets 
| Instance     | vCPU / RAM       | **C8g On-Demand** (USD/hr) | **C8g On-Demand** (₹/hr) | **C8g Spot** (USD/hr)   | **C8g Spot** (₹/hr) | **C8gd On-Demand** (USD/hr) | **C8gd On-Demand** (₹/hr) | **C8gd Spot** (USD/hr)  | **C8gd Spot** (₹/hr) |
| ------------ | ---------------- | -------------------------- | ------------------------ | ----------------------- | ------------------- | --------------------------- | ------------------------- | ----------------------- | -------------------- |
| **medium**   | 1 vCPU/2 GiB     | \$0.044 ([Vantage][1])     | ₹3.7                     | \$0.013 ([Vantage][1])  | ₹1.1                | \$0.054 ([Vantage][2])      | ₹4.5                      | \$0.012 ([Vantage][2])  | ₹1.0                 |
| **xlarge**   | 4 vCPU/8 GiB     | \$0.702 ([Vantage][3])     | ₹58.2                    | \$0.238 ([Vantage][3])  | ₹19.8               | \$0.196 ([Vantage][4])      | ₹16.3                     | \$0.080 ([Vantage][4])  | ₹6.6                 |
| **2xlarge**  | 8 vCPU/16 GiB    | \$0.319 ([Vantage][5])     | ₹26.5                    | \$0.128 ([Vantage][5])  | ₹10.6               | —                           | —                         | —                       | —                    |
| **4xlarge**  | 16 vCPU/32 GiB   | \$0.702 ([Vantage][3])     | ₹58.2                    | \$0.238 ([Vantage][3])  | ₹19.8               | \$0.784 ([Vantage][6])      | ₹65.1                     | \$0.298 ([Vantage][6])  | ₹24.7                |
| **8xlarge**  | 32 vCPU/64 GiB   | \$1.276 ([Vantage][7])     | ₹105.9                   | \$0.522 ([Vantage][7])  | ₹43.3               | \$1.720 ([Vantage][8])      | ₹142.8                    | \$— (spot NA)           | —                    |
| **16xlarge** | 64 vCPU/128 GiB  | \$2.808 ([Vantage][9])     | ₹233.1                   | \$0.731 ([Vantage][9])  | ₹60.7               | \$3.135 ([Vantage][10])     | ₹259.9                    | \$0.796 ([Vantage][10]) | ₹66.1                |
| **24xlarge** | 96 vCPU/192 GiB  | \$4.211 ([Vantage][8])     | ₹349.3                   | \$1.065 ([Vantage][8])  | ₹88.4               | \$5.173 ([Vantage][8])      | ₹429.4                    | \$1.476 ([Vantage][8])  | ₹122.5               |
| **48xlarge** | 192 vCPU/384 GiB | \$7.657 ([Vantage][11])    | ₹635.6                   | \$1.979 ([Vantage][11]) | ₹164.3              | \$9.406 ([Vantage][12])     | ₹780.8                    | \$3.323 ([Vantage][12]) | ₹275.6               |

---

### Notes

* “—” indicates data not available or not listed for that size.
* Spot prices fluctuate; these are **minimums** observed at time of lookup.
* INR values are approximate (USD × 83).
* NVMe-equipped C8gd variants include large local SSD at no extra configuration cost; C8g has no NVMe.
* Source pricing from Instances.Vantage.sh snapshots as of early August 2025.

[1]: https://instances.vantage.sh/aws/ec2/c8g.medium
[2]: https://instances.vantage.sh/aws/ec2/c8gd.medium
[3]: https://instances.vantage.sh/aws/ec2/c8g.4xlarge
[4]: https://instances.vantage.sh/aws/ec2/c8gd.xlarge?cost_duration=monthly&os=linux&region=us-east-1&reserved_term=Standard.noUpfront
[5]: https://instances.vantage.sh/aws/ec2/c8g.2xlarge
[6]: https://instances.vantage.sh/aws/ec2/c8gd.4xlarge
[7]: https://instances.vantage.sh/aws/ec2/c8g.8xlarge
[8]: https://instances.vantage.sh/aws/ec2/c8gd.24xlarge
[9]: https://instances.vantage.sh/aws/ec2/c8g.16xlarge
[10]: https://instances.vantage.sh/aws/ec2/c8gd.16xlarge
[11]: https://instances.vantage.sh/aws/ec2/c8g.48xlarge
[12]: https://instances.vantage.sh/aws/ec2/c8gd.48xlarge

</details>

# EXAMPLES

### Qdrant Payload Object (Vector Database Point)

* **Embedding:**
  Dense numeric vector representing the chunk’s semantic meaning for similarity search.

* **Metadata:**
  Includes `chunk_id`, `document_id`, `chunk_type`, `text` snippet, source info, timestamp, tags (e.g., `"occupation"`, `"tailor"`), and linked entity IDs (e.g., Wikidata QIDs).

* **Purpose:**
  Enables combined vector similarity search with metadata filtering for precise and flexible retrieval.

* **Use Case:**
  Helps find occupation data like “tailor” despite synonyms or multilingual terms, improving over simple keyword search.

* **Integration:**
  Bridges unstructured semantic data and structured metadata for hybrid search in NCO datasets.


```sh
{
  "chunk_id": "e4f72d3a9c5b4f17_5",                           // Unique chunk ID: <document_hash>_<chunk_index> (1-based)
  "document_id": "e4f72d3a9c5b4f17",                         // Unique document ID (128-bit hash of file path + size)
  "chunk_type": "page",                                       // Type of content: "paragraph", "heading", "section", "table", "page", etc.

  "text": "### National Classification of Occupations (NCO) – Tailors\n\nThe NCO code for Tailor (General) is **7531.0100**. This occupation falls under the broader category **Tailors, Dressmakers, Furriers, and Hatters**. Tailors are responsible for measuring, cutting, and sewing garments to fit clients' specifications. \n\nOther related occupations include:\n- Tailor, Tent (Machine) — NCO code 7534.0100\n- Dressmaker — closely associated within the same category\n\nThe classification covers various regional and linguistic variants of the term \"tailor,\" including seamstress, garment maker, and costume maker, reflecting the multilingual occupational landscape in India. This ensures comprehensive indexing despite naming variations across states and languages.\n\nSkills and tools commonly associated with tailoring include manual sewing, machine operation, pattern drafting, and fabric selection.",  // Final parsed content (Markdown) , 

  "embedding": [0.0234, -0.1457, 0.3782, 0.0923, -0.0567,..]  // Vector embedding (array of floats); null if not yet computed

  "source": {
    "file_type": "application/pdf",                           // MIME type preferred (e.g., "application/pdf", "text/html", "audio/mpeg")
    "source_path": "s3://mospi-data/data/raw/nco_2015_occupations.pdf",  // Full s3 path to original source
    "page_number": 5,                                         // For paged formats like PDF/ePub; null otherwise
    "time": [null, null],                                     // [start_time, end_time] in seconds for audio/video; nulls otherwise
    "line_range": null,                                       // For plain/tabular text: [start_line, end_line]; null otherwise
    "bbox": null                                              // For visual formats: [x0, y0, x1, y1] in pixel coordinates; null otherwise
  },

  "graph": {
    "graph_node_id": "e4f72d3a9c5b4f17_5",                   // Same as chunk_id (recommended)
    "parent_id": "e4f72d3a9c5b4f17_page5",                   // Parent node ID (e.g., page, section, table)
    "previous_id": "e4f72d3a9c5b4f17_4",                     // Optional: previous chunk
    "next_id": "e4f72d3a9c5b4f17_6"                          // Optional: next chunk
  },

  "metadata": {
    "timestamp": "2025-08-03T12:15:27Z",                     // UTC ISO timestamp of chunk creation/parsing
    "tags": ["occupation", "tailor", "NCO", "classification"],  // High-level content tags (semantic or manual)
    "layout_tags": ["heading", "paragraph", "list"]           // Structural tags (e.g., "heading", "table", etc.)
  },

  "entities": [
    "Q7531",                                                  // Example Wikidata ID for Tailor occupation (hypothetical)
    "Q1251134"                                                // Wikidata ID for National Classification of Occupations (example)
  ],                                                         // Optional: Linked entity IDs (Wikidata, etc.) or null if not yet computed

  "triplets": [                                              // Extracted subject-predicate-object relations
    {
      "subject": "Tailor (General)", "predicate": "hasNCOCode", "object": "7531.0100"
    },
    {
      "subject": "Tailor (General)","predicate": "belongsToCategory", "object": "Tailors, Dressmakers, Furriers, and Hatters"
    },
    {
      "subject": "Tailor, Tent (Machine)", "predicate": "hasNCOCode", "object": "7534.0100"
    },
    {
      "subject": "Tailor (General)", "predicate": "hasSynonyms", "object": "Seamstress, Dressmaker, Garment Maker, Costume Maker"
    }
  ]
}
```


### Corresponding ArangoDB Graph Data Model Representing NCO Tailor Chunk and Linked Entities
* The **chunk document** holds the main text and metadata with embedding vector stored for similarity search.
* **Entities** are individual occupational concepts or classifications, stored separately to normalize data.
* **Edges** explicitly represent relations such as "mentions", "classifiedBy", or attribute predicates.
* This graph structure enables **multi-hop queries** and semantic traversals beyond keyword matching.

---

#### 1. Chunk Document (`chunks` collection)

```json
{
  "_key": "e4f72d3a9c5b4f17_5",
  "chunk_id": "e4f72d3a9c5b4f17_5",
  "document_id": "e4f72d3a9c5b4f17",
  "chunk_type": "page",
  "text": "### National Classification of Occupations (NCO) – Tailors\n\nThe NCO code for Tailor (General) is **7531.0100**. This occupation falls under the broader category **Tailors, Dressmakers, Furriers, and Hatters**. Tailors are responsible for measuring, cutting, and sewing garments to fit clients' specifications.\n\nOther related occupations include:\n- Tailor, Tent (Machine) — NCO code 7534.0100\n- Dressmaker — closely associated within the same category\n\nThe classification covers various regional and linguistic variants of the term \"tailor,\" including seamstress, garment maker, and costume maker.\n\nSkills and tools commonly associated include manual sewing, machine operation, pattern drafting, and fabric selection.",
  "embedding": [0.0234, -0.1457, 0.3782, 0.0923, -0.0567, ...],
  "source": {
    "file_type": "application/pdf",
    "source_path": "s3://mospi-data/data/raw/nco_2025_occupations.pdf",
    "page_number": 5,
    "time": [null, null],
    "line_range": null,
    "bbox": null
  },
  "metadata": {
    "timestamp": "2025-08-03T12:15:27Z",
    "tags": ["occupation", "tailor", "NCO", "classification"],
    "layout_tags": ["heading", "paragraph", "list"]
  }
}
```

---

#### 2. Entity Documents (`entities` collection)

```json
{
  "_key": "Q7531",
  "wikidata_id": "Q7531",
  "label": "Tailor (General)",
  "aliases": ["Seamstress", "Dressmaker", "Garment Maker", "Costume Maker"]
}
```

```json
{
  "_key": "Q1251134",
  "wikidata_id": "Q1251134",
  "label": "National Classification of Occupations",
  "description": "Standard classification of occupations in India"
}
```

#### 3. Edges Linking Chunk to Entities (`chunkEntityEdges` edge collection)

```json
{
  "_from": "chunks/e4f72d3a9c5b4f17_5",
  "_to": "entities/Q7531",
  "relation": "mentions"
}
```

```json
{
  "_from": "chunks/e4f72d3a9c5b4f17_5",
  "_to": "entities/Q1251134",
  "relation": "sourceClassification"
}
```

---

#### 4. Entity Relations (`entityRelationEdges` edge collection)

```json
{
  "_from": "entities/Q7531",
  "_to": "entities/Q1251134",
  "predicate": "classifiedBy"
}
```

```json
{
  "_from": "entities/Q7531",
  "_to": null,
  "predicate": "hasNCOCode",
  "object": "7531.0100"
}
```

---
