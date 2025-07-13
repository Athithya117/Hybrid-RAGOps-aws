# Indexing_pipeline for unstructured data using the SOTA tools and stratergies
## 1. Overview Diagram

```
    INPUT FILE (.pdf / .html / .docx / .pptx / .mp3 / .png / .jpg / .md)
                           ↓
       DETECTION: OCR-heavy? Multilingual? Format type?
                           ↓
     ┌────────────────────────────────────────────────────────┐
     │ ROUTING ENGINE                                         │
     │                                                        │
     │ if OCR-heavy and multilingual → paddleocr + LayoutXLM  │
     │ if OCR-heavy and monolingual → paddleocr + LayoutLMv3  │
     │ if not OCR-heavy but multilingual → LayoutXLM          │
     │ if clean layout and monolingual → docling              │
     │ For HTML → trafilatura (+ docling fallback)            │
     │ For audio → semantic-codec + faster-whisper            │
     │ For docx/pptx → python-docx/python-pptx (+ docling)    │
     └────────────────────────────────────────────────────────┘
                           ↓
           PARSED TEXT + METADATA + TRIPLETS + ENTITIES
                           ↓
               NORMALIZED STRUCTURED SCHEMA
                           ↓
    → Qdrant (vector index)           → ArangoDB (graph store)
```

---

## 2. Detection Phase

### Goals:

* Avoid over-parsing by **routing efficiently**
* Determine:

  * File format
  * Whether OCR is needed
  * Whether the file is multilingual

### Tools:

| Task                          | Tool                                                   |
| ----------------------------- | ------------------------------------------------------ |
| File type detection           | `python-magic`                                         |
| Text extraction (lightweight) |`PyMuPDF`, `python-docx`, `python-pptx`, `BeautifulSoup`|
| OCR heaviness                 | PyMuPDF layout check and other types like .html, etc are always text-based by default. |
| Multilingual detection        | `fastText` on 1-2 pages of extracted text          |

**Logic**:

* Extract the **first 1–2 pages** of text
* Use `fastText` to detect multilingual text
* If few tokens or no extractable text → likely OCR-heavy
* Fallback to deeper inspection if confidence is low

---

## 3. Parsing Phase

### Routing Logic:

```python
if format in ["pdf", "png", "jpg"]:
    if is_ocr_heavy:
        if is_multilingual:
            parser = "paddleocr + LayoutXLM"
        else:
            parser = "paddleocr + LayoutLMv3"
    else:
        if is_multilingual:
            parser = "LayoutXLM (text)"
        else:
            parser = "docling"

elif format == "html":
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    
    if is_multilingual(text):
        parser = "docling"
    else:
        parser = "bs4 + docling"

elif format in ["docx", "pptx"]:
    raw_text = extract_text_docx_pptx(file)
    if is_multilingual(raw_text):
        parser = "docling"
    else:
        parser = "python-docx/python-pptx + docling"

elif format == "mp3":
    parser = "semantic-codec + faster-whisper"
elif format == "md":
    parser = "markdown-it + docling"
else:
    parser = "docling"
```

---

## 4. Parsing Tools Summary

| Format          | Parser                              | Notes                         |
| --------------- | ----------------------------------- | ----------------------------- |
| PDF             | PyMuPDF, paddleocr, LayoutLMv3/XLM  | Multilingual + layout OCR     |
| Image (PNG/JPG) | paddleocr + LayoutXLM               | Visual OCR pipeline           |
| DOCX/PPTX       | python-docx / python-pptx / docling | Structured office formats     |
| HTML            | trafilatura + docling               | Cleaned HTML with fallback    |
| MP3             | semantic-codec + faster-whisper     | Efficient multilingual ASR    |
| Markdown        | markdown-it + docling               | Simple + consistent structure |

---

## 5. Normalized Output Schema 


```json
{
  "id": "chunk_abc123",
  "document_id": "doc_001",
  "chunk_id": "chunk_3",
  "text": "John Doe purchased Product X for $500 on July 1, 2024.",
  "embedding": [],
  "source": {
    "path": "s3://bucket/invoice.pdf",
    "hash": "sha256:abc123...",
    "file_type": "pdf",
    "page_number": 4
  },
  "metadata": {
    "language": "en",
    "is_multilingual": false,
    "is_ocr": true,
    "chunk_type": "paragraph",
    "timestamp": "2024-07-01T00:00:00Z",
    "tags": []
  },
  "entities": [],
  "triplets": []
}

```

### **Qdrant Payload (Final JSON Format)**

This is the full, enriched payload per vector chunk to be indexed in Qdrant:

```json
{
  "id": "chunk_abc123",
  "embedding": [0.123, 0.456, 0.789, 0.321, 0.654],
  "payload": {
    "document_id": "doc_001",
    "chunk_id": "chunk_3",
    "chunk_index": 2,
    "text": "John Doe purchased Product X for $500 on July 1, 2024.",
    "parser": "paddleocr + layoutLM",
    "pipeline_stage": "embedded",
    "source": {
      "path": "s3://bucket/invoice.pdf",
      "hash": "sha256:abc123...",
      "file_type": "pdf",
      "page_number": 4
    },
    "metadata": {
      "language": "en",
      "is_multilingual": false,
      "is_ocr": true,
      "chunk_type": "paragraph",
      "timestamp": "2024-07-01T00:00:00Z",
      "tags": ["purchase", "finance", "invoice"]
    },
    "entities": [
      { "name": "John Doe", "type": "Person" },
      { "name": "Product X", "type": "Product" },
      { "name": "$500", "type": "Price" }
    ],
    "triplets": [
      ["John Doe", "purchased", "Product X"],
      ["Product X", "costs", "$500"],
      ["Purchase", "occurred_on", "2024-07-01"]
    ],
    "confidence": {
      "embedding": 0.98,
      "ocr": 0.95,
      "parser": 0.93
    }
  }
}
```

---

### 🔷 **ArangoDB Schema for Graph RAG**

You’ll model this as a **document collection** (nodes) + **edge collection** (triplets).

#### 1. **Document Collection (e.g., `Chunks`)**

```json
{
  "_key": "chunk_abc123",
  "document_id": "doc_001",
  "chunk_id": "chunk_3",
  "chunk_index": 2,
  "text": "John Doe purchased Product X for $500 on July 1, 2024.",
  "parser": "paddleocr + layoutLM",
  "pipeline_stage": "embedded",
  "source_path": "s3://bucket/invoice.pdf",
  "source_hash": "sha256:abc123...",
  "file_type": "pdf",
  "page_number": 4,
  "language": "en",
  "is_multilingual": false,
  "is_ocr": true,
  "chunk_type": "paragraph",
  "timestamp": "2024-07-01T00:00:00Z",
  "tags": ["purchase", "finance", "invoice"],
  "entities": [
    { "name": "John Doe", "type": "Person" },
    { "name": "Product X", "type": "Product" },
    { "name": "$500", "type": "Price" }
  ],
  "confidence": {
    "embedding": 0.98,
    "ocr": 0.95,
    "parser": 0.93
  }
}
```

#### 2. **Edge Collection (e.g., `KnowledgeTriplets`)**

Each triplet becomes a directed edge between entities.

```json
{
  "_from": "Entities/John_Doe",
  "_to": "Entities/Product_X",
  "predicate": "purchased",
  "source_chunk": "Chunks/chunk_abc123",
  "timestamp": "2024-07-01T00:00:00Z",
  "doc_id": "doc_001"
}
```

Repeat for other triplets like `["Product X", "costs", "$500"]`.

#### 3. **Entity Collection (Optional)**

```json
{
  "_key": "John_Doe",
  "name": "John Doe",
  "type": "Person"
}
```

---

### Summary

| Component           | Purpose           | Description                                         |
| ------------------- | ----------------- | --------------------------------------------------- |
| **Qdrant**          | Vector similarity | Stores embedding + full payload.                    |
| **ArangoDB Chunks** | Document nodes    | Stores parsed chunks as nodes.                      |
| **ArangoDB Edges**  | Knowledge graph   | Triplets as semantic links.                         |
| **Confidence**      | Auditable scoring | Optional but useful for fallback/routing decisions. |



.
├── Makefile                             # Automation of build/run commands (e.g., testing, cleaning)
├── README.md                            # Documentation of the project, pipeline overview, and usage
├── data                                 # Storage for various stages of data during the indexing pipeline
│   ├── chunked                          # Contains normalized, structured text chunks
│   ├── parsed                           # Parsed raw text before chunking or embedding
│   └── raw                              # Original unprocessed input files (PDFs, DOCXs, etc.)
├── efs                                  # Placeholder for Elastic File System or model storage
│   └── models                           # Houses model directories used in parsing/embedding
│       ├── bge                          # BAAI General Embedding (BGE) models for vectorization
│       ├── faster-whisper               # Fast ASR model used for MP3 transcription
│       ├── llama                        # LLM directory, possibly for downstream QA or RAG
│       └── relik                        # Custom/local model or pipeline logic (e.g., for re-ranking)
├── export.sh                            # Shell script to export data or push to remote stores
├── indexing_pipelines                   # Main indexing code and requirements
│   ├── requirements.txt                 # Dependencies for the indexing pipeline
│   └── unstructured                     # Parsing modules for unstructured documents
│       ├── __init__.py                  # Marks this directory as a Python package
│       └── parsing                      # Contains individual parsers for each input format
│           ├── __init__.py              # Initializes the parsing submodule
│           ├── docx_parser.py           # Extracts and normalizes content from .docx files
│           ├── fallback_pdf_parser.py   # Fallback logic when primary PDF parsing fails
│           ├── html_parser.py           # Extracts and cleans text from HTML files using BS4/trafilatura
│           ├── image_parser.py          # OCR + layout parsing for PNG, JPG using PaddleOCR + LayoutXLM
│           ├── mp3_parser.py            # Audio transcription using semantic-codec + faster-whisper
│           ├── multilingual_pdf_parser.py  # Handles multilingual PDFs via LayoutXLM
│           ├── ocr_multilingual_pdf_parser.py  # OCR + multilingual layout parsing pipeline
│           ├── ocr_pdf_parser.py        # Handles OCR-only (monolingual) PDFs via LayoutLMv3
│           ├── pptx_parser.py           # Extracts text + metadata from PowerPoint files (.pptx)
│           ├── router.py                # 🚦 Main routing engine:
│                                        #    - Detects format, OCR needs, multilingual status
│                                        #    - Dynamically assigns parsing strategy
│                                        #    - Uses `ray.data` for scalable distributed preprocessing
│                                        #    - Uses `hashlib` to hash input files for deduplication/versioning
│           └── txt_parser.py            # Simple line or paragraph extraction for .txt files
└── utils                                # Helper scripts for environment setup and data ingestion
    ├── create_s3.py                     # Creates/configures an S3 bucket (used for file input/output)
    ├── sync_data_with_s3.py             # Syncs raw/parsed/chunked files between local and S3
    └── web_scraper.py                   # Extracts data from webpages for HTML ingestion
