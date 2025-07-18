


### **Qdrant Payload (Final JSONL Format)**

This is the full, enriched payload per vector chunk to be indexed in Qdrant:

```sh
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
      "timestamp": "2025-07-01T00:00:00Z",
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
      ["Purchase", "occurred_on", "2025-07-01"]
    ]
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

