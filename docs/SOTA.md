

## Research Support

* NVIDIA (mid-2025): Page-level chunking is baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/




elastic/multilingual-e5-small-optimized



1. shavarani/SpEL(base only) for SOTA english 

2. **mReFinED** for fastest , supports 9 languages: Arabic, English, Spanish, German, Farsi, Japanese, Tamil, Turkish, and possibly Lithuanian.

3. fallback to wannaphong/BELA  from meta that supports
```sh
af  Afrikaans                 am  Amharic                   ar  Arabic
arg Aragonese                ast Asturian                 az  Azerbaijani
bar Bavarian                 be  Belarusian               bg  Bulgarian
bn  Bengali                  br  Breton                   bs  Bosnian
ca  Catalan                  ckb Central Kurdish         cs  Czech
cy  Welsh                    da  Danish                   de  German
el  Greek                    en  English                  eo  Esperanto
es  Spanish                  et  Estonian                 eu  Basque
fa  Persian/Farsi            fi  Finnish                  fr  French
fy  Western Frisian          ga  Irish                    gan Gan Chinese
gl  Galician                 gu  Gujarati                 he  Hebrew
hi  Hindi                    hr  Croatian                 hu  Hungarian
hy  Armenian                 ia  Interlingua              id  Indonesian
is  Icelandic                it  Italian                  ja  Japanese
jv  Javanese                 ka  Georgian                 kk  Kazakh
kn  Kannada                  ko  Korean                   ku  Kurdish
la  Latin                    lb  Luxembourgish            lt  Lithuanian
lv  Latvian                  mk  Macedonian               ml  Malayalam
mn  Mongolian                mr  Marathi                  ms  Malay
my  Burmese                  nds Low German              ne  Nepali
nl  Dutch                    nn  Norwegian (Nynorsk)      no  Norwegian
oc  Occitan                  pl  Polish                   pt  Portuguese
ro  Romanian                 ru  Russian                  scn Sicilian
sco Scots                   sh  Serbo-Croatian           si  Sinhala
sk  Slovak                   sl  Slovenian                sq  Albanian
sr  Serbian                  sv  Swedish                  sw  Swahili
ta  Tamil                    te  Telugu                   th  Thai
tl  Tagalog                  tr  Turkish                  tt  Tatar
uk  Ukrainian                ur  Urdu                     uz  Uzbek
vi  Vietnamese               war Waray                    wuu Wu Chinese
yi  Yiddish                  zh  Modern Standard Chinese  zh_classical Classical Chinese
zh_yue Cantonese
```


For ultra-high throughput, consider batching requests, using ONNX optimized pipelines, or running quantized inference — these can further boost speed while retaining accuracy


Run this model using ONNX or GGUF quantization via optimum or ctransformers to get sub-10 ms latency per query even on budget CPUs



These two together form a compact but accurate pipeline suitable for **graph+vector-based RAG** at production scale.






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

