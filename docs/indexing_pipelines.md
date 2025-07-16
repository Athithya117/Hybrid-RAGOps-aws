
## Research Support

* NVIDIA (mid-2025): Page-level chunking is baseline best https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/





### 📘 Information Extraction Models (ReLiK vs. REBEL)

| Model                        | Params (Retriever + Reader) | Entity Linking (Micro‑F1) | Relation Extraction (Micro‑F1)                                             |
| ---------------------------- | --------------------------- | ------------------------- | -------------------------------------------------------------------------- |
| **relik-ie/relik-cie-small** | 33 M + 141 M = **174 M**    | **94.4 %**                | **75.8 %**                                                                 |
| **relik-ie/relik-cie-large** | 33 M + 434 M = **467 M**    | **95.0 %**                | **78.1 %**                                                                 |
| **Babelscape/rebel-large**   | \~**406 M** (BART-based)    | N/A (no EL)               | **93.4 %** (NYT RE) <br> **76.65 %** (CoNLL04 RE) <br> **82.2 %** (ADE RE) |

---

### 📗 Embedding Models (Alibaba GTE/mGTE)

| Model                     | Params  | Embedding Dim | Max Seq Length | MTEB Score | LoCo Score | COIR Score |
| ------------------------- | ------- | ------------- | -------------- | ---------- | ---------- | ---------- |
| **gte-modernbert-base**   | 149 M   | 768           | 8192 tokens    | **64.38**  | **87.57**  | **79.31**  |
| **gte-base-en-v1.5**      | 137 M   | 768           | 8192 tokens    | **64.11**  | **87.44**  | —          |
| **gte-multilingual-base** | \~300 M | 768           | 8192 tokens    | —          | —          | —          |
| **gte-large-en-v1.5**     | 434 M   | 1024          | 8192 tokens    | **65.39**  | **86.71**  | —          |

---

### 🔍 Final Suggestion (Prod-Ready)

*  **relik-ie/relik-cie-small**: Best **entity + relation** extraction model for scale with high accuracy.
*  **gte-modernbert-base**: Best tradeoff for **embedding quality**, long context, and low latency.

These two together form a compact but accurate pipeline suitable for **graph+vector-based RAG** at production scale.




