# **Platform Model Architecture Overview**

This document provides a clear, production-grade explanation of the four core model types used in the platform’s retrieval-augmented generation (RAG) stack:

1. **LLM (reasoning/generation)**
2. **Dense Embedding Model**
3. **Sparse Embedding Model**
4. **Reranker**

Each component has a distinct role, deployment mode, and cost/performance tradeoff. Together they form a fast, scalable, and cost-efficient retrieval and answer-generation pipeline.

---
# Architecture Summary

| Layer | Component | Deployment | Reason |
|-------|-----------|------------|--------|
| **Dense Embeddings** | BGE-small | Self-hosted CPU/GPU | Cheap, fast, ideal for batch indexing + queries. |
| **Sparse Embeddings** | miniCOIL | Self-hosted CPU/GPU | Complements dense embedding; improves recall. |
| **Reranker** | MiniLM cross-encoder | Self-hosted CPU/GPU | Best ranking accuracy for minimal cost. |
| **LLM** | Groq models | Managed | Highest performance and reliability, no GPU ops. |

---

## 1. LLM (Reasoning + Generation)
**Default:** `llama-3.1-8b-instant` . More: https://console.groq.com/docs/models

### Purpose
Handles synthesis, reasoning, summarization, and final answer construction after retrieval.  
The LLM does *not* store knowledge for RAG queries; it reacts to the documents retrieved by Qdrant.

### Why we choose managed LLMs
- Lower operational burden  
- Highest tokens-per-second throughput (Groq)  
- Easier versioning and safety controls  
- No need to maintain GPU fleet

The rest of the retrieval stack is self-hosted; LLM inference is intentionally offloaded.

---

## 2. Dense Embedding Model  
**Default:** `BAAI/bge-small-en-v1.5` (384-dim). More: https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models

### Purpose
Generates dense embeddings for semantic similarity.  
Used for both:
- offline batch indexing  
- real-time query embedding

### Deployment Modes
| Mode | Notes |
|------|-------|
| **Self-hosted (fastembed CPU)** | Extremely fast, no GPU required. Perfect for production indexing. |
| **Hosted API** | More expensive and slower for high-volume ingestion. |

### Why we choose self-hosted dense embeddings
- FastEmbed CPU models are optimized C++ graphs (SIMD), giving near-GPU speed without hardware cost.
- Batch indexing pipelines embed millions of chunks cheaply.
- Predictable and very stable latency.

Dense embeddings form the primary semantic search layer.

---

## 3. Sparse Embedding Model  
**Default:** `Qdrant/minicoil-v1` More: https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-sparse-text-embedding-models

### Purpose
Gives lexical precision and token-level recall.   (BM25-style neural sparse).
Pairs with dense vectors for **hybrid search**:
- Dense catches meaning  
- Sparse catches exact keyword overlaps

### Deployment Modes
| Mode | Notes |
|------|-------|
| **Self-hosted (fastembed sparse)** | Lightweight and fast. Typically CPU-only. |
| **Hosted API** | Rarely needed and expensive for high-volume. |

### Why we include sparse embeddings
Hybrid search reliably outperforms pure dense search:
- better for technical docs  
- better for code  
- prevents “semantic drift”  
- improves determinism and reduces hallucinations

Sparse is optional but strongly recommended when documents are technical or long-form.

---

## 4. Reranker  
**Default:** `Xenova/ms-marco-MiniLM-L-6-v2`. More: https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-rerank-cross-encoder-models

### Purpose
Re-scores the top-k retrieved documents using cross-encoder ranking.  
Significantly improves final accuracy by understanding full query–document pair interactions.

### Deployment Modes
| Mode | Notes |
|------|-------|
| **Self-hosted (fastembed cross-encoder)** | Low cost, good latency, runs on CPU. |
| **Managed API** | Rarely economical; slower; unnecessary. |

### Why a reranker is used
- Dense+sparse retrieve “candidate” docs  
- Reranker identifies the *truly* most relevant ones  
- Combined system approaches state-of-the-art retrieval quality

This also reduces prompt size and improves LLM output correctness.

---

# Why This Setup

### Cost efficiency
- Embedding and reranking are CPU-friendly and cheap to self-host.
- LLM compute is the critical part and is delegated to high-performance managed providers (Groq).

### Reliability
- Deterministic search pipeline (dense → sparse → rerank).
- No dependency on external embedding APIs for bulk ingestion.

### Scalability
- Qdrant cluster + batch indexing pipeline handles millions of documents.
- Embedding service autoscaling is simple due to stateless design.

### Operational simplicity
- Only lightweight embedding and reranker services are self-hosted.
- LLM infrastructure is external, reducing GPU complexity.

---