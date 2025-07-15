### **Prerequisite:**

A full Linux setup is required (do **not** use Docker Desktop, WSL,devcontainers).

---

## **One-time installation prerequisites**

| Windows                                                                                                              | macOS/Linux                                                        |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [Visual Studio Code](https://code.visualstudio.com/) *(required)*                                                    | [Visual Studio Code](https://code.visualstudio.com/) *(required)*  |
| [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) | *(not required)*                                                   |
| [Git](https://git-scm.com/downloads)                                                                                 | [Git](https://git-scm.com/downloads)                               |
| [Vagrant 2.4.3](https://developer.hashicorp.com/vagrant/downloads)                                                   | [Vagrant 2.4.3](https://developer.hashicorp.com/vagrant/downloads) |
| [VirtualBox 7.0.14](https://download.virtualbox.org/virtualbox/7.0.14/)                                                              | [VirtualBox 7.0.14](https://download.virtualbox.org/virtualbox/7.0.14/)            |

> **Note:** If using windows search windows features and **Turn off Hyper‑V , Windows Hypervisor Platform** and delete **Windows Subsystem for Linux (WSL2)** if possible 


---

## **Restart your system and get started**

> Open a **Git Bash** terminal and run the following command. The first run will take longer(20-30 minutes) as the Ubuntu Jammy VM will be downloaded. 

```bash
rm -rf $HOME/RAG8s && mkdir $HOME/RAG8s && cd $HOME/RAG8s && git config --global core.autocrlf false && git clone https://github.com/Athithya-Sakthivel/RAG8s.git && cd RAG8s && bash utils/ssh.sh
```
> The default configs are RAM = 11GB, vcpus=8 and no gpu , override it in the Vagrantfile if needed 
---

## **Important: VM Lifecycle**

 ### **After a system reboot**, the VM will be shut down. Always start it manually before doing ssh.

  * Open VirtualBox → Right-click the VM → **Start → Headless Start and wait atleast 45-60 seconds before opening vscode**

  ![Start the VM](.vscode/Start_the_VM.png)



### Login to your github account
```
make login
```

### Install the neccessary cli tools, py packages 
```
make install
```











In that exact message, remove combsum and bge. ////////////Here’s your upgraded SOTA stack, now including SGLang, RAGApp, DSPy, ValKeye, and SPLADE++, placed in the sections where they deliver maximal impact:


---

🔷 1. Parsing & Preprocessing

Component	Tool(s) / Notes

Format Detection	python-magic, mimetypes
Language Detection	fastText (within ReLiK)
PDF Parser	Ferrulas (Rust, OCR & layout-aware)
DOCX, PPTX, HTML	parser_core (Rust-based unified parser)
MP3 Audio	semantic-codec + faster-whisper (multilingual ASR)


---

🔶 2. Chunking

Strategy	Tool(s)	Notes

Model-less	llm-utils, text-splitter	Scalable, efficient chunkers
Meta-aware	parser_core, Ferrulas	Emit chunks + entities aligned with your schema


---

🧠 3. Triplet Extraction / Graph Construction

Component	Tool(s)	Notes

Primary Extractor	ReLiK	Fast, structured triplets
Fallback	mREBEL	Multilingual / nuanced extraction


---

📚 4. Embedding & Reranking

Task	Model / Tool	Notes

Embedding	Alibaba-NLP/gte-multilingual-base	Small, multilingual, highly performant
Reranking	GTE (or BGE)	Efficient, accurate
Prompt Pipeline Optimization	DSPy	Declarative LLM-pipeline programming & auto-tuning


---

🧭 5. Retrieval (Hybrid)

Modality	Tool(s)	Description

Dense	SimLM (or Contriever)	Single-vector, fast + accurate
Sparse	SPLADE++	Learned sparse representations with inverted-index speed
Fusion	RRF / CombSUM	Combine dense + sparse scores
Vector Store	Qdrant	Stores high-dim embeddings + metadata
Graph Store	ArangoDB	Stores entity-triplet graphs
Graph Query	AQL + RGL	Structured reasoning logic


---

🚀 6. Indexing, Caching & Orchestration

Component	Tool(s) / Notes

DAG Engine	Ray Workflows (distributed indexing)
Structured Data	Ray Data
Chunk Routing	parser_core, Ray Tasks, fastText
Cache / Memory	ValKeye – high-performance key/value store for embeddings, session and agent memory caching


---

🧠 7. Inference Layer

Component	Tool(s) / Notes

Inference API	Ray Serve + FastAPI + gRPC
Runtime Framework	SGLang – ultra-fast LLM/VLM serving with RadixAttention, speculative decoding, LoRA batching
Agentic RAG	LangGraph + ValKeye (multi-turn memory retention)
Prompt Pipelines	DSPy (orchestration & auto-tuning of retrieval → reranker → LLM stages)
Image Adapter	CoCa
LLMs	OpenAI GPT-4, Claude, Mistral, or self-hosted via vLLM / SGLang


---

💻 8. Frontend & Developer UX

Component	Tool(s) / Notes

End-User Chat UI	RAGApp – no-code, enterprise-grade RAG chat + file viewer
Admin / Debug Console	Retrieval trace inspector + prompt sleeve for repro & tuning


---

🌐 9. DevOps / Infrastructure

Component	Tool(s) / Notes

Kubernetes	AWS EKS
Autoscaling	Karpenter
Ray on K8s	KubeRay
Networking	Traefik + Cloudflare
Storage	EFS (models), EBS (stateful stores)
AMIs	Custom pre-baked images for cold-start performance


---

🔐 10. Security / Monitoring / Auth

Category	Tool(s) / Notes

Observability	Prometheus + Grafana + Helicone
Auth / RBAC	OIDC / API Gateway + Keycloak
Tracing / Logs	Helicone + RAGAI-Catalyst


---

 This is your final, unified SOTA RAG platform—now truly multilingual, multimodal, LLM-efficient, highly scalable, cloud-native, observability-first, and production ready. Let me know if you’d like sample code, deployment scripts, or an architecture diagram!

///////////

