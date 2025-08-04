



```

export GHCR_USER=
export HF_TOKEN=hf_
export GHCR_PAT=ghp_
export LOCAL_QWEN3_MAX_WORKERS=1
export TAG=default

export HF_TOKEN=hf_
export GHCR_PAT=ghp_
export GITHUB_USER=
export LOCAL_QWEN3_MAX_WORKERS=1
export TAG=v1
export EMAIL=example@gmail.com



```



```
                        ┌───────────────────────────────┐
                        │      Your Application / UI     │
                        │  (e.g., React, LangChain, CLI) │
                        └────────────┬────────────────────┘
                                     │ REST API call
                                     ▼
                         http://localhost:8000/llm/generate
                                     │
                            ┌────────▼────────┐
                            │   Ray Serve     │  ← starts inference service
                            │   Controller    │
                            └────────┬────────┘
                                     │ deploys
                                     ▼
                       ┌────────────────────────────┐
                       │   LLMServer Deployment     │
                       │ (ctransformers + FastAPI)  │
                       └────────────┬───────────────┘
                                    │
                     ┌──────────────▼──────────────┐
                     │ ctransformers.ModelWrapper  │
                     │  loads GGUF (Qwen, Mistral) │
                     └──────────────┬──────────────┘
                                    │
                            ┌───────▼─────────┐
                            │  CPU/GPU Backend│  ← libopenblas used internally
                            └─────────────────┘

```



```
                    +----------------------------------------+
                    |  Developer (You)                      |
                    |  - Writes Dockerfile + rayserve_app.py|
                    |  - Creates Helm chart + values.yaml   |
                    +-----------------+----------------------+
                                      |
                                      v
+-------------------------------+     |     +------------------------------+
|  Docker Build (with BuildKit) |<----+---->|  HuggingFace Hub (GGUF model)|
|  ---------------------------- |           |  - Model pulled securely     |
|  - Uses multi-stage build     |           +------------------------------+
|  - Injects HF_TOKEN via secret|
|  - Installs Ray + ctransformers|
|  - Copies model + code        |
+---------------+---------------+
                |
                v
       +--------+---------+
       | Push to GHCR     |  <-- authenticated using GHCR_PAT
       | ghcr.io/you/ctr...|
       +------------------+
                |
                v
       +--------+---------+
       | Helm Chart Deploy |
       | `helm upgrade ...`|
       +--------+---------+
                |
                v
+------------------------------------ Kubernetes Cluster ------------------------------------+
|                                                                                           |
|  +------------------------+         +---------------------------------+                   |
|  |  Ray Head Pod          |         |  Ray Worker Pod (autoscaling)   |                   |
|  |------------------------|         |---------------------------------|                   |
|  | Image: ghcr.io/...     |         | Image: same as head             |                   |
|  | Entrypoint:            |         | Entrypoint:                     |                   |
|  |   serve run rayserve...|         |   serve run rayserve...         |                   |
|  | Starts Ray Head (GCS)  |<------->| Connects to Ray Head            | <--- autoscaled   |
|  | Deploys FastAPI + LLM  |         | Hosts replicas of LLM app       |       via        |
|  | Mounts /models dir     |         +---------------------------------+   RayService spec |
|  +------------------------+                                                           |
|         |                                                                                 |
|         |                                                                                 |
|         v                                                                                 |
|  HTTP Server on :8000 (Ray Serve Proxy + FastAPI)                                         |
|     - Accepts POST /v1/chat/completions                                                   |
|     - Loads GGUF model with ctransformers                                                 |
|     - Returns output from model                                                           |
|                                                                                           |
+-------------------------------------------------------------------------------------------+

```