



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
