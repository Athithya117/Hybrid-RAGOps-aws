# RAG8s ONNX: embedder + reranker (mini Helm chart)

**Location:** `infra/onnx/`
**Purpose:** Docker image + Ray Serve code + Helm-style manifest templates for running the ONNX embedder & reranker on KubeRay (RayService) in a cloud Kubernetes environment.

This README is a concise, production-oriented reference for developers/operators: what is in `infra/onnx/`, how to build and test, how to deploy, configuration and secrets, monitoring and autoscaling notes, and common troubleshooting steps.

---

## Quick summary (one-liner)

Build and push the image (CI), create your HF secret in the cluster, then deploy the `RayService` via the templated Helm manifest; the RayService will bootstrap Ray + Serve and deploy the two services: **Embedder** and **Reranker**.

---

## Contents (important files & purpose)

```
infra/onnx/
├── Chart.yaml                     # Helm metadata
├── values.yaml                    # All runtime knobs (image, resources, autoscale, env, probe timings)
├── Dockerfile                     # Multi-stage build: build->runtime (pre-generate gRPC stubs)
├── requirements.txt               # pinned runtime deps
├── grpc.proto                     # gRPC API (generate stubs at build time)
├── rayserve-embedder-reranker.py  # Model code: Serve deployments (importable, no side-effects)
├── rayserve_entrypoint.py         # Importable entrypoint for serveConfigV2 (import_path)
├── build_and_push.sh              # local helper (prefer CI)
├── templates/                     # Helm templates (rayservice.yaml, hf-secret, RBAC, PDB, servicemonitor)
├── tests/
│   ├── smoke_test.sh              # CI / local smoke test
│   └── test_inference.py          # small integration test
└── docs/
    ├── runbook.md                 # on-call runbook
    └── ops-checklist.md
```

---

## Key concepts & design decisions

* **Image-first approach**: runtime image contains Ray\[serve], ONNX runtime, model files (optional). Using a single image for head & workers avoids expensive runtime\_env transfers.
* **serveConfigV2 import path**: `rayserve_entrypoint.py` is importable inside the image and either returns an `app` object or deploys the Serve deployments idempotently.
* **KubeRay RayService**: the Helm template renders a `RayService` CR (`ray.io/v1`) with `rayClusterConfig` & `serveConfigV2`.
* **Secrets**: HF token is **not** baked into image. Use a Kubernetes `Secret` (template provided).
* **Production-grade**: readiness/liveness probes, non-root container, resource requests/limits, PodDisruptionBudget, anti-affinity, Prometheus scrape annotations, and CI scanning.

---

## Build & CI (recommended)

**CI location:** `.github/workflows/onnx.yaml` (workflow triggers on `infra/onnx/**` and supports `workflow_dispatch`).

Essential CI steps:

1. Checkout repo
2. Build image from `infra/onnx/Dockerfile` (use buildx for multi-arch)
3. Run security scan (Trivy or equivalent)
4. Push image to registry (`IMAGE_REPO` set in secrets)
5. Optionally run container smoke test (`HF_TOKEN` required) to confirm health endpoint/inference

**Local quick build & test (dev):**

```bash
# build locally
docker build -f infra/onnx/Dockerfile -t $IMAGE_REPO:$TAG infra/onnx

# run locally (mount HF_TOKEN as env)
docker run --rm -p 8000:8000 -e HF_TOKEN="${HF_TOKEN}" $IMAGE_REPO:$TAG

# run smoke test against local container
infra/onnx/tests/smoke_test.sh
```

---

## Deployment (Helm / KubeRay)

Prereqs:

* KubeRay operator installed in cluster
* Kubernetes cluster with node autoscaler or Karpenter if autoscaling nodes is required
* Secret containing Hugging Face token present (`hf_token` key)

Create HF secret (example):

```bash
kubectl create secret generic rag8s-hf-token \
  --from-literal=hf_token="${HF_TOKEN}" \
  -n rag8s-prod
```

Install (Helm example):

```bash
# Optionally edit infra/onnx/values.yaml for your environment
helm upgrade --install rag8s-onnx infra/onnx \
  --namespace rag8s-prod --create-namespace \
  --set image.repository=yourrepo/rag8s-onnx \
  --set image.tag=gte-modernbert
```

Verify:

```bash
kubectl -n rag8s-prod get rayservices
kubectl -n rag8s-prod get pods -l app.kubernetes.io/name=rag8s-onnx
kubectl -n rag8s-prod logs <ray-head-pod> -c ray-head
kubectl -n rag8s-prod port-forward <ray-head-pod> 8000:8000
curl http://127.0.0.1:8000/healthz
```

---

## Configuration (values.yaml highlights)

Edit `infra/onnx/values.yaml` for environment-specific settings:

* `image.repository` & `image.tag` – the image to use for both head & worker
* `rayVersion` – must match Ray installed in the image (2.48.0 for the provided image)
* `replicaDefaults.head|worker.resources.requests/limits` – pod sizing
* `serve.importPath` – should be `rayserve_entrypoint:app` (or adjust to your entrypoint)
* `env.*` – ONNX/OMP thread knobs, model names, HF\_HOME path, batching params
* `secrets.hfTokenSecretName` – secret name referencing the HF token

**Important:** do not commit actual secret values to `values.yaml`.

---

## Runtime env vars (important)

Make these available in pod template (via Helm values):

* `HF_HOME` – local cache path for Hugging Face models or baked-in model dir
* `MODEL_EMBEDDER_NAME`, `MODEL_RERANKER_NAME` – HF model ids (used if you opt to lazy-download)
* `EMBEDDER_OMP_NUM_THREADS`, `RERANKER_OMP_NUM_THREADS` – set to match CPU request (avoid oversubscribe)
* `EMBEDDER_BATCH_MAX_SIZE`, `RERANKER_BATCH_MAX_SIZE` – batch tuning
* `HF_TOKEN` – must come from K8s Secret via `valueFrom.secretKeyRef`

---

## Health, readiness & graceful shutdown

* The container exposes `/healthz` which returns `200` only after models are loaded and the Serve deployments are ready. This is wired to readinessProbe.
* `terminationGracePeriodSeconds` is set (e.g., 120s): on SIGTERM the entrypoint attempts to call `serve.shutdown()` and stop health server to drain requests.
* Probes are configurable via `values.yaml`.

---

## Monitoring & metrics

* Prometheus annotations are added by default (toggle in `values.yaml`).
* Application-level metrics use `prometheus_client`:

  * `rag8s_inference_requests_total{model=...}`
  * `rag8s_inference_latency_seconds{model=...}`
* Export Ray dashboard metrics (dashboard port 8265) and scrape via ServiceMonitor when Prometheus operator is present.

---

## Autoscaling & resource guidance

* Two layers:

  1. **Ray-level autoscaling** (Serve autoscaling per deployment: min/max replicas, target ongoing requests per replica)
  2. **Kubernetes node autoscaler** (cloud autoscaler or Karpenter) — necessary so cluster can provide nodes when Ray requests more pods
* Tune:

  * `ray_actor_options.num_cpus` to reflect actor CPU usage
  * set `EMBEDDER_OMP_NUM_THREADS` = CPUs allocated to each actor/pod to avoid oversubscription
  * load-test to determine `target_num_ongoing_requests_per_replica` and `max_replicas`

---

## ONNX runtime tuning tips

* Use explicit `onnxruntime.SessionOptions()` and set `intra_op_num_threads` to match CPU assignment.
* Set `OMP_NUM_THREADS` / `MKL_NUM_THREADS` environment variables in pod to prevent oversubscription.
* Tune `max_batch_size` and `batch_wait_timeout_s` according to SLOs (latency vs throughput).

---

## Testing

* Local smoke test: `infra/onnx/tests/smoke_test.sh` (calls `/healthz` and optionally runs a tiny inference)
* Integration: `infra/onnx/tests/test_inference.py` can be run in CI against a running container or a staging RayService
* Load testing: use k6/locust with realistic request patterns to tune autoscaling.

---

## Troubleshooting checklist (quick)

* `kubectl -n <ns> get rayservice` → ensure RayService status indicates success
* `kubectl -n <ns> get pods` → check head & worker pods; `kubectl describe pod <pod>` for events
* `kubectl logs <ray-head-pod> -c ray-head` → Ray head logs (dashboard, serve errors)
* Health probe failing: check `/healthz` logs, verify model paths and HF token access
* Image pull failure: verify `image.repository` and registry credentials
* Model not found: check `EMBEDDER_ONNX_PATH` / `RERANKER_ONNX_PATH` or the `HF_HOME` layout if models are baked
* OOM/CPU throttling: inspect `kubectl top pod` and adjust requests/limits and ORT threads
* Ray version mismatch: ensure `rayVersion` in `values.yaml` and Ray installed in image match

---

## Security & secrets

* Never bake HF tokens into images or commit them to `values.yaml`.
* Use Kubernetes Secrets (or external secret managers) and mount via `valueFrom.secretKeyRef`.
* Run container as non-root (Dockerfile creates a non-root user).
* Use least-privilege ServiceAccount & RBAC (`templates/role*.yaml` available).

---

## Operational runbook pointers

* See `infra/onnx/docs/runbook.md` for on-call steps, key metrics, and rollback instructions.
* See `infra/onnx/docs/ops-checklist.md` for pre-deploy validation list (CI green, scan results, capacity checks).

---

## Quick commands (cheat sheet)

Build locally:

```bash
docker build -f infra/onnx/Dockerfile -t $IMAGE_REPO:$TAG infra/onnx
docker push $IMAGE_REPO:$TAG
```

Create HF secret:

```bash
kubectl create secret generic rag8s-hf-token \
  --from-literal=hf_token="${HF_TOKEN}" -n rag8s-prod
```

Install chart:

```bash
helm upgrade --install rag8s-onnx infra/onnx -n rag8s-prod --create-namespace \
  --set image.repository=$IMAGE_REPO --set image.tag=$TAG
```

Tail logs:

```bash
kubectl -n rag8s-prod logs -f <pod> -c ray-head
```

Run GH Actions manually: GitHub → Actions → select `Build, scan, push ONNX image` → Run workflow.

---

