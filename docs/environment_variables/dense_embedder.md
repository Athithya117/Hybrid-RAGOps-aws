# Dense embedder — deployment documentation

## Overview

This document explains how to generate and apply Kubernetes manifests for the **dense embedder** service. It provides two ready-to-use environment export blocks (staging / local Kind and production EKS) and a detailed reference for every environment variable the generator accepts. Use the staging block for local testing or CI dry-runs; use the production block for real cluster deployments (EKS). Each export block includes one-line TL;DR annotations on the right.

---

## TL;DR export — staging / local (Kind)

```bash
# TLDR: local/kind → single replica, CPU-only, monitoring disabled (safe for CI/Kind)
export ENV=STAGING                          # environment selection (STAGING|PROD)
export DENSE_IMAGE="athithya5354/dense:amd64-arm64-v1"  # container image
export DENSE_REPLICAS=1                     # deployment replicas
export DENSE_CPU_REQ="500m"                 # cpu request
export DENSE_MEM_REQ="1Gi"                  # memory request
export DENSE_CPU_LIMIT="2000m"              # cpu limit
export DENSE_MEM_LIMIT="4Gi"                # memory limit
export DENSE_PORT=8200                      # container port
export PROBE_INITIAL_DELAY=10               # readiness probe initial delay (s)
export PROBE_TIMEOUT=3                      # probe timeout (s)
export PROBE_PERIOD=10                      # probe period (s)
export HPA_MIN=1                            # HPA min replicas
export HPA_MAX=6                            # HPA max replicas
export HPA_CPU_TARGET=60                    # HPA target CPU utilization (%)
export DENSE_GPU=0                          # GPU enabled? (0/1)
export DENSE_GPU_COUNT=1                    # GPU count when enabled
export DENSE_NODE_SELECTOR=""               # nodeSelector (e.g. "gpu=true")
export DENSE_TAINT_KEY=""                   # toleration key if needed
export DISABLE_MONITORING=true              # disable ServiceMonitor/PromRule in small clusters
export DENSE_NAMESPACE="models"             # k8s namespace to write/apply manifests into
export DENSE_SA="dense-sa"                  # ServiceAccount name
```

---

## TL;DR export — production (EKS)

```bash
# TLDR: prod/eks → multi-replica, higher cpu/memory, monitoring enabled, GPU supported
export ENV=PROD                             # environment selection (STAGING|PROD)
export DENSE_IMAGE="athithya5354/dense:amd64-arm64-v1"  # container image (override for release tag)
export DENSE_REPLICAS=2                     # default replica target (adjust for scale)
export DENSE_CPU_REQ="1000m"                # cpu request
export DENSE_MEM_REQ="4Gi"                  # memory request
export DENSE_CPU_LIMIT="4000m"              # cpu limit
export DENSE_MEM_LIMIT="16Gi"               # memory limit
export DENSE_PORT=8200                      # container port
export PROBE_INITIAL_DELAY=10               # readiness probe initial delay (s)
export PROBE_TIMEOUT=3                      # probe timeout (s)
export PROBE_PERIOD=10                      # probe period (s)
export HPA_MIN=2                            # HPA min replicas
export HPA_MAX=6                            # HPA max replicas
export HPA_CPU_TARGET=60                    # HPA target CPU utilization (%)
export DENSE_GPU=0                          # GPU enabled? (set to 1 if GPU nodes available)
export DENSE_GPU_COUNT=1                    # GPU count if DENSE_GPU=1
export DENSE_NODE_SELECTOR=""               # set to e.g. "gpu=true" when targeting GPU nodes
export DENSE_TAINT_KEY=""                   # taint key to tolerate if nodes are tainted
export DISABLE_MONITORING=false             # keep monitoring enabled in production
export DENSE_NAMESPACE="models"             # k8s namespace to write/apply manifests into
export DENSE_SA="dense-sa"                  # ServiceAccount name
```

---

## Environment variable reference — explanation and examples

Each variable below is the canonical variable the generator reads. For each var: short description, effect, recommended values and a concrete staging and production example.

---

### `ENV`

* **Purpose:** Selects generation defaults and behavior (affects replica and resource defaults).
* **Accepted:** `STAGING` or `PROD` (case-insensitive).
* **Staging example:** `ENV=STAGING` — generator uses conservative defaults suitable for Kind/CI.
* **Prod example:** `ENV=PROD` — generator uses larger defaults and enables production assumptions.

---

### `DENSE_IMAGE`

* **Purpose:** Container image for the dense embedder. The generator writes this into the Deployment image field.
* **Format:** `repository:tag` (Docker image reference).
* **Staging example:** `DENSE_IMAGE=athithya5354/dense:amd64-arm64-v1`
* **Prod example:** `DENSE_IMAGE=athithya5354/dense:release-2025-12-01` (pin to immutable tag for production).

---

### `DENSE_REPLICAS`

* **Purpose:** Initial number of Deployment replicas (used when generating Deployment.spec.replicas). HPA may scale above or down to min/max.
* **Staging example:** `DENSE_REPLICAS=1`
* **Prod example:** `DENSE_REPLICAS=2`

---

### `DENSE_CPU_REQ` / `DENSE_MEM_REQ`

* **Purpose:** CPU and memory **requests** for container (k8s resources.requests). Requests determine scheduling.
* **Format:** CPU (e.g., `500m`, `1000m`), Memory (e.g., `1Gi`, `4Gi`).
* **Staging example:** `DENSE_CPU_REQ=500m`, `DENSE_MEM_REQ=1Gi`
* **Prod example:** `DENSE_CPU_REQ=1000m`, `DENSE_MEM_REQ=4Gi`

---

### `DENSE_CPU_LIMIT` / `DENSE_MEM_LIMIT`

* **Purpose:** CPU and memory **limits** for container (k8s resources.limits). Limits cap runtime usage.
* **Staging example:** `DENSE_CPU_LIMIT=2000m`, `DENSE_MEM_LIMIT=4Gi`
* **Prod example:** `DENSE_CPU_LIMIT=4000m`, `DENSE_MEM_LIMIT=16Gi`

---

### `DENSE_PORT`

* **Purpose:** Container HTTP port for `/health` and `/metrics`. Generator exposes Service port and containerPort.
* **Default:** `8200`
* **Staging & Prod example:** `DENSE_PORT=8200`

---

### Probe timing: `PROBE_INITIAL_DELAY`, `PROBE_TIMEOUT`, `PROBE_PERIOD`

* **Purpose:** Configure readiness and liveness probe timing to match startup latency. All values are seconds.
* **Staging example:** `PROBE_INITIAL_DELAY=10`, `PROBE_TIMEOUT=3`, `PROBE_PERIOD=10`
* **Prod example:** same defaults recommended, tune upward for slower cold starts.

---

### Horizontal Pod Autoscaler (HPA) settings: `HPA_MIN`, `HPA_MAX`, `HPA_CPU_TARGET`

* **Purpose:** Controls autoscaling behavior in generated HPA (v2 API, CPU utilization target).
* **Staging example:** `HPA_MIN=1`, `HPA_MAX=6`, `HPA_CPU_TARGET=60`
* **Prod example:** `HPA_MIN=2`, `HPA_MAX=10`, `HPA_CPU_TARGET=60` (adjust max to expected peak load).

---

### GPU support: `DENSE_GPU`, `DENSE_GPU_COUNT`

* **Purpose:** When `DENSE_GPU=1`, the generator will add a resource limit `nvidia.com/gpu: <DENSE_GPU_COUNT>` to the container and you can set nodeSelector/tolerations. This leaves CPU/memory fields unchanged; ensure GPU node sizing fits your model.
* **Staging example:** `DENSE_GPU=0`, `DENSE_GPU_COUNT=1` (default CPU mode)
* **Prod example (GPU-enabled):** `DENSE_GPU=1`, `DENSE_GPU_COUNT=1` and `DENSE_NODE_SELECTOR="gpu=true"` (assumes node label `gpu=true` and device-plugin installed)

---

### `DENSE_NODE_SELECTOR`

* **Purpose:** Single key=value pair to target nodes (used in `Pod.spec.nodeSelector`). Example format `gpu=true` or `kubernetes.io/hostname=node-01`.
* **Staging example:** `DENSE_NODE_SELECTOR=""` (no selector)
* **Prod example:** `DENSE_NODE_SELECTOR="gpu=true"`

---

### `DENSE_TAINT_KEY`

* **Purpose:** If node taints are used for isolation, set this to the taint key so generator emits a toleration for it (operator `Exists`, effect `NoSchedule`). Use when dense pods must land on tainted nodes.
* **Staging example:** `DENSE_TAINT_KEY=""`
* **Prod example:** `DENSE_TAINT_KEY="workload=gpu"`

---

### `DISABLE_MONITORING`

* **Purpose:** When `true`, generator omits `ServiceMonitor` and `PrometheusRule` objects. Set `true` for small local clusters without Prometheus operator. Default is `false`.
* **Staging example:** `DISABLE_MONITORING=true` (avoid missing CRD apply failures in Kind)
* **Prod example:** `DISABLE_MONITORING=false` (enable monitoring in EKS with Prometheus operator)

---

### `DENSE_NAMESPACE`

* **Purpose:** Kubernetes namespace where manifests will be written / applied. Generator writes a Namespace object when generating.
* **Staging example:** `DENSE_NAMESPACE=models`
* **Prod example:** `DENSE_NAMESPACE=models` or `DENSE_NAMESPACE=platform-models` per naming conventions

---

### `DENSE_SA`

* **Purpose:** ServiceAccount name created for the deployment; can be overridden to bind specific IAM/IRSA roles in EKS.
* **Staging example:** `DENSE_SA=dense-sa`
* **Prod example:** `DENSE_SA=dense-sa` with IRSA annotation applied separately if using IAM roles for service accounts.

---

### `SERVICEACCOUNT / IAM / IRSA notes`

* The generator writes a ServiceAccount (`DENSE_SA`) but does **not** automatically create IRSA role bindings. For EKS production use, bind an IAM role to the ServiceAccount (IRSA) or attach the necessary IAM permissions externally. If IRSA is used, annotate the ServiceAccount via your infra tooling rather than embedding secrets.

---

### Labels / metadata

* The generator uses labels `{app.kubernetes.io/name: dense, app.kubernetes.io/component: embedder}` by default. These can be overridden in the generator or post-processed by template tools to align with cluster conventions.

---

### Files & locations produced

* By default the generator writes a single multi-document YAML to:

  ```
  infra/manifests/dense/dense-manifests.yaml
  ```

  It also writes a `.inputs_hash` in that directory to prevent redundant rewrites.

---

### Typical workflows

* **Local / CI smoke test**

  1. Build container locally and load into Kind/Docker.
  2. `ENV=STAGING DISABLE_MONITORING=true DENSE_IMAGE=dense:local python3 infra/generators/gen_dense.py --generate`
  3. `kubectl apply -f infra/manifests/dense/dense-manifests.yaml` (or run generator `--apply` if `kubectl` configured).

* **Production deployment (EKS)**

  1. Ensure cluster has Prometheus operator if `DISABLE_MONITORING=false`.
  2. Ensure GPUs, node labels, taints and IAM/IRSA are provisioned if using GPU/IRSA.
  3. Pin and upload image to registry: use an immutable tag (e.g. release SHA).
  4. Set `ENV=PROD DENSE_IMAGE=<registry>/dense:<tag> DENSE_GPU=<0|1> python3 infra/generators/gen_dense.py --generate` then `--apply` (or let CI/CD apply).

---

## Where to change values for production readiness

* Pin `DENSE_IMAGE` to a digest or immutable tag.
* Increase `HPA_MAX` and resource limits to match inference throughput and latency SLOs.
* Enable `DISABLE_MONITORING=false` and make sure Prometheus CRDs exist in cluster.
* For GPUs: set `DENSE_GPU=1`, set `DENSE_NODE_SELECTOR` to match labeled GPU nodes and validate device plugin.
* Use infra tooling (Pulumi/Helm/ArgoCD) to annotate ServiceAccount for IRSA rather than embedding credentials.

---

## Final notes

* The generator is deterministic (it hashes canonical inputs to avoid spurious file changes). The exported environment variables shown above are exhaustive for the generator behavior. Modify only the env vars listed to keep manifest generation predictable.
