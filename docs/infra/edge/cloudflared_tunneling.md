Below is a **professional, concrete, and precise** document suitable for
`docs/infra/edge/cloudflared_tunneling.md`.
It is written as engineering documentation, not marketing material.

---

## Cloudflared Tunneling (Edge Access)

### Purpose

This document describes how **Cloudflare Tunnel (cloudflared)** is used to expose the RAG frontend running inside Kubernetes to the public internet **without opening inbound ports**, and how traffic flows from the edge to in-cluster services.

The design prioritizes:

* Zero inbound firewall rules
* Deterministic routing
* Minimal coupling between infra and app layers
* Compatibility with local (kind) and managed (AKS) clusters

---

## High-level Architecture

At a high level, Cloudflare Tunnel establishes **outbound-only connections** from the Kubernetes cluster to Cloudflare’s edge. Incoming user requests are terminated at Cloudflare and forwarded through the tunnel to an internal Kubernetes service.

```
Browser
  │
  ▼
Cloudflare Edge (DNS + HTTPS)
  │
  ▼
Cloudflare Tunnel
(cloudflared pods)
  │
  ▼
frontend-svc (ClusterIP)
  │
  ▼
FastAPI frontend (app.py)
```

No Kubernetes `LoadBalancer`, `NodePort`, or public ingress is required.

---

## Components

### Cloudflare Edge

* Owns DNS and TLS termination.
* Routes traffic to a specific tunnel based on hostname.
* Enforces Cloudflare-level protections (WAF, rate limits, auth if enabled).

### cloudflared (in-cluster)

* Runs as a Kubernetes `Deployment`.
* Maintains outbound WebSocket/QUIC connections to Cloudflare.
* Forwards HTTP requests to internal services based on `config.yml`.

### Frontend Service

* Kubernetes `Service` (ClusterIP).
* Exposes the FastAPI frontend on an internal DNS name:

  ```
  frontend-svc.<namespace>.svc.cluster.local
  ```

---

## Deployment Model

### Namespace

By default, cloudflared is deployed into the same namespace as the frontend (`inference`), but this is configurable.

### Authentication Modes

Two Cloudflare Tunnel auth modes are supported:

1. **Tunnel Token (recommended)**

   * Uses `CLOUDFLARE_TUNNEL_TOKEN`
   * No credentials file required
   * Simpler rotation and automation

2. **Credentials File**

   * Uses `credentials.json`
   * Mounted from a Kubernetes Secret
   * Requires tunnel name coordination

Only one mode is active at runtime.

---

## Generated Kubernetes Resources

The generator (`infra/generators/cloudflared.py`) produces:

1. **ServiceAccount**

   * Minimal permissions (no cluster-wide RBAC).

2. **ConfigMap**

   * Contains `config.yml`:

     ```yaml
     ingress:
       - hostname: rag.example.com
         service: http://frontend-svc.inference.svc.cluster.local:8000
       - service: http_status:404
     ```

3. **Deployment**

   * Runs one or more cloudflared replicas.
   * Health checks via `cloudflared --version`.

4. **Secrets** (optional)

   * Tunnel token or credentials file.
   * Can be embedded or applied dynamically.

---

## Runtime Control Flow

### Request Path (End-to-End)

1. **User Request**

   * Browser sends `https://rag.example.com`.

2. **Cloudflare Edge**

   * Terminates TLS.
   * Matches hostname to tunnel.
   * Selects an active tunnel connection.

3. **Tunnel Forwarding**

   * Cloudflare sends the request over the tunnel.
   * cloudflared receives it inside the cluster.

4. **Ingress Rule Match**

   * cloudflared matches the hostname rule.
   * Forwards request to:

     ```
     frontend-svc.inference.svc.cluster.local:8000
     ```

5. **Frontend Application**

   * `app.py` handles routing:

     * `/` → UI
     * `/auth/*` → auth service
     * `/run` → proxy to retrieval service
   * Optional auth enforcement.

6. **Response**

   * Returned along the same path in reverse.
   * TLS is terminated only at Cloudflare.

---

## Mental Model (Tree View)

Think of the system as a **strictly layered tree**, not a mesh:

```
Internet
 └─ Cloudflare Edge
     └─ Tunnel (logical)
         └─ cloudflared Pods
             └─ frontend Service (ClusterIP)
                 └─ FastAPI App
                     ├─ UI
                     ├─ Auth
                     └─ Proxy to Retrieval
```

Key properties:

* No lateral routing at the edge layer
* No inbound traffic reaches Kubernetes directly
* cloudflared is **not** an ingress controller; it is a tunnel endpoint

---

## Failure Modes and Behavior

### cloudflared Pod Failure

* Other replicas continue serving traffic.
* Cloudflare automatically re-routes to healthy tunnel connections.

### Frontend Service Failure

* cloudflared returns 502 to Cloudflare.
* Cloudflare returns error to client.

### Tunnel Disconnect

* Cloudflare retries until tunnel reconnects.
* No DNS changes required.

### DNS Resolution

* cloudflared relies on **cluster DNS** to resolve:

  ```
  frontend-svc.<namespace>.svc.cluster.local
  ```
* DNS instability in kind will surface as:

  ```
  Name or service not known
  ```

---

## Local Development (kind)

Important caveats when running on kind:

* CoreDNS instability can break cloudflared routing.
* Resource pressure (CPU/memory) frequently impacts DNS.
* cloudflared failures often manifest as *DNS errors*, not HTTP errors.

Recommended mitigations:

* Increase Docker memory allocation.
* Pin CoreDNS resources.
* Avoid frequent pod churn during indexing jobs.

---

## Security Considerations

* No public Kubernetes endpoints.
* TLS handled entirely by Cloudflare.
* Secrets are not written to Git unless explicitly requested.
* JWT and session handling remain application-layer concerns.

---

## Operational Guidelines

* Treat cloudflared as **edge infrastructure**, not application code.
* Do not colocate cloudflared lifecycle with indexing or batch jobs.
* Rotate tunnel tokens independently of app deployments.
* Monitor tunnel health via Cloudflare dashboard, not Kubernetes alone.

---

## Summary

Cloudflare Tunnel provides a clean, outbound-only edge for the RAG frontend.
The design intentionally separates:

* **Edge exposure** (cloudflared)
* **Application routing** (FastAPI)
* **Authentication** (stateless OpenID)

This separation keeps failure domains small, improves security posture, and makes the system portable across local and managed Kubernetes environments.
