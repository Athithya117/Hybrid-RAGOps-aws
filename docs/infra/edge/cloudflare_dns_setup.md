## Cloudflare DNS and Tunnel Setup

### Purpose

This document explains how to:

1. Acquire a public domain
2. Attach it to Cloudflare DNS
3. Authenticate locally with Cloudflare
4. Create and bind a Cloudflare Tunnel
5. Export credentials for Kubernetes deployment
6. Cleanly log out and revoke local state

The setup enables **secure, outbound-only exposure** of the frontend without opening inbound ports.

---

## Prerequisites

* A public domain (any registrar)
* A Cloudflare account
* Local machine with:

  * `bash`
  * `curl`
  * `kubectl` (for later deployment)
* Optional but recommended:

  * `jq`
* Linux or macOS (script assumes GNU utilities)

---

## Step 1: Acquire a Domain

Purchase a domain from any registrar (e.g. Namecheap, Google Domains, Route53).

After purchase:

* You must be able to modify **nameservers**
* No DNS records need to exist yet

Example domain used throughout this document:

```
athithya.site
```

---

## Step 2: Add Domain to Cloudflare

1. Log in to Cloudflare Dashboard
2. Click **Add a Site**
3. Enter your domain (e.g. `athithya.site`)
4. Select a plan (Free is sufficient)
5. Cloudflare will assign **two nameservers**

At your registrar:

* Replace existing nameservers with Cloudflare-provided ones
* Save changes

Propagation typically completes within minutes (can take up to 24 hours).

Cloudflare now becomes the authoritative DNS provider.

---

## Step 3: Local Cloudflare Authentication

Authentication is performed **locally**, not in Kubernetes.
This creates a Cloudflare account certificate under `~/.cloudflared`.

### Recommended entrypoint (Makefile)

```
make cloudflare-setup
```

### Under the hood

This runs:

```
bash infra/setup/cloudflared.sh
```

The script:

* Installs a pinned `cloudflared` version if missing
* Launches an interactive browser login
* Creates (or reuses) a tunnel
* Exports credentials into a secrets file

During login:

* A browser URL is printed
* You must select the correct Cloudflare account
* You must authorize access to the domain zone

On success:

```
~/.cloudflared/cert.pem
```

is created.

This file represents **account-level trust**, not a specific tunnel.

---

## Step 4: Tunnel Creation and Credential Export

After authentication, the script ensures a tunnel exists.

Default tunnel name:

```
rag-frontend
```

Two credential strategies are supported:

### Preferred: Tunnel Token

* Generated via `cloudflared tunnel token`
* Stored as:

  ```
  export CLOUDFLARE_TUNNEL_TOKEN=...
  ```

### Fallback: credentials.json

* Base64-encoded
* Stored as:

  ```
  export CLOUDFLARE_TUNNEL_CREDENTIALS_B64=...
  ```

Credentials are persisted to:

```
~/.config/rag/secrets.env
```

Permissions are locked to `600`.

### Reuse in new shells

```
source ~/.config/rag/secrets.env
```

---

## Step 5: DNS Binding (Hostname → Tunnel)

The script can optionally bind a public hostname to the tunnel.

Default behavior:

* Enabled
* Safe overwrite of existing DNS record

Example hostname:

```
ui.athithya.site
```

Cloudflare creates a **CNAME-like record** pointing to the tunnel UUID.

Important:

* No IP address is exposed
* Traffic only flows through Cloudflare Tunnel

DNS binding can also be done manually in the Cloudflare dashboard if desired.

---

## Runtime Mental Model

```
Browser
  ↓
Cloudflare DNS
  ↓
Cloudflare Edge (TLS termination)
  ↓
Cloudflare Tunnel
  ↓
cloudflared (Kubernetes)
  ↓
frontend Service (ClusterIP)
  ↓
FastAPI frontend
```

Key properties:

* No inbound ports
* No Kubernetes ingress
* TLS never reaches the cluster
* DNS is authoritative at Cloudflare

---

## Step 6: Kubernetes Consumption

At deploy time, Kubernetes only needs **one of**:

* `CLOUDFLARE_TUNNEL_TOKEN`, or
* `CLOUDFLARE_TUNNEL_CREDENTIALS_B64`

These are injected into the cloudflared Deployment via Secret.

The frontend does **not** know or care about Cloudflare.

---

## Makefile Integration

Recommended Makefile targets:

```
cloudflare-setup:
	bash infra/setup/cloudflared.sh

cloudflare-logout:
	rm -rf ~/.cloudflared \
	       ~/.config/rag/secrets.env && \
	unset CLOUDFLARE_TUNNEL_TOKEN \
	      CLOUDFLARE_TUNNEL_CREDENTIALS_B64 \
	      CLOUDFLARE_TUNNEL_NAME
```

Usage:

```
make cloudflare-setup
```

To fully reset local Cloudflare state:

```
make cloudflare-logout
```

This:

* Removes account cert
* Deletes persisted secrets
* Clears environment variables

---

## Failure Modes and Recovery

### Login did not complete

* Re-run `make cloudflare-setup`
* Ensure browser authorization completed

### DNS not resolving

* Verify nameservers at registrar
* Check Cloudflare DNS tab
* Ensure hostname matches tunnel binding

### Multiple tunnels accidentally created

* List with:

  ```
  cloudflared tunnel list
  ```
* Delete unused tunnels in Cloudflare dashboard

---

## Security Notes

* Account cert stays local
* Tunnel token is scoped to a single tunnel
* Kubernetes never receives account-level credentials
* Revocation is immediate by deleting the tunnel in Cloudflare

---

## Summary

This setup provides:

* Secure domain ownership
* Zero-trust edge exposure
* Deterministic DNS routing
* Clean separation between infra, edge, and app layers

Cloudflare handles **DNS, TLS, and edge routing**.
Kubernetes remains **private and outbound-only**.
