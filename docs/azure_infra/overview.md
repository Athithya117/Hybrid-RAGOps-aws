## 1. **Platform Foundation (Shared Infrastructure Primitives)**

**Scope**

* Core Azure resources that everything else depends on.
* Network, identity primitives, and baseline storage.
* Should be deployed first and change least often.

**Files**

* `core_network.py`
  *VNETs, subnets, routing, base networking constructs.*

* `storage_account.py`
  *Blob/storage accounts used by downstream services.*

* `uai_key_vault_secrets.py`
  *User-assigned identities and Key Vault secrets wiring.*

* `bootstrap.sh`
  *Initial Pulumi/bootstrap setup for the stack.*

---

## 2. **Security, Identity & Access Layer**

**Scope**

* Authentication, authorization, and identity-facing infrastructure.
* Explicit trust boundaries and access control logic.

**Files**

* `auth.py`
  *OIDC / Entra / identity provider integration.*

* `edge.py`
  *Ingress, edge exposure, public access, TLS termination.*

---

## 3. **Compute & Deployment Orchestration**

**Scope**

* Workload execution environment and deployment glue.
* Ties infrastructure primitives to actual running services.

**Files**

* `aks.py`
  *AKS cluster definition and configuration.*

* `__main__.py`
  *Pulumi entry point; orchestrates resource creation order.*

* `run.sh`
  *Local execution wrapper for Pulumi commands.*

---

# Non-grouped / Auxiliary Artifacts

These are intentionally excluded from logical groupings because they are not runtime infrastructure code:

* `Pulumi.yaml`
* `Pulumi.staging.yaml`
* `pulumi-exports.sh`
* `pulumi-outputs.json`
* `requirements.txt`
* `test_common_env`
* `venv/`
* `__pycache__/`

---

# Resulting Mental Model

```
Foundation
 ├─ core_network.py
 ├─ storage_account.py
 └─ uai_key_vault_secrets.py

Security & Edge
 ├─ auth.py
 └─ edge.py

Compute & Orchestration
 ├─ aks.py
 └─ __main__.py
```

This structure:

* Makes dependency direction explicit.
* Prevents circular imports.
* Allows partial stack deployment (e.g., foundation-only).
* Aligns with Pulumi’s execution semantics.

