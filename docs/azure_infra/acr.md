# Azure Container Registry (ACR) — Platform Documentation

This document defines how the platform uses Azure Container Registry, how registries are created or adopted, how Pulumi determines ownership, and which operational rules are enforced to prevent common failures such as `SkuNotSupported`, accidental re-creation, or broken AKS image pulls.

---

## Design Objectives

* Deterministic container image supply chain with explicit ownership semantics
* Clear opt-in behavior for reusing existing registries; safe defaults otherwise
* Predictable behavior across regions with known SKU availability gaps
* Minimal, well-defined operator configuration surface

---

## Operational Summary (TL;DR)

* To **reuse an existing ACR**, explicitly set:

  * `ACR_NAME` to the registry name
  * `ACR_RESOURCE_RG` to the resource group containing that registry
    Pulumi will detect and adopt the registry without attempting creation.

* To let Pulumi **create and own** the registry:

  * Set `ACR_NAME`
  * Leave `ACR_RESOURCE_RG` unset or equal to the stack resource group (`rg-e2e-rag` by default)

* In India regions (for example `centralindia`, `southindia`), use `ACR_SKU=Premium`.
  `Standard` is frequently rejected with `SkuNotSupported`.

* To allow AKS to pull images without `imagePullSecrets`, set:

  * `AKS_ACR_ATTACH=true`
    Pulumi will assign the `AcrPull` role to the AKS kubelet identity.

---

## SKU Model and Cost Characteristics

* Available SKUs: **Basic**, **Standard**, **Premium**
* Pricing model:

  * Fixed monthly price per SKU tier
  * Additional per-GB charges only apply beyond included storage

Approximate operational characteristics:

* **Basic**

  * Lowest cost
  * Included storage ≈ 10 GB
  * Limited throughput and feature set
  * Suitable only for small, disposable development environments

* **Standard**

  * Balanced cost and capacity
  * Included storage ≈ 100 GB
  * Adequate for most production workloads
  * Not reliably available in all regions or subscription types

* **Premium**

  * Highest cost
  * Required for private endpoints, geo-replication, and customer-managed keys
  * Most reliable SKU across regions

**Recommendation:**
Use **Premium** in India regions and whenever private networking is required.
Use **Standard** elsewhere for production.
Use **Basic** only for short-lived development environments.

---

## Environment Variables (Authoritative)

```bash
export ACR_NAME=acr49251
export ACR_RESOURCE_RG=rg-acr-b1e221f4
export ACR_LOCATION="${AKS_LOCATION:-${AZURE_LOCATION:-eastus}}"
export ACR_SKU=Standard
export AKS_ACR_ATTACH=true
export ACR_PRIVATE_ENDPOINT_ENABLED=false
export ACR_PUBLIC_ACCESS=true
export ACR_ADMIN_ENABLED=false
export ACR_RETENTION_DAYS=30
```

**Variable definitions:**

* `ACR_NAME`
  Registry name. Required.
  Must be 5–50 characters, lowercase letters and numbers only.

* `ACR_RESOURCE_RG`
  Resource group containing the registry or where it should be created.
  When set to an existing registry’s RG, Pulumi switches to reuse mode.

* `ACR_LOCATION`
  Region used only when creating a new registry.
  Ignored when reusing an existing registry.

* `ACR_SKU`
  One of `Basic`, `Standard`, `Premium`.
  Platform logic may coerce `Standard` → `Premium` in unsupported regions.

* `AKS_ACR_ATTACH`
  When `true`, Pulumi assigns `AcrPull` to the AKS kubelet identity.

* `ACR_PRIVATE_ENDPOINT_ENABLED`
  Requires `ACR_SKU=Premium`. Additional DNS and networking configuration is required.

* `ACR_PUBLIC_ACCESS`
  Must be `false` when using private-only access.

---

## Pulumi Ownership and Adoption Model

Pulumi determines registry behavior using an explicit lookup-first strategy:

1. Pulumi attempts:

   ```
   containerregistry.get_registry(
     resource_group_name=ACR_RESOURCE_RG,
     registry_name=ACR_NAME
   )
   ```

2. Outcomes:

   * **Registry found**

     * Pulumi reuses the registry
     * No create or replace occurs
     * SKU and location are preserved
     * RBAC assignments may still be created
   * **Registry not found**

     * Pulumi creates a new registry
     * Uses `ACR_LOCATION` and `ACR_SKU`

This model enforces explicit ownership and prevents accidental cross-resource-group adoption.

For known problematic regions, the platform intentionally coerces SKUs to avoid non-deterministic deployment failures.

---

## Operational Commands

List all registries with location and SKU:

```bash
az acr list --query "[].{name:name,resourceGroup:resourceGroup,location:location,sku:sku.name}" -o table
```

Inspect a specific registry:

```bash
az acr show --name "$ACR_NAME" --resource-group "$ACR_RESOURCE_RG" -o json
```

Check registry name availability:

```bash
az acr check-name --name "$ACR_NAME" -o json
```

Adopt an existing registry into Pulumi state:

```bash
pulumi import azure-native:containerregistry:Registry acrRegistry \
  /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.ContainerRegistry/registries/<acr-name>
```

Manually assign `AcrPull`:

```bash
az role assignment create \
  --assignee <kubeletPrincipalId> \
  --role AcrPull \
  --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.ContainerRegistry/registries/<acr-name>
```

---

## Common Errors and Deterministic Remediation

### `SkuNotSupported`

```
Status=400 Code="SkuNotSupported"
```

**Cause:** SKU not available in the selected region or subscription.

**Resolution:**

* Immediate: set `ACR_SKU=Premium`
* Permanent: reuse an existing registry in a supported region or rely on platform SKU coercion

---

### `az: 'list-skus' not recognized`

**Cause:** Incorrect or deprecated CLI command.

**Resolution:**
Use `az provider show` or `az rest`. Upgrade Azure CLI if necessary.

---

### `AuthorizationFailed` during role assignment

**Cause:** Executing identity lacks `roleAssignments/write`.

**Resolution:**
Run Pulumi as Owner or assign `AcrPull` manually.

---

### Pulumi `AttributeError: 'Registry' object has no attribute 'apply'`

**Cause:** Treating a resource object as an Output.

**Resolution:**
Export properties directly or wrap static values using `pulumi.Output.from_input`.
This issue is resolved in the platform codebase.

---

## Operational Best Practices

1. Always set `ACR_RESOURCE_RG` when reusing an existing registry.
2. Prefer reliability over cost: use `Premium` where SKU support is uncertain.
3. Import manually created registries into Pulumi state immediately.
4. Allow Pulumi to manage `AcrPull` when possible; otherwise assign manually.
5. Enable private endpoints only with `Premium` SKU and preplanned DNS.

---

## Recommended Workflows

### Reuse an existing registry

```bash
export ACR_NAME=acr49251
export ACR_RESOURCE_RG=rg-acr-b1e221f4
export AKS_ACR_ATTACH=true
make pulumi-up
```

### Create and own a new registry

```bash
unset ACR_RESOURCE_RG
export ACR_NAME=ragacr
export ACR_LOCATION=eastus
export ACR_SKU=Standard
make pulumi-up
```

### Resolve `SkuNotSupported` in India regions

```bash
export ACR_SKU=Premium
make pulumi-up
```

---

## Troubleshooting Checklist

* Is `ACR_NAME` valid and globally unique?
* Does `ACR_RESOURCE_RG` correctly identify the registry’s resource group?
* Is the selected SKU supported in the target region?
* Does the executing identity have permission to assign roles?

---

## Summary

Azure Container Registry is a tiered, region-sensitive service.
Deterministic behavior requires explicit ownership via `ACR_RESOURCE_RG`, conservative SKU selection in constrained regions, and formal adoption of manually created registries into Pulumi state.
