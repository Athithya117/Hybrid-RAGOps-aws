# AKS Cluster Architecture

This document describes the Azure Kubernetes Service (AKS) cluster architecture used by the platform.
The design prioritizes deterministic networking, workload isolation, predictable scaling, and
cloudflared tunnel compatibility.

---

## High-Level Design Goals

- Deterministic outbound egress (required for cloudflared tunnels)
- Explicit node pool separation by workload class
- VNET-native pod networking (Azure CNI)
- Minimal but sufficient configuration surface
- Safe defaults that work in staging and production

---

## Networking Model

- Network Plugin: Azure CNI
- Network Policy: Calico
- Outbound Type: userAssignedNATGateway
- Egress Control: NAT Gateway with static public IP
- Service CIDR: 10.0.0.0/16
- DNS Service IP: 10.0.0.10

This model ensures:
- Stable source IPs for outbound traffic
- No SNAT churn under load
- Long-lived TCP/TLS connections remain intact (cloudflared)

---

## Virtual Network Layout

| Component | CIDR |
|----------|------|
| VNET | 10.1.0.0/16 |
| AKS Subnet | 10.1.1.0/24 |
| App Gateway / L7 Subnet | 10.1.2.0/24 |

The App Gateway subnet is reserved even if unused, to avoid future VNET surgery.

---

## Node Pool Strategy

Node pools are intentionally separated by workload characteristics.
This prevents noisy neighbors, improves scheduling predictability, and allows
per-pool scaling strategies.

### Node Pool Matrix

| Node Pool               | Purpose / Workloads                                                                 | Hard Requirements                                                                 | Scheduling Rules                                                                 | Staging VM                 | Production VM                                      |
|-------------------------|--------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------|----------------------------|---------------------------------------------------|
| System (AKS Core)       | kube-system pods, CNI, CoreDNS, CSI, metrics-server, control-plane agents            | Must exist at cluster create, ≥1 node, no scale-to-zero, predictable CPU/RAM      | mode=System, no user workloads, optional taint node-role.kubernetes.io/system=true:NoSchedule | Standard_D4ds_v4           | Standard_D4ds_v5 / Standard_D8ds_v5               |
| Balanced (General)      | API gateway, frontend, orchestration                                                 | Stateless, fast scale, low latency                                                 | No taints, default priority                                                       | Standard_D2s_v5            | Standard_D4s_v5                                   |
| CPU-Heavy (Models)      | Embeddings, rerankers, tokenizers, CPU inference                                     | Sustained CPU, predictable latency                                                 | Taint: workload=cpu-heavy:NoSchedule                                             | Standard_F4s_v2            | Standard_F8s_v2 (AVX2, not AVX-512)               |
| Qdrant (Vector DB)      | Vector storage, HNSW index, WAL, RocksDB                                              | High RAM, local NVMe, zero interference                                             | Taint: workload=qdrant:NoSchedule, one pod per node                              | Standard_D4dsv5            | Standard_E8ds_v5 / Standard_E16ds_v5              |

---

## System Node Pool Rules (Non-Negotiable)

- Created inside ManagedCluster
- Cannot be removed
- Cannot scale to zero
- Must not host user workloads
- Must be stable across upgrades

Failure to meet these rules results in:
- AKS create/update failures
- Control plane instability
- Upgrade deadlocks

---

## Outbound Traffic Policy

| Outbound Type | Allowed | Notes |
|--------------|--------|------|
| userAssignedNATGateway | Yes | Required for production and cloudflared |
| managedNATGateway | Limited | Acceptable for small clusters |
| loadBalancer | No | SNAT churn, tunnel instability |

---

## Kubernetes Versioning

- Kubernetes version is explicitly pinned
- Minor upgrades are manual and intentional
- No automatic adoption of newest AKS versions

---

## Summary

This AKS design is cloudflared-safe, deterministic, and operationally boring.
Changes should only be made with a clear understanding of AKS control-plane constraints.
