#!/usr/bin/env python3
"""
uai_key_vault_secrets.py

Idempotent helper to:
 - create (or reuse) two User Assigned Identities (UAI): rw + ro
 - create or reuse an Azure Key Vault (RBAC model)
 - assign Key Vault RBAC roles (Key Vault Secrets User/Officer) to UAIs
 - optionally assign Storage Blob Data roles on storage account or container to UAIs
 - create secrets in the Key Vault from:
     - KEYVAULT_SECRETS_JSON env var (JSON map), or
     - env vars that start with KV_SECRET_<NAME>=value
 - prints recommended export block (UAI client IDs, principals, KV URI)

Required ENV:
  AZURE_SUBSCRIPTION_ID
  AZURE_RESOURCE_GROUP_NAME
  AZURE_LOCATION
  KEYVAULT_NAME

Optional:
  UAI_RW_NAME (default uai-rag-rw)
  UAI_RO_NAME (default uai-rag-ro)
  KEYVAULT_SECRETS_JSON (JSON string: {"DB_PASS":"s3cret", ...})
  and/or environment variables prefixed KV_SECRET_
  ASSIGN_STORAGE_SCOPE (full ARM scope to storage account or container to assign blob roles)
  FORCE (1 to skip prompts)

Notes:
 - This script uses `az` CLI. Ensure `az login` and you have permission to assign roles.
 - Role GUIDs used are documented built-in GUIDs for Storage Blob Data and Key Vault roles. :contentReference[oaicite:4]{index=4}
"""
from __future__ import annotations
import os
import sys
import json
import time
import subprocess
import argparse
from typing import Dict, Tuple, Optional, List

# -------- logger -------
def now(): return time.strftime("%Y-%m-%dT%H:%M:%S%z")
def info(msg): print(f"{now()} INFO: {msg}")
def warn(msg): print(f"{now()} WARN: {msg}")
def die(msg, code=2):
    print(f"{now()} ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

# -------- run helper -------
def run(cmd: List[str], check=True) -> Tuple[int,str,str]:
    info("RUN: " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if check and proc.returncode != 0:
        die(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nstdout: {out}\nstderr: {err}")
    return proc.returncode, out, err

def az(*args):
    return run(["az", *args], check=True)

# -------- config ----------
AZ_SUBS = os.getenv("AZURE_SUBSCRIPTION_ID") or os.getenv("AZ_SUBSCRIPTION_ID")
AZ_RG = os.getenv("AZURE_RESOURCE_GROUP_NAME")
AZ_LOC = os.getenv("AZURE_LOCATION", "eastus")
KEYVAULT_NAME = os.getenv("KEYVAULT_NAME")
UAI_RW = os.getenv("UAI_RW_NAME", "uai-rag-rw")
UAI_RO = os.getenv("UAI_RO_NAME", "uai-rag-ro")
FORCE = os.getenv("FORCE", "0") in ("1","true","True")
ASSIGN_STORAGE_SCOPE = os.getenv("ASSIGN_STORAGE_SCOPE")  # optional ARM scope for storage (account or container)
KEYVAULT_SECRETS_JSON = os.getenv("KEYVAULT_SECRETS_JSON")

# Built-in role GUIDs (stable; documented)
ROLE_STORAGE_BLOB_CONTRIB = "ba92f5b4-2d11-453d-a403-e96b0029c9fe"   # Storage Blob Data Contributor
ROLE_STORAGE_BLOB_READER = "2a2b9908-6ea1-4ae2-8e65-a410df84e7d1"    # Storage Blob Data Reader
ROLE_KV_SECRETS_OFFICER = "b86a8fe4-44ce-4948-aee5-eccb2c155cd7"    # Key Vault Secrets Officer
ROLE_KV_SECRETS_USER = "4633458b-17de-408a-b874-0445c86b69e6"       # Key Vault Secrets User

def validate_min():
    missing=[]
    if not AZ_SUBS: missing.append("AZURE_SUBSCRIPTION_ID")
    if not AZ_RG: missing.append("AZURE_RESOURCE_GROUP_NAME")
    if not KEYVAULT_NAME: missing.append("KEYVAULT_NAME")
    if missing:
        die("Missing required envs: " + ", ".join(missing))

# --------- helpers ----------
def ensure_subscription():
    az("account","set","--subscription",AZ_SUBS)
    info(f"Subscription set to {AZ_SUBS}")

def rg_exists(name: str) -> bool:
    rc,out,_ = run(["az","group","exists","--name",name], check=False)
    return out.strip().lower() == "true"

def ensure_resource_group(name: str, location: str):
    if rg_exists(name):
        info(f"Resource group {name} exists.")
        return
    info(f"Creating resource group {name} ...")
    az("group","create","--name",name,"--location",location,"-o","json")

def uai_exists(name: str, rg: str) -> Optional[dict]:
    rc,out,_ = run(["az","identity","show","--name",name,"--resource-group",rg,"-o","json"], check=False)
    if rc == 0 and out:
        return json.loads(out)
    return None

def ensure_uai(name: str, rg: str, location: str) -> dict:
    existing = uai_exists(name, rg)
    if existing:
        info(f"Using existing UAI {name}")
        return existing
    info(f"Creating UAI {name} ...")
    rc,out,_ = run(["az","identity","create","--name",name,"--resource-group",rg,"--location",location,"-o","json"])
    return json.loads(out)

def keyvault_exists(name: str, rg: str) -> Optional[dict]:
    rc,out,_ = run(["az","keyvault","show","--name",name,"--resource-group",rg,"-o","json"], check=False)
    if rc == 0 and out:
        return json.loads(out)
    return None

def create_or_get_keyvault(name: str, rg: str, location: str) -> dict:
    kv = keyvault_exists(name, rg)
    if kv:
        info(f"Using existing Key Vault {name}")
        return kv
    # create with RBAC enabled (default) and standard sku
    info(f"Creating Key Vault {name} (RBAC permission model) ...")
    # Don't pass flags that older CLI versions might not accept - keep to stable flags
    rc,out,err = run([
        "az","keyvault","create",
        "--name", name,
        "--resource-group", rg,
        "--location", location,
        "--sku", "standard",
        "--enable-rbac-authorization",
        "-o","json"
    ], check=True)
    return json.loads(out)

def assign_role_to_principal(principal_object_id: str, scope: str, role_definition_id: str, max_retries=6):
    """
    Uses az role assignment create --assignee-object-id ... with retries to handle propagation delays.
    role_definition_id may be a GUID (role id) - az accepts either name or full id.
    """
    if not principal_object_id:
        die("principal_object_id empty for role assignment")
    cmd = [
        "az","role","assignment","create",
        "--assignee-object-id", principal_object_id,
        "--assignee-principal-type", "ServicePrincipal",
        "--role", role_definition_id,
        "--scope", scope,
        "-o","json"
    ]
    for attempt in range(1, max_retries+1):
        rc,out,err = run(cmd, check=False)
        stderr = (err or "").lower()
        if rc == 0:
            info(f"Assigned role {role_definition_id} to principal {principal_object_id} on {scope}")
            return
        if "already exists" in stderr or "exists" in stderr:
            info(f"Role assignment already present for principal {principal_object_id} on {scope}")
            return
        if "principalnotfound" in stderr.replace(" ", "") or "principal not found" in stderr:
            warn(f"Principal not ready yet (attempt {attempt}/{max_retries}), retrying...")
            time.sleep(3 * attempt)
            continue
        if attempt < max_retries:
            warn(f"Transient error assigning role (attempt {attempt}/{max_retries}): {err or out}. Retrying.")
            time.sleep(2 ** attempt)
            continue
        die(f"Role assignment failed finally: {err or out}")
    die("Role assignment attempts exhausted")

def put_secret_to_kv(vault_name: str, name: str, value: str):
    if value is None:
        warn(f"Skipping null secret {name}")
        return
    # az keyvault secret set will upsert the secret
    az("keyvault","secret","set","--vault-name", vault_name, "--name", name, "--value", value, "-o", "json")
    info(f"Set secret '{name}' in vault '{vault_name}'")

def gather_secrets_from_env() -> Dict[str,str]:
    out = {}
    # explicit JSON env
    if KEYVAULT_SECRETS_JSON:
        try:
            parsed = json.loads(KEYVAULT_SECRETS_JSON)
            if isinstance(parsed, dict):
                for k,v in parsed.items():
                    out[str(k)] = str(v)
        except Exception as e:
            die("Unable to parse KEYVAULT_SECRETS_JSON: " + str(e))
    # KV_SECRET_ prefix
    for k,v in os.environ.items():
        if not k.startswith("KV_SECRET_"):
            continue
        name = k[len("KV_SECRET_"):]
        if not name:
            continue
        out[name] = v
    return out

# -------- high level flow ----------
def do_create():
    validate_min()
    ensure_subscription()
    ensure_resource_group(AZ_RG, AZ_LOC)

    # UAIs
    uai_rw = ensure_uai(UAI_RW, AZ_RG, AZ_LOC)
    uai_ro = ensure_uai(UAI_RO, AZ_RG, AZ_LOC)

    # Key Vault
    kv = create_or_get_keyvault(KEYVAULT_NAME, AZ_RG, AZ_LOC)
    kv_uri = kv.get("properties", {}).get("vaultUri") or kv.get("properties", {}).get("vaultUri") or kv.get("vaultUri") or (f"https://{KEYVAULT_NAME}.vault.azure.net/")

    # Assign Key Vault roles (data plane) to UAIs at the Key Vault scope
    kv_scope = f"/subscriptions/{AZ_SUBS}/resourceGroups/{AZ_RG}/providers/Microsoft.KeyVault/vaults/{KEYVAULT_NAME}"
    assign_role_to_principal(uai_rw.get("principalId"), kv_scope, ROLE_KV_SECRETS_OFFICER)
    assign_role_to_principal(uai_ro.get("principalId"), kv_scope, ROLE_KV_SECRETS_USER)

    # Optionally assign storage roles if requested
    if ASSIGN_STORAGE_SCOPE:
        info(f"Assigning Storage roles on {ASSIGN_STORAGE_SCOPE}")
        assign_role_to_principal(uai_rw.get("principalId"), ASSIGN_STORAGE_SCOPE, ROLE_STORAGE_BLOB_CONTRIB)
        assign_role_to_principal(uai_ro.get("principalId"), ASSIGN_STORAGE_SCOPE, ROLE_STORAGE_BLOB_READER)

    # Gather secrets and set them
    secrets = gather_secrets_from_env()
    if not secrets:
        info("No secrets found in KEYVAULT_SECRETS_JSON or KV_SECRET_* envs. Skipping secret writes.")
    else:
        for name, val in secrets.items():
            put_secret_to_kv(KEYVAULT_NAME, name, val)

    # Print exports
    print("### Copy-paste exports ###")
    print(f"export KEYVAULT_NAME={KEYVAULT_NAME}")
    print(f"export KEYVAULT_URI={kv_uri}")
    print(f"export UAI_RAG_RW_CLIENT_ID={uai_rw.get('clientId')}")
    print(f"export UAI_RAG_RW_PRINCIPAL_ID={uai_rw.get('principalId')}")
    print(f"export UAI_RAG_RO_CLIENT_ID={uai_ro.get('clientId')}")
    print(f"export UAI_RAG_RO_PRINCIPAL_ID={uai_ro.get('principalId')}")
    if ASSIGN_STORAGE_SCOPE:
        print(f"# Storage role assigned at scope: {ASSIGN_STORAGE_SCOPE}")
    print("### end ###")

def do_delete():
    ensure_subscription()
    if not FORCE:
        confirm = input(f"Delete Key Vault {KEYVAULT_NAME}? This will remove secrets. Type 'yes' to proceed: ")
        if confirm.strip().lower() != "yes":
            info("Aborted.")
            return
    # delete vault - note: if purge protection enabled, deletion may be blocked
    run(["az","keyvault","delete","--name", KEYVAULT_NAME, "--resource-group", AZ_RG], check=False)
    # delete UAIs
    run(["az","identity","delete","--name", UAI_RW, "--resource-group", AZ_RG], check=False)
    run(["az","identity","delete","--name", UAI_RO, "--resource-group", AZ_RG], check=False)
    info("Delete requests issued (may take time to fully remove resources).")

# -------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--create", action="store_true")
    g.add_argument("--delete", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        if args.create:
            do_create()
        elif args.delete:
            do_delete()
    except KeyboardInterrupt:
        die("Interrupted", 130)
    except Exception as e:
        die(f"Failed: {e}")

if __name__ == "__main__":
    main()
