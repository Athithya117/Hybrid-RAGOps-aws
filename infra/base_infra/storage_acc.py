#!/usr/bin/env python3
"""
infra/base_infra/storage_acc.py

Env-driven Azure Resource bootstrapper (create/delete) - KEY-AUTH mode.

Changes from prior version:
- All blob/container operations use --auth-mode key + --account-key (Option A).
- UAI creation/cleanup removed (simplified bootstrap).
- Role assignment helpers retained for optional future use.
- Adds optional lifecycle policy application for backup container using env:
    BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS
    BACKUP_AZ_CONTAINER_RETENTION_DAYS

Required env:
  AZURE_SUBSCRIPTION_ID
  AZURE_STORAGE_ACCOUNT_NAME
  AZURE_CONTAINER or AZURE_DATA_CONTAINER

Optional env:
  AZURE_RESOURCE_GROUP_NAME (default: rg-e2e-rag)
  AZURE_LOCATION (default: centralindia)
  PULUMI_AZ_CONTAINER, BACKUP_AZ_CONTAINER
  AZURE_DELETE_ACCOUNT (0/1) - when deleting storage account instead of containers
  FORCE_DELETE (0/1) - skip interactive confirmation
  UAI_RAG_RW_NAME / UAI_RAG_RO_NAME (not used by default)

  

Usage:
  python infra/base_infra/storage_acc.py --create
  python infra/base_infra/storage_acc.py --delete
"""
from __future__ import annotations
import os
import re
import sys
import time
import json
import argparse
import subprocess
import tempfile
from typing import List, Tuple, Optional, Dict

# ---------- small logger ----------
def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

def info(msg: str) -> None:
    print(f"{now()} INFO: {msg}")

def warn(msg: str) -> None:
    print(f"{now()} WARN: {msg}")

def err(msg: str) -> None:
    print(f"{now()} ERROR: {msg}", file=sys.stderr)

def die(msg: str, code: int = 2) -> None:
    err(msg)
    sys.exit(code)

# ---------- runner ----------
def run(cmd: List[str], check: bool = True) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        die(f"Command not found: {cmd[0]}. Install Azure CLI (az).")
    out = (proc.stdout or "").strip()
    er = (proc.stderr or "").strip()
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout: {out}\nstderr: {er}")
    return proc.returncode, out, er

def az(*args: str) -> Tuple[int, str, str]:
    return run(["az", *args], check=True)

# ---------- env/config ----------
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "").strip() or None
AZURE_RESOURCE_GROUP_NAME = os.getenv("AZURE_RESOURCE_GROUP_NAME", "rg-e2e-rag").strip()
AZURE_LOCATION = os.getenv("AZURE_LOCATION", "centralindia").strip()
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip() or None
AZURE_ENDPOINT_SUFFIX = os.getenv("AZURE_ENDPOINT_SUFFIX", "core.windows.net").strip()
STORAGE_TIER = os.getenv("STORAGE_TIER", "LRS").strip()
# accept either AZURE_CONTAINER (common) or AZURE_DATA_CONTAINER (legacy in repo)
AZURE_CONTAINER = (os.getenv("AZURE_CONTAINER") or os.getenv("AZURE_DATA_CONTAINER") or "").strip() or None
PULUMI_AZ_CONTAINER = os.getenv("PULUMI_AZ_CONTAINER", "").strip() or None
BACKUP_AZ_CONTAINER = os.getenv("BACKUP_AZ_CONTAINER", "").strip() or None

# Deletion defaults preserved (match existing usage)
AZURE_DELETE_ACCOUNT = os.getenv("AZURE_DELETE_ACCOUNT", "1").strip().lower() in ("1", "true", "yes")
FORCE_DELETE = os.getenv("FORCE_DELETE", "1").strip().lower() in ("1", "true", "yes")

UAI_RW_NAME = os.getenv("UAI_RAG_RW_NAME", "uai-rag-rw").strip()
UAI_RO_NAME = os.getenv("UAI_RAG_RO_NAME", "uai-rag-ro").strip()

# Backup lifecycle envs (optional)
BACKUP_AZ_CONTAINER_RETENTION_DAYS = os.getenv("BACKUP_AZ_CONTAINER_RETENTION_DAYS", "").strip()
BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS = os.getenv("BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS", "").strip()
BACKUP_PREFIX = os.getenv("BACKUP_PREFIX", "qdrant").strip()

def normalize_sku(token: str) -> str:
    t = token.strip().upper()
    map_simple = {"LRS":"Standard_LRS","ZRS":"Standard_ZRS","GRS":"Standard_GRS","RAGRS":"Standard_RAGRS","GZRS":"Standard_GZRS","RAGZRS":"Standard_RAGZRS"}
    if t in map_simple:
        return map_simple[t]
    if t.startswith("STANDARD_") or t.startswith("PREMIUM_"):
        return token
    return "Standard_" + token

SKU_NAME = normalize_sku(STORAGE_TIER)

# ---------- validations ----------
def validate_env_minimum():
    missing = []
    if not AZURE_SUBSCRIPTION_ID:
        missing.append("AZURE_SUBSCRIPTION_ID")
    if not AZURE_STORAGE_ACCOUNT_NAME:
        missing.append("AZURE_STORAGE_ACCOUNT_NAME")
    if not AZURE_CONTAINER:
        missing.append("AZURE_CONTAINER or AZURE_DATA_CONTAINER")
    if missing:
        die("Missing required environment variables: " + ", ".join(missing))
    if not re.fullmatch(r"[a-z0-9]{3,24}", AZURE_STORAGE_ACCOUNT_NAME):
        die("AZURE_STORAGE_ACCOUNT_NAME must be 3-24 chars, lowercase letters and numbers only.")
    # validate optional numeric lifecycle envs (if provided)
    if BACKUP_AZ_CONTAINER_RETENTION_DAYS:
        if not BACKUP_AZ_CONTAINER_RETENTION_DAYS.isdigit():
            die("BACKUP_AZ_CONTAINER_RETENTION_DAYS must be an integer number of days")
    if BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS:
        if not BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS.isdigit():
            die("BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS must be an integer number of days")
validate_env_minimum()

# ---------- azure helpers ----------
def ensure_subscription():
    try:
        az("account", "set", "--subscription", AZURE_SUBSCRIPTION_ID)
        info(f"Azure subscription set to {AZURE_SUBSCRIPTION_ID}")
    except Exception as e:
        die(f"Failed to set subscription {AZURE_SUBSCRIPTION_ID}: {e}")

def rg_exists(name: str) -> bool:
    rc, out, _ = run(["az","group","exists","--name",name], check=True)
    return out.strip().lower() == "true"

def create_resource_group(name: str, location: str):
    if rg_exists(name):
        info(f"Resource group {name} already exists.")
        return
    info(f"Creating resource group {name} in {location} ...")
    az("group","create","--name",name,"--location",location)
    info("Resource group created.")

def storage_account_show(name: str, rg: str) -> bool:
    rc, out, err = run(["az","storage","account","show","--name",name,"--resource-group",rg,"-o","json"], check=False)
    return rc == 0

def storage_account_check_name(name: str) -> Tuple[bool,str]:
    rc,out,_ = run(["az","storage","account","check-name","--name",name,"-o","json"], check=True)
    try:
        obj = json.loads(out)
        return bool(obj.get("nameAvailable", False)), obj.get("message","")
    except Exception:
        return False, "name-check-failed"

def create_storage_account(name: str, rg: str, location: str, sku: str):
    if storage_account_show(name, rg):
        info(f"Storage account {name} already exists in resource group {rg}.")
        return
    available, msg = storage_account_check_name(name)
    if not available:
        warn(f"Storage account name '{name}' may not be available: {msg}")
        warn("Proceeding; creation may fail if name is taken.")
    info(f"Creating storage account {name} in {rg} ({location}) sku={sku} ...")
    try:
        az("storage","account","create","--name",name,"--resource-group",rg,"--location",location,"--sku",sku,"--kind","StorageV2","--https-only","true")
    except Exception as e:
        die(f"Failed to initiate storage account creation: {e}")
    timeout = 600
    interval = 5
    elapsed = 0
    while elapsed < timeout:
        if storage_account_show(name, rg):
            info("Storage account is now active.")
            return
        time.sleep(interval); elapsed += interval
    die("Storage account did not appear within timeout; check portal/permissions.")

def get_storage_account_key(name: str, rg: str) -> str:
    info("Fetching storage account key via az ...")
    try:
        rc,out,_ = run(["az","storage","account","keys","list","--resource-group",rg,"--account-name",name,"--query","[0].value","-o","tsv"], check=True)
        key = out.strip()
        if not key:
            raise RuntimeError("empty key returned")
        info("Successfully retrieved storage account key.")
        return key
    except Exception as e:
        die(f"Failed to get storage account key: {e}")

def get_storage_account_resource(name: str, rg: str) -> dict:
    rc, out, _ = run(["az","storage","account","show","--name",name,"--resource-group",rg,"-o","json"], check=True)
    return json.loads(out)

# NOTE: UAI helpers intentionally removed to simplify bootstrap logic.

def assign_storage_rbac(principal_id: str, storage_scope: str, role: str, max_retries: int = 6, initial_delay: int = 3) -> None:
    """
    Assign role to principalId at storage scope using object-id + explicit principal type.
    Retries to tolerate Azure replication delay. Idempotent-ish: if already present, returns cleanly.
    """
    if not principal_id:
        raise RuntimeError("assign_storage_rbac called with empty principal_id")
    info(f"Assigning role '{role}' to principal '{principal_id}' on scope '{storage_scope}' (max_retries={max_retries}) ...")
    cmd = [
        "az", "role", "assignment", "create",
        "--assignee-object-id", principal_id,
        "--assignee-principal-type", "ServicePrincipal",
        "--role", role,
        "--scope", storage_scope,
        "-o", "json"
    ]
    for attempt in range(1, max_retries + 1):
        rc, out, err = run(cmd, check=False)
        stderr = (err or "").strip()
        stdout = (out or "").strip()
        if rc == 0:
            info(f"Role '{role}' assigned to principal '{principal_id}' (attempt {attempt}).")
            return
        low = stderr.lower()
        if "already exists" in low or "exists" in low:
            info(f"Role '{role}' already assigned to principal '{principal_id}'.")
            return
        if "principalnotfound" in low.replace(" ", "") or "principal not found" in low:
            warn(f"Principal not found yet (attempt {attempt}/{max_retries}). Will retry after backoff.")
            time.sleep(initial_delay * (2 ** (attempt - 1)))
            continue
        if ("throttling" in low or "too many requests" in low or "temporarily unavailable" in low) and attempt < max_retries:
            warn(f"Transient error assigning role (attempt {attempt}/{max_retries}): {stderr or stdout}. Retrying.")
            time.sleep(initial_delay * (2 ** (attempt - 1)))
            continue
        warn(f"Role assignment failed (attempt {attempt}/{max_retries}). rc={rc} stderr={stderr or stdout}")
        break
    info("Final attempt: trying role assignment using '--assignee' fallback (may accept object id or service principal name).")
    fallback_cmd = [
        "az", "role", "assignment", "create",
        "--assignee", principal_id,
        "--assignee-principal-type", "ServicePrincipal",
        "--role", role,
        "--scope", storage_scope,
        "-o", "json"
    ]
    rc2, out2, err2 = run(fallback_cmd, check=False)
    stderr2 = (err2 or "").strip().lower()
    if rc2 == 0 or "already exists" in stderr2 or "exists" in stderr2:
        info(f"Role '{role}' assigned to principal '{principal_id}' via fallback.")
        return
    warn(f"Final role-assignment attempt failed. rc={rc2} stderr={err2 or out2}")

def remove_role_assignments(principal_id: str, storage_scope: str, role: Optional[str] = None):
    info(f"Removing role assignments for principal '{principal_id}' on '{storage_scope}' (role={role}) ...")
    cmd = ["az","role","assignment","list","--assignee-object-id", principal_id, "--scope", storage_scope, "-o", "json"]
    rc, out, err = run(cmd, check=False)
    if rc != 0:
        warn(f"Failed to list role assignments for principal '{principal_id}': {err or out}")
        return
    try:
        arr = json.loads(out)
    except Exception:
        warn("Failed to parse role assignments list JSON; skipping removals.")
        return
    for a in arr:
        a_role = a.get("roleDefinitionName")
        a_id = a.get("id")
        if role and a_role != role:
            continue
        if not a_id:
            continue
        rc2, out2, err2 = run(["az","role","assignment","delete","--ids", a_id], check=False)
        if rc2 == 0:
            info(f"Removed role assignment '{a_role}' (id={a_id}).")
        else:
            warn(f"Failed removing role assignment id={a_id}: {err2 or out2}")

# ---------- storage ops using KEY auth (Option A) ----------
def list_containers(account: str, key: str) -> List[dict]:
    rc, out, err = run([
        "az","storage","container","list",
        "--account-name", account,
        "--auth-mode", "key",
        "--account-key", key,
        "-o", "json"
    ], check=False)
    if rc != 0:
        warn(f"Failed to list containers using key auth: {err or out}")
        return []
    try:
        return json.loads(out)
    except Exception:
        warn("Failed to parse container list JSON")
        return []

def create_container(account: str, key: str, container: str):
    info(f"Ensuring container '{container}' exists in account '{account}' (key auth) ...")
    rc, out, err = run([
        "az","storage","container","create",
        "--name", container,
        "--account-name", account,
        "--auth-mode", "key",
        "--account-key", key,
        "-o", "none"
    ], check=False)
    if rc == 0:
        info(f"Container '{container}' ensured.")
        return
    stderr = (err or "").lower()
    if "already exists" in stderr or "exists" in stderr:
        info(f"Container '{container}' already exists.")
        return
    die(f"Failed to create container '{container}': {err or out}")

def container_blob_count(account: str, key: str, container: str) -> Optional[int]:
    rc, out, err = run([
        "az","storage","blob","list",
        "--container-name", container,
        "--account-name", account,
        "--auth-mode", "key",
        "--account-key", key,
        "-o", "json"
    ], check=False)
    if rc != 0:
        warn(f"Unable to list blobs in {container}: {err or out}")
        return None
    try:
        arr = json.loads(out)
        return len(arr)
    except Exception:
        warn("Failed to parse blobs JSON")
        return None

def delete_container(account: str, key: str, container: str):
    info(f"Deleting container '{container}' (key auth) ...")
    rc, out, err = run([
        "az","storage","container","delete",
        "--name", container,
        "--account-name", account,
        "--auth-mode", "key",
        "--account-key", key
    ], check=False)
    if rc == 0:
        info(f"Deleted container '{container}'.")
        return
    stderr = (err or "").lower()
    if "was not found" in stderr or "not found" in stderr:
        info(f"Container '{container}' not found; nothing to delete.")
        return
    warn(f"Failed deleting container '{container}': {err or out}")

# ---------- lifecycle policy helpers ----------
def _build_lifecycle_policy_json(prefix: str, cool_after: Optional[int], delete_after: Optional[int]) -> Dict:
    """
    Build a management-policy JSON structure for the given prefix.
    prefix: container/prefix/ e.g. "backups/qdrant/"
    """
    rule: Dict = {
        "enabled": True,
        "name": "backup-tier-and-delete",
        "type": "Lifecycle",
        "definition": {
            "filters": {
                "blobTypes": ["blockBlob"],
                "prefixMatch": [prefix]
            },
            "actions": {
                "baseBlob": {}
            }
        }
    }
    base_blob_actions = rule["definition"]["actions"]["baseBlob"]
    if cool_after is not None:
        base_blob_actions["tierToCool"] = {"daysAfterModificationGreaterThan": cool_after}
    if delete_after is not None:
        base_blob_actions["delete"] = {"daysAfterModificationGreaterThan": delete_after}
    return {"rules": [rule]}

def apply_lifecycle_policy(account: str, rg: str, container: str, prefix_segment: str, cool_after_days: Optional[int], retention_days: Optional[int]) -> None:
    """
    Apply lifecycle policy scoped to container/prefix_segment/.
    This writes a temporary JSON file and calls az storage account management-policy create.
    Non-fatal: on failure we log a warning and continue.
    """
    if cool_after_days is None and retention_days is None:
        info("No lifecycle policy requested (both cool_after_days and retention_days are empty); skipping.")
        return
    # ensure prefix ends with slash and is composed as container/prefix/
    if not prefix_segment:
        prefix = f"{container}/"
    else:
        prefix = f"{container}/{prefix_segment.strip().strip('/')}/"
    policy = _build_lifecycle_policy_json(prefix, cool_after_days, retention_days)
    info(f"Applying lifecycle policy for storage account '{account}' in RG '{rg}' with prefix '{prefix}' (cool_after={cool_after_days} delete_after={retention_days})")
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tf:
            tf.write(json.dumps(policy, indent=2))
            tmp_path = tf.name
        # az storage account management-policy create will create or update
        rc, out, err_txt = run([
            "az", "storage", "account", "management-policy", "create",
            "--resource-group", rg,
            "--account-name", account,
            "--policy", tmp_path
        ], check=False)
        if rc != 0:
            warn(f"Failed to apply lifecycle policy via az CLI: {err_txt or out}")
        else:
            info("Lifecycle policy applied successfully.")
    except Exception as e:
        warn(f"Applying lifecycle policy failed: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ---------- deletion helpers ----------
def delete_storage_account(account: str, rg: str, wait_poll: bool = False, poll_timeout: int = 300):
    info(f"Initiating deletion of storage account {account} in {rg} ...")
    try:
        az("storage","account","delete","--name",account,"--resource-group",rg,"--yes")
        info("Storage account deletion initiated.")
    except Exception as e:
        die(f"Failed to initiate storage account deletion: {e}")
    if wait_poll:
        info("Waiting for storage account to disappear (polling)...")
        elapsed = 0
        interval = 5
        while elapsed < poll_timeout:
            if not storage_account_show(account, rg):
                info("Storage account no longer present.")
                return
            time.sleep(interval); elapsed += interval
        warn("Storage account still present after polling timeout; deletion may be in progress server-side.")

# ---------- high-level ops ----------
def do_create():
    ensure_subscription()
    create_resource_group(AZURE_RESOURCE_GROUP_NAME, AZURE_LOCATION)
    create_storage_account(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME, AZURE_LOCATION, SKU_NAME)

    sa_obj = get_storage_account_resource(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME)
    storage_id = sa_obj.get("id")
    if not storage_id:
        die("Failed to read storage account resource id; aborting.")

    # NOTE: UAI creation and role assignment logic intentionally removed.

    # Fetch account key and create containers using key auth
    key = get_storage_account_key(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME)
    create_container(AZURE_STORAGE_ACCOUNT_NAME, key, AZURE_CONTAINER)
    if PULUMI_AZ_CONTAINER:
        create_container(AZURE_STORAGE_ACCOUNT_NAME, key, PULUMI_AZ_CONTAINER)
    if BACKUP_AZ_CONTAINER:
        create_container(AZURE_STORAGE_ACCOUNT_NAME, key, BACKUP_AZ_CONTAINER)

    info("CREATED/ENSURED containers:")
    out_containers = [AZURE_CONTAINER]
    if PULUMI_AZ_CONTAINER: out_containers.append(PULUMI_AZ_CONTAINER)
    if BACKUP_AZ_CONTAINER: out_containers.append(BACKUP_AZ_CONTAINER)
    for c in out_containers:
        info(f" - {c}")

    # Apply lifecycle policy to backup container (if requested via env)
    try:
        retention = int(BACKUP_AZ_CONTAINER_RETENTION_DAYS) if BACKUP_AZ_CONTAINER_RETENTION_DAYS else None
    except ValueError:
        retention = None
    try:
        cool_after = int(BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS) if BACKUP_AZ_CONTAINER_COOL_AFTER_DAYS else None
    except ValueError:
        cool_after = None

    if BACKUP_AZ_CONTAINER and (retention is not None or cool_after is not None):
        # Use backup prefix (e.g. qdrant) if present; policy will apply to blobs under container/prefix/
        apply_lifecycle_policy(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME, BACKUP_AZ_CONTAINER, BACKUP_PREFIX, cool_after, retention)
    else:
        info("No backup lifecycle policy configured or BACKUP_AZ_CONTAINER not set; skipping lifecycle application.")

def do_delete():
    ensure_subscription()

    if AZURE_DELETE_ACCOUNT:
        info("Pre-delete inventory: listing containers and sample counts (may be slow for large containers)...")
        key = get_storage_account_key(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME)
        containers = list_containers(AZURE_STORAGE_ACCOUNT_NAME, key)
        if containers:
            info(f"Found {len(containers)} containers in '{AZURE_STORAGE_ACCOUNT_NAME}':")
            for c in containers:
                name = c.get("name")
                cnt = container_blob_count(AZURE_STORAGE_ACCOUNT_NAME, key, name)
                info(f"  - {name} (blob count: {cnt if cnt is not None else 'unknown'})")
        else:
            info("No containers found or failed to list containers (proceeding).")

        if not FORCE_DELETE:
            warn("You are about to DELETE the entire storage account and ALL its containers/blobs.")
            confirm = input("Type 'yes' to confirm: ")
            if confirm.strip().lower() != "yes":
                info("Aborted by user.")
                return

        # NOTE: UAI cleanup removed (no UAIs were created)

        delete_storage_account(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME, wait_poll=True, poll_timeout=300)
        return

    # containers-only deletion path (key-auth required)
    if not storage_account_show(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME):
        warn(f"Storage account {AZURE_STORAGE_ACCOUNT_NAME} not found in resource group {AZURE_RESOURCE_GROUP_NAME}; cannot delete containers.")
        return

    key = get_storage_account_key(AZURE_STORAGE_ACCOUNT_NAME, AZURE_RESOURCE_GROUP_NAME)

    planned = [c for c in [AZURE_CONTAINER, PULUMI_AZ_CONTAINER, BACKUP_AZ_CONTAINER] if c]
    info("Pre-delete container inventory (planned deletions):")
    for c in planned:
        cnt = container_blob_count(AZURE_STORAGE_ACCOUNT_NAME, key, c)
        info(f"  - {c}: {cnt if cnt is not None else 'unknown'} blobs")

    if not FORCE_DELETE:
        warn("Deleting containers will remove all blobs inside them.")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            info("Aborted by user.")
            return

    for c in planned:
        delete_container(AZURE_STORAGE_ACCOUNT_NAME, key, c)

    info("Container deletion attempts complete.")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Env-driven Azure Storage Account + Containers bootstrapper (key auth).")
    gp = p.add_mutually_exclusive_group(required=True)
    gp.add_argument("--create", action="store_true", help="Create resources.")
    gp.add_argument("--delete", action="store_true", help="Delete (containers or storage account controlled via env AZURE_DELETE_ACCOUNT).")
    return p.parse_args()

def main():
    args = parse_args()
    try:
        if args.create:
            do_create()
        elif args.delete:
            do_delete()
    except KeyboardInterrupt:
        die("Interrupted by user", 130)
    except Exception as e:
        die(f"Operation failed: {e}")

if __name__ == "__main__":
    main()
