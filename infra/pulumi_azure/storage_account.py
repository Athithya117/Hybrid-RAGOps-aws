#!/usr/bin/env python3
"""
storage_account.py

Idempotent helper to ensure:
 - Resource Group exists (reuse/create)
 - Storage Account exists (reuse/create)
 - Blob containers exist (create if missing)
 - Prints environment variable export block for subsequent steps.

Required ENV (or pass via CLI): 
  AZURE_SUBSCRIPTION_ID
  AZURE_RESOURCE_GROUP_NAME
  AZURE_LOCATION
  AZURE_STORAGE_ACCOUNT_NAME
  AZURE_CONTAINER  (comma-separated list allowed)

Optional:
  PULUMI_AZ_CONTAINER
  FLOW_LOG_CONTAINER
  BACKUP_AZ_CONTAINER
  STORAGE_SKU (Standard_LRS | Standard_ZRS | Standard_GRS | ...)
  FORCE (1 to skip user prompts)

Usage:
  export AZURE_SUBSCRIPTION_ID=...
  export AZURE_RESOURCE_GROUP_NAME=rg-rag-prod
  python3 storage_account.py --create
  python3 storage_account.py --delete       # deletes containers by default; use AZURE_DELETE_ACCOUNT=1 to remove account
"""
from __future__ import annotations
import os
import sys
import json
import time
import subprocess
import argparse
from typing import Tuple, List

# ---------- small logger ----------
def now(): return time.strftime("%Y-%m-%dT%H:%M:%S%z")
def info(msg): print(f"{now()} INFO: {msg}")
def warn(msg): print(f"{now()} WARN: {msg}")
def die(msg, code=2):
    print(f"{now()} ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

# ---------- runner ----------
def run(cmd: List[str], check=True) -> Tuple[int, str, str]:
    info("RUN: " + " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if check and proc.returncode != 0:
        die(f"Command failed ({proc.returncode}): {' '.join(cmd)}\nstdout: {out}\nstderr: {err}")
    return proc.returncode, out, err

def az(*args):
    return run(["az", *args], check=True)

# ---------- config/env ----------
AZ_SUBS = os.getenv("AZURE_SUBSCRIPTION_ID") or os.getenv("AZ_SUBSCRIPTION_ID")
AZ_RG = os.getenv("AZURE_RESOURCE_GROUP_NAME")
AZ_LOC = os.getenv("AZURE_LOCATION", "eastus")
STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
CONTAINERS_CSV = os.getenv("AZURE_CONTAINER") or os.getenv("AZURE_CONTAINER_LIST") or ""
PULUMI_AZ_CONTAINER = os.getenv("PULUMI_AZ_CONTAINER")
FLOW_LOG_CONTAINER = os.getenv("FLOW_LOG_CONTAINER")
BACKUP_AZ_CONTAINER = os.getenv("BACKUP_AZ_CONTAINER")
FORCE = os.getenv("FORCE", "0") in ("1","true","True")
DELETE_ACCOUNT = os.getenv("AZURE_DELETE_ACCOUNT", "0") in ("1","true","True")
STORAGE_SKU = os.getenv("STORAGE_SKU", "Standard_LRS")

def validate_minimal_create():
    missing = []
    if not AZ_SUBS: missing.append("AZURE_SUBSCRIPTION_ID")
    if not AZ_RG: missing.append("AZURE_RESOURCE_GROUP_NAME")
    if not STORAGE_ACCOUNT: missing.append("AZURE_STORAGE_ACCOUNT_NAME")
    if not CONTAINERS_CSV: missing.append("AZURE_CONTAINER")
    if missing:
        die("Missing required envs: " + ", ".join(missing))

# ---------- helpers ----------
def ensure_subscription():
    az("account", "set", "--subscription", AZ_SUBS)
    info(f"Using subscription {AZ_SUBS}")

def rg_exists(name: str) -> bool:
    rc, out, _ = run(["az","group","exists","--name", name], check=False)
    return out.strip().lower() == "true"

def ensure_resource_group(name: str, location: str):
    if rg_exists(name):
        info(f"Resource group '{name}' exists.")
        return
    info(f"Creating resource group '{name}' in {location} ...")
    az("group","create","--name",name,"--location",location,"-o","json")
    info("Resource group created.")

def storage_account_exists(account: str, rg: str) -> bool:
    rc, out, _ = run(["az","storage","account","show","--name", account, "--resource-group", rg, "-o", "json"], check=False)
    return rc == 0

def create_storage_account(account: str, rg: str, location: str, sku: str):
    if storage_account_exists(account, rg):
        info(f"Storage account '{account}' already exists in RG '{rg}'.")
        return
    info(f"Creating storage account '{account}' in {rg}/{location} sku={sku} ...")
    az("storage","account","create",
       "--name", account,
       "--resource-group", rg,
       "--location", location,
       "--sku", sku,
       "--kind", "StorageV2",
       "--https-only", "true",
       "-o", "json")
    # wait until show reports present
    timeout = 300
    start = time.time()
    while time.time() - start < timeout:
        if storage_account_exists(account, rg):
            info("Storage account is now available.")
            return
        time.sleep(3)
    die("Storage account did not become available within timeout.")

def get_storage_account_key(account: str, rg: str) -> str:
    rc, out, _ = run(["az","storage","account","keys","list","--resource-group", rg, "--account-name", account, "--query", "[0].value", "-o", "tsv"])
    key = out.strip()
    if not key:
        die("Failed to obtain storage account key.")
    return key

def create_container(account: str, key: str, container: str):
    rc, out, err = run(["az","storage","container","create","--name", container, "--account-name", account, "--auth-mode", "key", "--account-key", key, "-o", "json"], check=False)
    if rc == 0:
        info(f"Container '{container}' ensured.")
        return
    lower = (err or out or "").lower()
    if "already exists" in lower or "exists" in lower:
        info(f"Container '{container}' already exists.")
        return
    die(f"Failed to create container '{container}': {err or out}")

def list_containers(account: str, key: str):
    rc, out, _ = run(["az","storage","container","list","--account-name", account, "--auth-mode", "key", "--account-key", key, "-o", "json"], check=False)
    if rc != 0:
        return []
    try:
        return json.loads(out)
    except Exception:
        return []

# ---------- high level ops ----------
def do_create():
    validate_minimal_create()
    ensure_subscription()
    ensure_resource_group(AZ_RG, AZ_LOC)
    create_storage_account(STORAGE_ACCOUNT, AZ_RG, AZ_LOC, STORAGE_SKU)

    key = get_storage_account_key(STORAGE_ACCOUNT, AZ_RG)
    containers = [c.strip() for c in CONTAINERS_CSV.split(",") if c.strip()]
    if PULUMI_AZ_CONTAINER: containers.append(PULUMI_AZ_CONTAINER)
    if FLOW_LOG_CONTAINER: containers.append(FLOW_LOG_CONTAINER)
    if BACKUP_AZ_CONTAINER: containers.append(BACKUP_AZ_CONTAINER)

    for c in sorted(set(containers)):
        create_container(STORAGE_ACCOUNT, key, c)

    # Print exports (do not actually set them — user copy/pastes)
    endpoint = f"https://{STORAGE_ACCOUNT}.blob.core.windows.net"
    info("CREATION COMPLETE. Add these exports to your shell (copy/paste):\n")
    print("### Copy-paste exports ###")
    print(f"export AZURE_STORAGE_ACCOUNT_NAME={STORAGE_ACCOUNT}")
    print(f"export AZURE_RESOURCE_GROUP_NAME={AZ_RG}")
    print(f"export AZURE_LOCATION={AZ_LOC}")
    print(f"export AZURE_STORAGE_ACCOUNT_KEY='{key}'")
    print(f"export AZURE_STORAGE_CONNECTION_STRING='DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={key};EndpointSuffix=core.windows.net'")
    print(f"export AZURE_BLOB_ENDPOINT={endpoint}")
    print("# Containers: " + ", ".join(sorted(set(containers))))
    print("### end ###")

def do_delete():
    ensure_subscription()
    if DELETE_ACCOUNT:
        if not FORCE:
            confirm = input("DELETE ENTIRE STORAGE ACCOUNT? Type 'yes' to proceed: ")
            if confirm.strip().lower() != "yes":
                info("Aborted.")
                return
        az("storage","account","delete","--name", STORAGE_ACCOUNT, "--resource-group", AZ_RG, "--yes")
        info("Storage account deletion requested.")
        return

    # delete containers only
    key = get_storage_account_key(STORAGE_ACCOUNT, AZ_RG)
    containers = [c.strip() for c in CONTAINERS_CSV.split(",") if c.strip()]
    if PULUMI_AZ_CONTAINER: containers.append(PULUMI_AZ_CONTAINER)
    if FLOW_LOG_CONTAINER: containers.append(FLOW_LOG_CONTAINER)
    if BACKUP_AZ_CONTAINER: containers.append(BACKUP_AZ_CONTAINER)
    if not FORCE:
        print("Containers to delete: ", containers)
        confirm = input("Type 'yes' to delete these containers: ")
        if confirm.strip().lower() != "yes":
            info("Aborted.")
            return
    for c in containers:
        run(["az","storage","container","delete","--account-name", STORAGE_ACCOUNT, "--name", c, "--auth-mode", "key", "--account-key", get_storage_account_key(STORAGE_ACCOUNT, AZ_RG)], check=False)
        info(f"Requested delete for container {c}")

# ---------- CLI ----------
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
        die("Aborted by user", 130)

if __name__ == "__main__":
    main()
