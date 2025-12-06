from __future__ import annotations
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from jinja2 import Environment, BaseLoader

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_ROOT = ROOT / "infra" / "manifests"
OUT_DIR = MANIFESTS_ROOT / "argocd"
OUT_FILE = OUT_DIR / "applicationset.yaml"

TEMPLATE = """apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: rag-applications
  namespace: argocd
spec:
  generators:
    - list:
        elements:
{% for e in elements %}
          - name: {{ e.name }}
            path: {{ e.path | tojson }}
            destNamespace: {{ e.destNamespace }}
{% endfor %}
  template:
    metadata:
      name: '{{name}}'
    spec:
      project: default
      source:
        repoURL: "{{ repo_url }}"
        targetRevision: "{{ target_revision }}"
        path: '{{path}}'
      destination:
        server: https://kubernetes.default.svc
        namespace: '{{destNamespace}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
        syncOptions:
          - CreateNamespace=true
"""

# ---------- helpers ----------
def info(msg: str):
    print("[generator] " + msg, file=sys.stderr)

def env_list(name: str):
    v = os.environ.get(name, "").strip()
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]

def map_dest_namespace(name: str) -> str:
    m = {
        "k8s-core": "kube-system",
        "ingress": "ingress-nginx",
        "karpenter": "karpenter",
        "monitoring": "monitoring",
        "qdrant": "qdrant",
        "rag": "rag"
    }
    return m.get(name, name)

def enum_elements(manifests_root: Path):
    # Support GENERATOR_ALLOW and GENERATOR_SKIP
    allow = env_list("GENERATOR_ALLOW")
    skip = set(["argocd-root", "argocd", ".", ".."])
    skip |= set(env_list("GENERATOR_SKIP"))

    elements = []
    if not manifests_root.exists():
        return elements

    # iterate deterministic sorted directories
    for p in sorted([x for x in manifests_root.iterdir() if x.is_dir() and not x.name.startswith(".")]):
        name = p.name
        # allowlist overrides skiplist
        if allow:
            if name not in allow:
                continue
        else:
            if name in skip:
                continue
        dest = map_dest_namespace(name)
        elements.append({"name": name, "path": str((manifests_root / name).as_posix()), "destNamespace": dest})
    return elements

def render(ctx) -> str:
    env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = lambda v: json.dumps(v)
    return env.from_string(TEMPLATE).render(**ctx).strip() + "\n"

def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(path)

# ---------- main ----------
def main():
    p = argparse.ArgumentParser(description="Generate ApplicationSet (write only with --generate).")
    p.add_argument("--generate", action="store_true", help="Write infra/manifests/argocd/applicationset.yaml (idempotent).")
    args = p.parse_args()

    repo_url = os.environ.get("REPO_URL") or detect_repo_url()
    target_revision = os.environ.get("TARGET_REVISION") or detect_branch()

    elements = enum_elements(MANIFESTS_ROOT)
    if not elements:
        info("No folders found under infra/manifests; no ApplicationSet elements generated.")
        # print empty JSON for automation
        print(json.dumps([], indent=2))
        return

    ctx = {
        "elements": elements,
        "repo_url": repo_url or "https://example.com/replace-with-repo.git",
        "target_revision": target_revision or "main",
        "name": "{{name}}",
        "path": "{{path}}",
        "destNamespace": "{{destNamespace}}"
    }
    rendered = render(ctx)

    # Always emit machine JSON summary to stdout for automation
    summary = {
        "generated_count": len(elements),
        "elements": elements
    }
    print(json.dumps(summary, indent=2))

    if args.generate:
        prev_hash = None
        if OUT_FILE.exists():
            prev_hash = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()
        new_hash = hashlib.sha256(rendered.encode()).hexdigest()
        if prev_hash == new_hash:
            info(f"No change for {OUT_FILE}; file is up-to-date.")
            return
        atomic_write(OUT_FILE, rendered)
        info(f"Wrote {OUT_FILE} ({len(elements)} elements).")
    else:
        info("Dry-run (no file written). Use --generate to write the applicationset.yaml to the repo.")

# ---------- small git helpers (fallback detection) ----------
def run(cmd):
    import subprocess
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)

def detect_repo_url():
    rc, out, err = run(["git", "remote", "get-url", "origin"])
    if rc == 0 and out:
        u = out.strip()
        if u.startswith("git@"):
            try:
                host, path = u.split(":", 1)
                host = host.split("@", 1)[1]
                return f"https://{host}/{path}"
            except Exception:
                return u
        return u
    return None

def detect_branch():
    rc, out, err = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if rc == 0 and out:
        return out.strip()
    return "main"

if __name__ == "__main__":
    main()
