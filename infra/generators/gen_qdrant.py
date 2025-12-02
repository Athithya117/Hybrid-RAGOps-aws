import os,sys,json,hashlib,shutil,argparse,re,subprocess
from pathlib import Path
from jinja2 import Environment, BaseLoader

def load_config():
    env=os.environ.get("ENV","STAGING").upper()
    cfg={}
    cfg["ENV"]=env
    cfg["MANIFESTS_DIR"]=Path(os.environ.get("MANIFESTS_DIR","infra/manifests/qdrant"))
    cfg["QDRANT_NAMESPACE"]=os.environ.get("QDRANT_NAMESPACE","qdrant")
    cfg["QDRANT_RELEASE"]=os.environ.get("QDRANT_RELEASE","qdrant")
    cfg["QDRANT_IMAGE_TAG"]=os.environ.get("QDRANT_IMAGE_TAG","v1.16.1")
    cfg["QDRANT_REPLICAS"]=int(os.environ.get("QDRANT_REPLICAS","1" if env=="STAGING" else "3"))
    cfg["QDRANT_CPU"]=os.environ.get("QDRANT_CPU","1" if env=="STAGING" else "4")
    cfg["QDRANT_MEMORY"]=os.environ.get("QDRANT_MEMORY","4Gi" if env=="STAGING" else "16Gi")
    cfg["QDRANT_STORAGE"]=os.environ.get("QDRANT_STORAGE","emptyDir")
    cfg["QDRANT_NODE_SELECTOR"]=os.environ.get("QDRANT_NODE_SELECTOR","")
    cfg["QDRANT_TAINT_KEY"]=os.environ.get("QDRANT_TAINT_KEY","qdrant-dedicated")
    cfg["QDRANT_TAINT_EFFECT"]=os.environ.get("QDRANT_TAINT_EFFECT","NoSchedule")
    cfg["QDRANT__STORAGE__SNAPSHOTS_PATH"]=os.environ.get("QDRANT__STORAGE__SNAPSHOTS_PATH","/qdrant/snapshots")
    cfg["BACKUP_S3_BUCKET"]=os.environ.get("BACKUP_S3_BUCKET","e2e-rag-system-42")
    cfg["BACKUP_S3_PREFIX"]=os.environ.get("BACKUP_S3_PREFIX","qdrant/backups")
    cfg["BACKUP_S3_REGION"]=os.environ.get("BACKUP_S3_REGION",os.environ.get("AWS_REGION","us-east-1"))
    cfg["BACKUP_S3_ENDPOINT"]=os.environ.get("BACKUP_S3_ENDPOINT","")
    cfg["BACKUP_COMPRESSION"]=os.environ.get("BACKUP_COMPRESSION","zstd")
    cfg["BACKUP_RETENTION"]=int(os.environ.get("BACKUP_RETENTION","5"))
    cfg["BACKUP_SCHEDULE"]=os.environ.get("BACKUP_SCHEDULE","0 */6 * * *") or "0 */6 * * *"
    cfg["BACKUP_IMAGE"]=os.environ.get("BACKUP_IMAGE","athithya5354/qdrant-backup:v2")
    cfg["IRSA_ROLE_ARN"]=os.environ.get("IRSA_ROLE_ARN","")
    cfg["INPUTS_HASH_PATH"]=cfg["MANIFESTS_DIR"]/".inputs_hash"
    cfg["SENSITIVE_KEYS"]=set(["AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_SESSION_TOKEN","QDRANT__SERVICE__API_KEY"])
    return cfg

def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k,v in obj.items()}
    return obj

def canonical_inputs_hash(cfg):
    keys=sorted(k for k in cfg.keys() if k not in ("INPUTS_HASH_PATH","SENSITIVE_KEYS"))
    payload={k:_json_safe(cfg[k]) for k in keys}
    clean_payload={k:v for k,v in payload.items()}
    return hashlib.sha256(json.dumps(clean_payload,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest()

VALUES_TPL = """
replicaCount: {{ qdrant_replicas }}
image:
  repository: "qdrant/qdrant"
  tag: "{{ qdrant_image_tag }}"
  pullPolicy: IfNotPresent
service:
  type: ClusterIP
p2p:
  port: 6335
cluster:
  enabled: true
persistence:
  enabled: {% if qdrant_storage == 'pvc' %}true{% else %}false{% endif %}

snapshots:
  enabled: true
  s3:
    bucket: "{{ backup_bucket }}"
    endpoint: "{{ backup_endpoint }}"
    region: "{{ backup_region }}"
    prefix: "{{ backup_prefix }}"

resources:
  requests:
    cpu: "{{ qdrant_cpu }}"
    memory: "{{ qdrant_memory }}"
  limits:
    cpu: "{{ qdrant_cpu }}"
    memory: "{{ qdrant_memory }}"

{% if env == 'PROD' and node_selector_key %}
nodeSelector:
  {{ node_selector_key }}: "{{ node_selector_value }}"
{% endif %}

tolerations:
  - key: "{{ taint_key }}"
    operator: "Exists"
    effect: "{{ taint_effect }}"
"""

SERVICEACCOUNT_TPL = """
apiVersion: v1
kind: ServiceAccount
metadata:
  name: qdrant-backup-sa
  namespace: {{ namespace }}
{% if irsa_role_arn %}
  annotations:
    eks.amazonaws.com/role-arn: "{{ irsa_role_arn }}"
{% endif %}
"""

CRONJOB_TPL = """
apiVersion: batch/v1
kind: CronJob
metadata:
  name: qdrant-backup
  namespace: {{ namespace }}
spec:
  schedule: "{{ schedule }}"
  concurrencyPolicy: Forbid
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: qdrant-backup-sa
          restartPolicy: OnFailure
          containers:
            - name: backup
              image: "{{ backup_image }}"
              imagePullPolicy: IfNotPresent
              env:
                - name: BACKUP_S3_BUCKET
                  value: "{{ backup_bucket }}"
                - name: BACKUP_S3_PREFIX
                  value: "{{ backup_prefix }}"
                - name: BACKUP_S3_REGION
                  value: "{{ backup_region }}"
                - name: BACKUP_S3_ENDPOINT
                  value: "{{ backup_endpoint }}"
                - name: SNAPSHOTS_PATH
                  value: "{{ snapshots_path }}"
                - name: BACKUP_COMPRESSION
                  value: "{{ backup_compression }}"
                - name: AWS_ACCESS_KEY_ID
                  valueFrom:
                    secretKeyRef:
                      name: qdrant-backup-aws
                      key: AWS_ACCESS_KEY_ID
                      optional: true
                - name: AWS_SECRET_ACCESS_KEY
                  valueFrom:
                    secretKeyRef:
                      name: qdrant-backup-aws
                      key: AWS_SECRET_ACCESS_KEY
                      optional: true
              command:
                - /bin/sh
                - -c
              args:
                - |
                  set -euo pipefail
                  TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
                  BACKUP_ID="${TIMESTAMP}-$(cat /proc/sys/kernel/random/uuid | cut -c1-8)"
                  PREFIX="{{ backup_prefix }}/${BACKUP_ID}"
                  TMPDIR="/tmp/qdrant-backup-${BACKUP_ID}"
                  mkdir -p "${TMPDIR}"
                  PODS=$(kubectl -n {{ namespace }} get pods -l app={{ release }} -o jsonpath='{.items[*].metadata.name}')
                  MANIFEST_TMP="${TMPDIR}/manifest.json"
                  echo '{"backup_id":"'"${BACKUP_ID}"'","created_at":"'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'","env":"{{ env }}","qdrant_image":"{{ qdrant_image_tag }}","nodes":[],"status":"IN_PROGRESS"}' > "${MANIFEST_TMP}"
                  aws s3 cp "${MANIFEST_TMP}" "s3://{{ backup_bucket }}/${PREFIX}/manifest.json" --region {{ backup_region }} || true
                  NODES_JSON="[]"
                  for POD in $PODS; do
                    PODFILE="${TMPDIR}/${POD}.tar.zst"
                    if kubectl -n {{ namespace }} exec "${POD}" -- sh -c "test -d {{ snapshots_path }} >/dev/null 2>&1"; then
                      kubectl -n {{ namespace }} exec "${POD}" -- sh -c "tar -C {{ snapshots_path }} -cf - . | zstd -19 --long=31 -o -" > "${PODFILE}" || { echo "snapshot failed for ${POD}"; continue; }
                    else
                      echo "no snapshots dir for ${POD}, skipping"
                      continue
                    fi
                    sha256sum "${PODFILE}" | awk '{print $1}' > "${PODFILE}.sha256"
                    aws s3 cp "${PODFILE}" "s3://{{ backup_bucket }}/${PREFIX}/${POD}.tar.zst" --region {{ backup_region }}
                    aws s3 cp "${PODFILE}.sha256" "s3://{{ backup_bucket }}/${PREFIX}/${POD}.tar.zst.sha256" --region {{ backup_region }}
                    SIZE=$(stat -c%s "${PODFILE}" 2>/dev/null || wc -c < "${PODFILE}")
                    CHECK=$(cat "${PODFILE}.sha256" || echo "")
                    NODE_OBJ=$(jq -n --arg pod "${POD}" --arg file "{{ backup_prefix }}/${BACKUP_ID}/${POD}.tar.zst" --arg size "${SIZE}" --arg sha "${CHECK}" '{"pod":$pod,"archive":$file,"size_bytes":($size|tonumber),"sha256":$sha,"taken_at":"'"$(date -u +"%Y-%m-%dT%H:%M:%SZ")"'"}')
                    NODES_JSON=$(echo "${NODES_JSON}" | jq ". + [${NODE_OBJ}]")
                    rm -f "${PODFILE}" "${PODFILE}.sha256"
                  done
                  FINAL_MANIFEST=$(jq -n --arg bid "${BACKUP_ID}" --arg ts "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" --arg env "{{ env }}" --arg qimg "{{ qdrant_image_tag }}" --argjson nodes "${NODES_JSON}" '{"backup_id":$bid,"created_at":$ts,"env":$env,"qdrant_image":$qimg,"nodes":$nodes,"status":"COMPLETE"}')
                  echo "${FINAL_MANIFEST}" > "${MANIFEST_TMP}"
                  aws s3 cp "${MANIFEST_TMP}" "s3://{{ backup_bucket }}/${PREFIX}/manifest.json" --region {{ backup_region }}
                  aws s3 cp "${MANIFEST_TMP}" "s3://{{ backup_bucket }}/{{ backup_prefix }}/latest.manifest.json" --region {{ backup_region }}
                  rm -rf "${TMPDIR}"
          terminationGracePeriodSeconds: 120
"""

SECRET_SAMPLE = """
apiVersion: v1
kind: Secret
metadata:
  name: qdrant-backup-aws
  namespace: qdrant
type: Opaque
stringData:
  AWS_ACCESS_KEY_ID: "AKIAxxxxxxxxxxxx"
  AWS_SECRET_ACCESS_KEY: "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
"""

def render(tpl, ctx):
    env=Environment(loader=BaseLoader(),trim_blocks=True,lstrip_blocks=True)
    return env.from_string(tpl).render(**ctx)

def ensure_dir(p:Path):
    p.mkdir(parents=True,exist_ok=True)

def atomic_write(path:Path, content:str):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(content)
    tmp.replace(path)

def valid_s3_bucket(name):
    if not name or len(name) < 3:
        return False
    if re.match(r'^[a-z0-9.\-]+$', name) is None:
        return False
    return True

def ensure_namespace(cfg, dry_run=False):
    try:
        if dry_run:
            print(f"DRY-RUN: ensure namespace {cfg['QDRANT_NAMESPACE']}")
            return True
        subprocess.run(["kubectl","create","namespace",cfg["QDRANT_NAMESPACE"],"--dry-run=client","-o","yaml"],check=True,stdout=subprocess.PIPE)
        subprocess.run(["kubectl","apply","-f","-"],input=f"apiVersion: v1\nkind: Namespace\nmetadata:\n  name: {cfg['QDRANT_NAMESPACE']}\n".encode(),check=True,stdout=subprocess.PIPE)
        return True
    except FileNotFoundError:
        print("kubectl not found; cannot ensure namespace. Skipping secret creation.")
        return False
    except subprocess.CalledProcessError:
        return True

def create_staging_secret(cfg, dry_run=False):
    if cfg["ENV"]!="STAGING":
        return
    aws_id=os.environ.get("AWS_ACCESS_KEY_ID","")
    aws_secret=os.environ.get("AWS_SECRET_ACCESS_KEY","")
    aws_token=os.environ.get("AWS_SESSION_TOKEN","")
    if not aws_id or not aws_secret:
        print("No AWS credentials in environment; skipping Secret creation in STAGING.")
        return
    if not ensure_namespace(cfg,dry_run=dry_run):
        return
    secret_yaml=f"apiVersion: v1\nkind: Secret\nmetadata:\n  name: qdrant-backup-aws\n  namespace: {cfg['QDRANT_NAMESPACE']}\ntype: Opaque\nstringData:\n  AWS_ACCESS_KEY_ID: \"{aws_id}\"\n  AWS_SECRET_ACCESS_KEY: \"{aws_secret}\"\n"
    if aws_token:
        secret_yaml += f"  AWS_SESSION_TOKEN: \"{aws_token}\"\n"
    if dry_run:
        print("=== secret to be applied to cluster (dry-run) ===")
        print(secret_yaml)
        return
    try:
        subprocess.run(["kubectl","apply","-f","-"],input=secret_yaml.encode(),check=True)
        print("Created/updated Kubernetes Secret qdrant-backup-aws in namespace",cfg["QDRANT_NAMESPACE"])
    except FileNotFoundError:
        print("kubectl not found; cannot create secret. Install kubectl or provide creds via ExternalSecret.")
    except subprocess.CalledProcessError as e:
        stderr=(e.stderr.decode() if e.stderr else "")
        print("kubectl apply failed:",stderr)

def delete_staging_secret(cfg):
    if cfg["ENV"]!="STAGING":
        return
    try:
        subprocess.run(["kubectl","delete","secret","qdrant-backup-aws","-n",cfg["QDRANT_NAMESPACE"],"--ignore-not-found"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        print("Deleted Kubernetes Secret qdrant-backup-aws (if it existed)")
    except FileNotFoundError:
        print("kubectl not found; cannot delete secret.")
    except subprocess.CalledProcessError:
        print("kubectl failed to delete secret; manual cleanup may be required.")

def generate(cfg, dry_run=False):
    ensure_dir(cfg["MANIFESTS_DIR"])
    if not valid_s3_bucket(cfg["BACKUP_S3_BUCKET"]):
        print("WARNING: BACKUP_S3_BUCKET looks invalid:",cfg["BACKUP_S3_BUCKET"])
    h=canonical_inputs_hash(cfg)
    existing=None
    if cfg["INPUTS_HASH_PATH"].exists():
        existing=cfg["INPUTS_HASH_PATH"].read_text().strip()
    if existing==h and not dry_run:
        print("No non-secret changes detected; skipping write.")
    else:
        node_key,node_val=("","")
        if cfg["QDRANT_NODE_SELECTOR"]:
            if "=" in cfg["QDRANT_NODE_SELECTOR"]:
                node_key,node_val=cfg["QDRANT_NODE_SELECTOR"].split("=",1)
            else:
                node_key="role"; node_val=cfg["QDRANT_NODE_SELECTOR"]
        ctx={
            "env":cfg["ENV"],
            "namespace":cfg["QDRANT_NAMESPACE"],
            "release":cfg["QDRANT_RELEASE"],
            "qdrant_image_tag":cfg["QDRANT_IMAGE_TAG"],
            "qdrant_replicas":cfg["QDRANT_REPLICAS"],
            "qdrant_cpu":cfg["QDRANT_CPU"],
            "qdrant_memory":cfg["QDRANT_MEMORY"],
            "qdrant_storage":cfg["QDRANT_STORAGE"],
            "backup_bucket":cfg["BACKUP_S3_BUCKET"],
            "backup_prefix":cfg["BACKUP_S3_PREFIX"],
            "backup_region":cfg["BACKUP_S3_REGION"],
            "backup_endpoint":cfg["BACKUP_S3_ENDPOINT"],
            "backup_compression":cfg["BACKUP_COMPRESSION"],
            "backup_schedule":cfg["BACKUP_SCHEDULE"],
            "backup_image":cfg["BACKUP_IMAGE"],
            "snapshots_path":cfg["QDRANT__STORAGE__SNAPSHOTS_PATH"],
            "node_selector_key":node_key,
            "node_selector_value":node_val,
            "taint_key":cfg["QDRANT_TAINT_KEY"],
            "taint_effect":cfg["QDRANT_TAINT_EFFECT"],
            "irsa_role_arn":cfg["IRSA_ROLE_ARN"]
        }
        values_yaml=render(VALUES_TPL,ctx)
        sa_yaml=render(SERVICEACCOUNT_TPL,ctx)
        cron_yaml=render(CRONJOB_TPL,ctx)
        secret_sample=SECRET_SAMPLE
        if dry_run:
            print("=== values.yaml ===\n")
            print(values_yaml)
            print("\n=== serviceaccount-backup.yaml ===\n")
            print(sa_yaml)
            print("\n=== backup-cronjob.yaml ===\n")
            print(cron_yaml)
            print("\n=== secret-sample.yaml ===\n")
            print(secret_sample)
        else:
            values_path=cfg["MANIFESTS_DIR"]/"values.yaml"
            sa_path=cfg["MANIFESTS_DIR"]/"serviceaccount-backup.yaml"
            cron_path=cfg["MANIFESTS_DIR"]/"backup-cronjob.yaml"
            secret_path=cfg["MANIFESTS_DIR"]/"secret-sample.yaml"
            atomic_write(values_path,values_yaml)
            atomic_write(sa_path,sa_yaml)
            atomic_write(cron_path,cron_yaml)
            atomic_write(secret_path,secret_sample)
            cfg["INPUTS_HASH_PATH"].write_text(h)
            print("Wrote manifests to",str(cfg["MANIFESTS_DIR"]))
    if cfg["ENV"]=="STAGING":
        create_staging_secret(cfg,dry_run=dry_run)
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            print("STAGING secret present in cluster (from env). Do NOT commit real credentials to git.")
    if cfg["ENV"]=="PROD" and not cfg["IRSA_ROLE_ARN"]:
        print("WARNING: ENV=PROD but IRSA_ROLE_ARN not set. Provide IRSA_ROLE_ARN to avoid using secrets in PROD.")

def delete(cfg):
    if cfg["MANIFESTS_DIR"].exists():
        for p in sorted(cfg["MANIFESTS_DIR"].glob("*")):
            try:
                p.unlink()
            except IsADirectoryError:
                shutil.rmtree(p)
        try:
            cfg["INPUTS_HASH_PATH"].unlink()
        except FileNotFoundError:
            pass
        print("Deleted manifests at",str(cfg["MANIFESTS_DIR"]))
    delete_staging_secret(cfg)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--delete",action="store_true")
    return p.parse_args()

def main():
    args=parse_args()
    cfg=load_config()
    if args.delete:
        delete(cfg)
        return
    generate(cfg,dry_run=args.dry_run)

if __name__=="__main__":
    main()
