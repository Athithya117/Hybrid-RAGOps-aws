# ray-custom-ami.pkr.hcl (fixed: escape ${...} -> $${...} so HCL doesn't interpolate)
packer {
  required_plugins {
    amazon = {
      source  = "hashicorp/amazon"
      version = ">= 1.2.0"
    }
  }
}

variable "region" {
  type    = string
  default = "us-east-1"
}

variable "instance_type" {
  type    = string
  default = "t2.micro"
}

variable "ami_name" {
  type    = string
  default = "ray-custom-ami"
}

variable "source_ami_owner" {
  type    = string
  default = "099720109477"
}

variable "source_ami_filter" {
  type    = string
  default = "ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"
}

variable "app1_image" {
  type    = string
  default = ""
}

variable "app2_image" {
  type    = string
  default = ""
}

variable "hf_token" {
  type    = string
  default = ""
}

source "amazon-ebs" "ray-custom" {
  ami_name      = "${var.ami_name}-{{timestamp}}"
  instance_type = var.instance_type
  region        = var.region

  source_ami_filter {
    filters = {
      name                = var.source_ami_filter
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    owners      = [var.source_ami_owner]
    most_recent = true
  }

  ssh_username = "ubuntu"
}

build {
  sources = ["source.amazon-ebs.ray-custom"]

  provisioner "shell" {
    # Write full provisioning script to /tmp/provision-ami.sh and execute it.
    inline = [
      "cat > /tmp/provision-ami.sh <<'PROVISION_SCRIPT'",
      "#!/usr/bin/env bash",
      "set -euo pipefail",
      "trap 'echo \"ERROR: Script failed at line $LINENO\"; exit 1' ERR",
      "",
      "export DEBIAN_FRONTEND=noninteractive",
      "",
      "# Configurable variables (set by packer templating via -var or defaults)",
      "WORKSPACE_MODELS=$${WORKSPACE_MODELS:-/workspace/models}",
      "HF_TOKEN=\"{{user `hf_token`}}\"",
      "APP1_IMAGE=\"{{user `app1_image`}}\"",
      "APP2_IMAGE=\"{{user `app2_image`}}\"",
      "PYTHON_CMD=$${PYTHON_CMD:-python3}",
      "PIP_VERSION=$${PIP_VERSION:-25.2.0}",
      "",
      "log() { echo -e \"[provision-ami] $*\"; }",
      "",
      "log \"Starting AMI provisioning\"",
      "log \"WORKSPACE_MODELS=$${WORKSPACE_MODELS}\"",
      "log \"APP1_IMAGE=$${APP1_IMAGE}\"",
      "log \"APP2_IMAGE=$${APP2_IMAGE}\"",
      "if [ -n \"$${HF_TOKEN}\" ]; then log \"HF token provided (will be used by huggingface_hub)\"; fi",
      "",
      "log \"Updating apt and installing required packages...\"",
      "apt-get update -yq",
      "apt-get install -yq --no-install-recommends $${PYTHON_CMD} $${PYTHON_CMD}-venv $${PYTHON_CMD}-pip git curl ca-certificates gnupg lsb-release software-properties-common jq",
      "",
      "log \"Upgrading pip to $${PIP_VERSION}...\"",
      "$${PYTHON_CMD} -m pip install --no-warn-script-location --upgrade \"pip==$${PIP_VERSION}\"",
      "",
      "log \"Installing python libs required for model download\"",
      "$${PYTHON_CMD} -m pip install --no-cache-dir huggingface-hub==0.34.3 tqdm==4.67.1 requests==2.32.4 || true",
      "",
      "log \"Preparing model workspace: $${WORKSPACE_MODELS}\"",
      "mkdir -p \"$${WORKSPACE_MODELS}\"",
      "",
      "log \"Writing model downloader to /opt/download_models.py\"",
      "cat >/opt/download_models.py <<'PY'",
      "import os",
      "import sys",
      "import logging",
      "import shutil",
      "from pathlib import Path",
      "try:",
      "    from huggingface_hub import hf_hub_download",
      "except Exception as e:",
      "    raise ImportError(",
      "        \"huggingface_hub is required. Install with: pip install huggingface_hub\"",
      "    ) from e",
      "",
      "logging.basicConfig(level=os.getenv(\"LOG_LEVEL\", \"INFO\"))",
      "logger = logging.getLogger(\"download_hf\")",
      "",
      "WORKSPACE_MODELS = Path(os.getenv(\"WORKSPACE_MODELS\", \"/workspace/models\"))",
      "FORCE = os.getenv(\"FORCE_DOWNLOAD\", \"0\").lower() in (\"1\", \"true\", \"yes\")",
      "",
      "MODELS = [",
      "    {",
      "        \"repo_id\": \"RAG8s/gte-modernbert-base-onnx-int8\",",
      "        \"name\": \"gte-modernbert-base-onnx-int8\",",
      "        \"base\": \"onnx\",",
      "        \"items\": [",
      "            \"model.onnx\",",
      "            \"config.json\",",
      "            \"tokenizer.json\",",
      "            \"tokenizer_config.json\",",
      "            \"special_tokens_map.json\",",
      "        ],",
      "    }",
      "]",
      "",
      "def download_one(repo_id: str, remote: str, target: Path) -> bool:",
      "    if target.exists() and not FORCE:",
      "        logger.info(\"SKIP exists %s\", target)",
      "        return True",
      "",
      "    tmp_dir = Path(\"/tmp\") / \"hf_download\"",
      "    tmp_dir.mkdir(parents=True, exist_ok=True)",
      "",
      "    try:",
      "        got = hf_hub_download(",
      "            repo_id=repo_id,",
      "            filename=remote,",
      "            local_dir=str(tmp_dir),",
      "            local_dir_use_symlinks=False,",
      "            force_download=FORCE,",
      "        )",
      "        got_path = Path(got)",
      "        if got_path.exists():",
      "            target.parent.mkdir(parents=True, exist_ok=True)",
      "            if target.exists():",
      "                try:",
      "                    target.unlink()",
      "                except Exception:",
      "                    pass",
      "            shutil.move(str(got_path), str(target))",
      "            try:",
      "                os.chmod(str(target), 0o444)",
      "            except Exception:",
      "                pass",
      "            logger.info(\"Downloaded %s -> %s\", remote, target)",
      "            return True",
      "    except Exception as e:",
      "        logger.warning(\"Failed to download %s:%s (%s)\", repo_id, remote, e)",
      "    return False",
      "",
      "def ensure_model(model: dict) -> bool:",
      "    repo_id = model[\"repo_id\"]",
      "    name = model[\"name\"]",
      "    base = model.get(\"base\", \"llm\")",
      "",
      "    model_root = WORKSPACE_MODELS / base / name",
      "    ok = True",
      "    for item in model.get(\"items\", []):",
      "        remote_rel = str(item)",
      "        target = model_root / Path(remote_rel)",
      "        required = not remote_rel.endswith(\"special_tokens_map.json\")",
      "        success = download_one(repo_id, remote_rel, target)",
      "        if not success and required:",
      "            ok = False",
      "            logger.error(\"Missing required %s:%s\", name, remote_rel)",
      "    return ok",
      "",
      "def main() -> None:",
      "    all_ok = True",
      "    for m in MODELS:",
      "        if not ensure_model(m):",
      "            all_ok = False",
      "",
      "    if not all_ok:",
      "        logger.error(\"Some required files failed to download\")",
      "        sys.exit(2)",
      "",
      "    logger.info(\"All model artifacts are present under %s\", WORKSPACE_MODELS)",
      "",
      "if __name__ == \"__main__\":",
      "    main()",
      "PY",
      "",
      "chmod 755 /opt/download_models.py || true",
      "",
      "# Provide HF token to environment for huggingface_hub if available",
      "if [ -n \"$${HF_TOKEN}\" ]; then",
      "  log \"Exporting HF_TOKEN to environment\"",
      "  export HF_TOKEN=\"$${HF_TOKEN}\"",
      "else",
      "  log \"No HF_TOKEN provided; public models only or download may fail\"",
      "fi",
      "",
      "log \"Running model downloader (/opt/download_models.py)\"",
      "/usr/bin/env $${PYTHON_CMD} /opt/download_models.py || {",
      "  log \"Model download script failed; continuing (if models optional)\"",
      "}",
      "",
      "log \"Fixing permissions on $${WORKSPACE_MODELS}\"",
      "chown -R 1000:1000 \"$${WORKSPACE_MODELS}\" || true",
      "chmod -R 755 \"$${WORKSPACE_MODELS}\" || true",
      "chmod -R a+rX \"$${WORKSPACE_MODELS}\" || true",
      "",
      "log \"Installing Docker from official Docker repo\"",
      "apt-get remove -yq docker docker-engine docker.io containerd runc || true",
      "curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /tmp/docker_gpg.key",
      "mkdir -p /etc/apt/keyrings",
      "gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg /tmp/docker_gpg.key",
      "chmod a+r /etc/apt/keyrings/docker.gpg",
      "echo \"deb [arch=$$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $$(lsb_release -cs) stable\" >/etc/apt/sources.list.d/docker.list",
      "",
      "apt-get update -yq",
      "DOCKER_VERSION=$$(apt-cache madison docker-ce | awk '{print $3; exit}' || true)",
      "CLI_VERSION=$$(apt-cache madison docker-ce-cli | awk '{print $3; exit}' || echo \"$${DOCKER_VERSION}\")",
      "CONTAINERD_VERSION=$$(apt-cache madison containerd.io | awk '{print $3; exit}' || true)",
      "",
      "if [ -n \"$${DOCKER_VERSION}\" ] && [ -n \"$${CLI_VERSION}\" ] && [ -n \"$${CONTAINERD_VERSION}\" ]; then",
      "  log \"Installing specific versions: docker-ce=$${DOCKER_VERSION} docker-ce-cli=$${CLI_VERSION} containerd.io=$${CONTAINERD_VERSION}\"",
      "  apt-get install -yq --allow-downgrades docker-ce=\"$${DOCKER_VERSION}\" docker-ce-cli=\"$${CLI_VERSION}\" containerd.io=\"$${CONTAINERD_VERSION}\"",
      "else",
      "  log \"Installing latest docker packages\"",
      "  apt-get install -yq docker-ce docker-ce-cli containerd.io",
      "fi",
      "",
      "apt-mark hold docker-ce docker-ce-cli containerd.io || true",
      "systemctl enable --now docker || true",
      "log \"Adding 'ubuntu' (or build user) to docker group\"",
      "usermod -aG docker \"$${SUDO_USER:-ubuntu}\" || true",
      "",
      "log \"Cleaning apt cache and logs\"",
      "apt-get autoremove -yq",
      "apt-get clean -yq",
      "rm -rf /var/lib/apt/lists/*",
      "",
      "if command -v cloud-init >/dev/null 2>&1; then",
      "  log \"Cleaning cloud-init for AMI\"",
      "  cloud-init clean -s -l || true",
      "fi",
      "",
      "log \"Truncating logs\"",
      "truncate -s 0 /var/log/*log || true",
      "rm -rf /tmp/hf_download || true",
      "",
      "log \"Pre-pulling docker images onto AMI (if provided)\"",
      "if [ -n \"$${APP1_IMAGE}\" ]; then",
      "  log \"Pulling $${APP1_IMAGE}\"",
      "  docker pull \"$${APP1_IMAGE}\" || log \"Warning: failed to pull $${APP1_IMAGE}\"",
      "fi",
      "if [ -n \"$${APP2_IMAGE}\" ]; then",
      "  log \"Pulling $${APP2_IMAGE}\"",
      "  docker pull \"$${APP2_IMAGE}\" || log \"Warning: failed to pull $${APP2_IMAGE}\"",
      "fi",
      "",
      "# Tag the images to short names used in Ray config (optional)",
      "if docker image inspect \"$${APP1_IMAGE}\" >/dev/null 2>&1; then",
      "  docker tag \"$${APP1_IMAGE}\" app1 || true",
      "fi",
      "if docker image inspect \"$${APP2_IMAGE}\" >/dev/null 2>&1; then",
      "  docker tag \"$${APP2_IMAGE}\" app2 || true",
      "fi",
      "",
      "log \"Provisioning finished successfully\"",
      "exit 0",
      "PROVISION_SCRIPT",
      # make script executable and run it
      "chmod +x /tmp/provision-ami.sh",
      "/tmp/provision-ami.sh"
    ]

    environment_vars = [
      "APP1_IMAGE={{user `app1_image`}}",
      "APP2_IMAGE={{user `app2_image`}}",
      "HF_TOKEN={{user `hf_token`}}",
      "WORKSPACE_MODELS=/workspace/models"
    ]
  }
}
