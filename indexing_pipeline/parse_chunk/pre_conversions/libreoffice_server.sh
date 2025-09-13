#!/usr/bin/env bash
set -euo pipefail

need_install=false

if ! command -v libreoffice >/dev/null 2>&1; then
  need_install=true
fi

if ! dpkg -s python3-uno >/dev/null 2>&1; then
  need_install=true
fi

if [ "$need_install" = true ]; then
  echo "[libreoffice-server] Installing LibreOffice (headless) + UNO bridge..."
  sudo add-apt-repository ppa:libreoffice/ppa -y || true
  sudo apt-get update -yq
  sudo apt-get install -y \
    libreoffice-script-provider-python \
    libreoffice-core \
    libreoffice-writer \
    libreoffice-calc \
    python3-uno \
    --no-install-recommends || true
fi

PORT=${1:-7003}
echo "[libreoffice-server] Starting LibreOffice UNO server on port ${PORT}..."
echo "[libreoffice-server] Connect with: uno:socket,host=127.0.0.1,port=${PORT};urp;StarOffice.ComponentContext"

exec soffice \
  --headless \
  --invisible \
  --nologo \
  --nofirststartwizard \
  --nodefault \
  --nocrashreport \
  --nolockcheck \
  --accept="socket,host=127.0.0.1,port=${PORT};urp;StarOffice.ServiceManager"
