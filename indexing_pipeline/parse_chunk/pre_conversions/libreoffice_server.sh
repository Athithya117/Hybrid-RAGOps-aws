

#!/usr/bin/env bash
set -euo pipefail

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

