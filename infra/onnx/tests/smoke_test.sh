#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${BASE_URL:-http://localhost:8000}
echo "Checking health endpoint at ${BASE_URL}/healthz"
if curl --silent --fail "${BASE_URL}/healthz"; then
  echo "Health check passed"
else
  echo "Health check failed"
  exit 2
fi

# You may run a minimal gRPC test using grpcurl or a small python client.
echo "Smoke test completed."
