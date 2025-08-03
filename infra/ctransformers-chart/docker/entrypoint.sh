#!/usr/bin/env bash
set -eo pipefail

# 1) Start Ray head in background
ray start --head --dashboard-host=0.0.0.0 --block &
RAY_PID=$!

# 2) Wait for GCS service to be ready
echo "Waiting for Ray GCS on port 6379..."
until nc -z localhost 6379; do sleep 1; done

# 3) Launch Ray Serve app
serve run rayserve_app:entrypoint &

# 4) Keep container alive under the Ray head process
wait $RAY_PID
