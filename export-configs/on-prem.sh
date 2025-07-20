

# FILL THESE CONFIGS, PASTE EVERYTHING IN THE TERMINAL DIRECTLY AND THEN EMPTY THIS FILE






export GITHUB_TOKEN="your_github_token"
export GITHUB_USER="your_github_username"
export REPO="RAG8s"
export FLUX_NAMESPACE="flux-system"
export FLUX_PATH="infra/fluxCD/on-premise"

flux bootstrap github \
  --owner="$GITHUB_USER" \
  --repository="$REPO" \
  --branch=main \
  --path="$FLUX_PATH" \
  --personal \
  --token-auth


  