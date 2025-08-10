#!/usr/bin/env bash
set -euo pipefail

CHART_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VALUES_DIR="$CHART_DIR/values"

echo "=== [helm_charts.sh] Updating dependencies for rag8s-aws chart ==="
echo "Chart directory: $CHART_DIR"

echo "[1/4] Cleaning existing chart dependencies..."
rm -rf "$CHART_DIR/charts"/*
mkdir -p "$CHART_DIR/charts"

echo "[2/4] Processing dependencies from Chart.yaml..."

DEPS=$(yq e '.dependencies[] | {name: .name, repo: .repository, ver: .version, cond: .condition}' -o=json "$CHART_DIR/Chart.yaml")

for row in $(echo "$DEPS" | jq -r '@base64'); do
    _jq() {
        echo "$row" | base64 --decode | jq -r "$1"
    }

    name=$(_jq '.name')
    repo=$(_jq '.repo')
    version=$(_jq '.ver')
    condition=$(_jq '.cond')

    # Determine which values file to check
    values_file="$VALUES_DIR/$(echo "$condition" | cut -d'.' -f1).yaml"
    enabled_flag=false

    # 1️⃣ Check per-dependency values file if exists
    if [[ -f "$values_file" ]]; then
        enabled_flag=$(yq e ".$condition // false" "$values_file" || echo "false")
    fi

    # 2️⃣ Fallback: check base.yaml if not already enabled
    if [[ "$enabled_flag" != "true" && -f "$VALUES_DIR/base.yaml" ]]; then
        enabled_flag=$(yq e ".$condition // false" "$VALUES_DIR/base.yaml" || echo "false")
    fi

    if [[ "$enabled_flag" != "true" ]]; then
        echo "Skipping $name (condition '$condition' is false or not set in $values_file or base.yaml)"
        continue
    fi

    echo "Updating $name → $version from $repo"
    helm repo add "$name" "$repo" 2>/dev/null || true

    if [[ "$repo" == oci://* ]]; then
        echo "Pulling OCI dependency $name..."
        helm pull "$repo/$name" --version "$version" --untar --untardir "$CHART_DIR/charts"
    else
        helm pull "$name" --repo "$repo" --version "$version" --untar --untardir "$CHART_DIR/charts"
    fi
done

echo "[3/4] Pulling OCI-based dependencies (if any) not already handled..."

oci_repos=$(yq e '.dependencies[].repository' "$CHART_DIR/Chart.yaml" | grep '^oci://' || true)

if [[ -n "$oci_repos" ]]; then
    for oci_repo in $oci_repos; do
        echo "Handling OCI repo: $oci_repo"
    done
fi

echo "[4/4] Finalizing Helm dependency update..."
helm dependency update "$CHART_DIR"
helm dependency list "$CHART_DIR"

echo "Dependencies synced successfully."
