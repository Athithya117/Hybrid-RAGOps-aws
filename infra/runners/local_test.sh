BASE_PORT=7000
NAMESPACE=qdrant

i=0
for pod in "${PODS[@]}"; do
    local_port=$((BASE_PORT + i))
    echo "Forwarding $pod → localhost:$local_port"

    kubectl port-forward "pod/$pod" "$local_port:6333" -n "$NAMESPACE" \
        > "pf-$pod.log" 2>&1 &

    i=$((i + 1))
done
