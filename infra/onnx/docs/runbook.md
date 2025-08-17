Runbook for rag8s-onnx RayService
================================

Key metrics:
- rag8s_inference_requests_total{model=...}
- rag8s_inference_latency_seconds_bucket{model=...}
- Pod restart count, OOMKilled
- CPU throttling and node CPU utilization

Common issues & recovery:
1. Model file not found:
   - Check pod logs for FileNotFoundError with model path.
   - Verify HF token secret & local model presence if baked in image.
2. OOM / CPU exhaustion:
   - Inspect resource usage.
   - Tune EMBEDDER_OMP_NUM_THREADS / RERANKER_OMP_NUM_THREADS to match pod CPU.
   - Increase pod resources and adjust autoscaling.
3. Failed Deploy:
   - Check RayService status and operator logs.
   - Re-apply YAML with "kubectl apply -f onnx/templates/rayservice.yaml"

Rollback:
- Use `helm rollback <release> <revision>` or re-deploy previous image tag with `helm upgrade --set image.tag=<old-tag>`.

Contact:
- Maintain contact & escalation in Chart.yaml/README.
