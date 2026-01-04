TS_MS=$(date +%s)000
kubectl -n monitoring port-forward svc/victoria-metrics 8428:8428 >/tmp/_vm_pf.log 2>&1 & PF_VM=$!
kubectl -n monitoring port-forward svc/vmagent 8429:8429 >/tmp/_va_pf.log 2>&1 & PF_VA=$!
sleep 1
PAYLOAD=$(
cat <<EOF
vm_promscrape_discovery_kubernetes_objects{role="pod"} 14 ${TS_MS}
vmagent_remotewrite_bytes_sent_total{url="1:secret-url"} 500000 ${TS_MS}
vmagent_remotewrite_rows_pushed_after_relabel_total{path="/vmagent-remotewrite-data"} 1200 ${TS_MS}
vmagent_remotewrite_samples_dropped_total 0 ${TS_MS}
vmagent_remotewrite_pending_data_bytes{path="/vmagent-remotewrite-data"} 0 ${TS_MS}
vm_promscrape_scrape_errors_total 0 ${TS_MS}
retrieval_requests_total{service="retriever"} 500 ${TS_MS}
retrieval_errors_total{service="retriever"} 5 ${TS_MS}
retrieval_request_duration_seconds_bucket{service="retriever",le="0.1"} 100 ${TS_MS}
retrieval_request_duration_seconds_bucket{service="retriever",le="0.5"} 200 ${TS_MS}
retrieval_request_duration_seconds_bucket{service="retriever",le="1"} 250 ${TS_MS}
retrieval_request_duration_seconds_bucket{service="retriever",le="+Inf"} 250 ${TS_MS}
retrieval_request_duration_seconds_sum{service="retriever"} 80 ${TS_MS}
retrieval_request_duration_seconds_count{service="retriever"} 250 ${TS_MS}
rest_responses_total{service="qdrant"} 300 ${TS_MS}
rest_responses_fail_total{service="qdrant"} 2 ${TS_MS}
rest_responses_duration_seconds_bucket{service="qdrant",le="0.1"} 50 ${TS_MS}
rest_responses_duration_seconds_bucket{service="qdrant",le="0.5"} 180 ${TS_MS}
rest_responses_duration_seconds_bucket{service="qdrant",le="1"} 250 ${TS_MS}
rest_responses_duration_seconds_bucket{service="qdrant",le="+Inf"} 300 ${TS_MS}
rest_responses_duration_seconds_sum{service="qdrant"} 40 ${TS_MS}
rest_responses_duration_seconds_count{service="qdrant"} 300 ${TS_MS}
collection_dead_replicas 0 ${TS_MS}
snapshot_creation_running 0 ${TS_MS}
EOF
)
curl -s -X POST --data-binary "$PAYLOAD" "http://127.0.0.1:8428/api/v1/import/prometheus" >/tmp/_vm_post.json 2>/dev/null || true
sleep 2
echo "=== COMPACT VERIFICATION ==="
curl -s --get --data-urlencode 'query=sum(up)/count(up)*100' 'http://127.0.0.1:8428/api/v1/query' | python3 -c "import sys,json; j=json.load(sys.stdin); r=j.get('data',{}).get('result',[]); print('scrape%=' + (r[0]['value'][1] if r else 'N'))"
curl -s --get --data-urlencode 'query=vm_promscrape_discovery_kubernetes_objects{role=\"pod\"}' 'http://127.0.0.1:8428/api/v1/query' | python3 -c "import sys,json; j=json.load(sys.stdin); r=j.get('data',{}).get('result',[]); print('discovered_pods=' + (r[0]['value'][1] if r else 'N'))"
curl -s --get --data-urlencode 'query=increase(vmagent_remotewrite_rows_pushed_after_relabel_total[5m])' 'http://127.0.0.1:8428/api/v1/query' | python3 -c "import sys,json; j=json.load(sys.stdin); r=j.get('data',{}).get('result',[]); print('rows_5m=' + (r[0]['value'][1] if r else 'N'))"
curl -s --get --data-urlencode 'query=increase(retrieval_requests_total[5m])' 'http://127.0.0.1:8428/api/v1/query' | python3 -c "import sys,json; j=json.load(sys.stdin); r=j.get('data',{}).get('result',[]); print('retrieval_req_5m=' + (r[0]['value'][1] if r else 'N'))"
curl -s --get --data-urlencode 'query=increase(retrieval_errors_total[5m])' 'http://127.0.0.1:8428/api/v1/query' | python3 -c "import sys,json; j=json.load(sys.stdin); r=j.get('data',{}).get('result',[]); print('retrieval_err_5m=' + (r[0]['value'][1] if r else 'N'))"
curl -s --get --data-urlencode 'query=histogram_quantile(0.95, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))' 'http://127.0.0.1:8428/api/v1/query' | python3 -c "import sys,json; j=json.load(sys.stdin); r=j.get('data',{}).get('result',[]); print('retriever_p95=' + (r[0]['value'][1] if r else 'N'))"
curl -s --get --data-urlencode 'query=increase(rest_responses_fail_total[5m])' 'http://127.0.0.1:8428/api/v1/query' | python3 -c "import sys,json; j=json.load(sys.stdin); r=j.get('data',{}).get('result',[]); print('qdrant_fail_5m=' + (r[0]['value'][1] if r else 'N'))"
echo; echo "=== vmagent /metrics sample (truncated) ==="
curl -sS http://127.0.0.1:8429/metrics | egrep -i 'vmagent_remotewrite|vm_promscrape_discovery|vm_promscrape_scrape_errors' | head -n 80
kill $PF_VM $PF_VA 2>/dev/null || true
