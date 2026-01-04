#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time
import json
import socket
import atexit
import signal
import logging
import argparse
import tempfile
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timedelta
try:
    from flask import Flask, jsonify, request, render_template_string
except Exception:
    sys.stderr.write("runtime error: Flask required (pip install flask)\n")
    raise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dashboards_final")

DATASOURCE_URL = os.getenv("DATASOURCE_URL", "http://victoria-metrics.monitoring.svc:8428").rstrip("/")
CLICKHOUSE_SERVICE = os.getenv("CLICKHOUSE_SERVICE_NAME", "clickhouse")
CLICKHOUSE_NAMESPACE = os.getenv("CLICKHOUSE_NAMESPACE", "clickhouse")
CLICKHOUSE_HTTP_PORT = int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
SLO_SUCCESS_TARGET = os.getenv("SLO_SUCCESS_TARGET", "0.999")
SLO_LATENCY_QUANTILE = os.getenv("SLO_LATENCY_QUANTILE", "0.95")
APP_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
REQUEST_TIMEOUT = int(os.getenv("DASHBOARD_HTTP_TIMEOUT_SEC", "8"))
DEFAULT_RANGE_MINUTES = int(os.getenv("DASHBOARD_RANGE_MINUTES", "60"))
KUBECTL_BIN = os.getenv("KUBECTL_BIN", "kubectl")

PF_PROCS = {}

def validate_envs():
    try:
        s = float(SLO_SUCCESS_TARGET)
        if not (0.0 < s < 1.0):
            raise ValueError()
    except Exception:
        LOG.error("invalid SLO_SUCCESS_TARGET %s", SLO_SUCCESS_TARGET)
        raise RuntimeError("SLO_SUCCESS_TARGET must be float between 0 and 1")
    if SLO_LATENCY_QUANTILE not in ("0.95", "0.99"):
        LOG.error("invalid SLO_LATENCY_QUANTILE %s", SLO_LATENCY_QUANTILE)
        raise RuntimeError("SLO_LATENCY_QUANTILE must be '0.95' or '0.99'")
    LOG.info("env validation passed")

def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p

def start_port_forward(ns: str, svc: str, remote_port: int) -> tuple[str, subprocess.Popen]:
    local_port = free_port()
    cmd = [KUBECTL_BIN, "-n", ns, "port-forward", f"svc/{svc}", f"{local_port}:{remote_port}"]
    logfile = tempfile.NamedTemporaryFile(prefix="pf_", suffix=".log", delete=False)
    LOG.info("starting port-forward: %s -> http://127.0.0.1:%d (log=%s)", f"{svc}.{ns}:{remote_port}", local_port, logfile.name)
    p = subprocess.Popen(cmd, stdout=logfile, stderr=logfile)
    PF_PROCS[(ns, svc, remote_port)] = (local_port, p, logfile.name)
    time.sleep(1.5)
    return f"http://127.0.0.1:{local_port}", p

def stop_all_port_forwards():
    for key, (port, proc, logfile) in list(PF_PROCS.items()):
        try:
            LOG.info("stopping port-forward for %s -> pid=%s logfile=%s", key, getattr(proc, "pid", None), logfile)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        except Exception as e:
            LOG.warning("error stopping port-forward %s: %s", key, e)
        try:
            os.unlink(logfile)
        except Exception:
            pass
    PF_PROCS.clear()

atexit.register(stop_all_port_forwards)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

def http_get_json(url: str, timeout: int = REQUEST_TIMEOUT):
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            if text.lstrip().startswith("{") or text.lstrip().startswith("["):
                return json.loads(text)
            return text
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace") if hasattr(e, "read") else ""
        LOG.error("HTTPError %s %s", e.code, body[:400])
        raise
    except urllib.error.URLError as e:
        LOG.debug("URLError %s", e)
        raise
    except Exception as e:
        LOG.debug("error %s", e)
        raise

def ensure_service_url(ns: str, svc: str, remote_port: int, base_url_hint: str | None = None) -> str:
    if base_url_hint:
        try:
            http_get_json(base_url_hint + "/api/v1/query?query=up")
            LOG.info("service reachable at hint %s", base_url_hint)
            return base_url_hint
        except Exception as e:
            LOG.info("hint unreachable %s -> %s", base_url_hint, e)
    cluster_url = f"http://{svc}.{ns}.svc:{remote_port}"
    try:
        http_get_json(cluster_url + "/api/v1/query?query=up")
        LOG.info("service reachable at cluster DNS %s", cluster_url)
        return cluster_url
    except Exception as e:
        LOG.info("cluster DNS unreachable or refused for %s: %s", cluster_url, e)
    if shutil_which(KUBECTL_BIN):
        try:
            local, proc = start_port_forward(ns, svc, remote_port)
            for i in range(10):
                try:
                    http_get_json(local + "/api/v1/query?query=up", timeout=2)
                    LOG.info("service reachable via port-forward %s", local)
                    return local
                except Exception:
                    time.sleep(1)
            LOG.error("port-forward started but service did not respond in time; see logs")
            raise RuntimeError("port-forward not answering")
        except Exception as e:
            LOG.error("port-forward failed: %s", e)
            raise
    else:
        LOG.error("kubectl not found and cluster DNS failed; cannot access service %s.%s", svc, ns)
        raise RuntimeError("no access to cluster service and kubectl not available")

def shutil_which(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None

def vm_query_instant(base_url: str, query: str):
    q = urllib.parse.urlencode({"query": query})
    url = base_url.rstrip("/") + "/api/v1/query?" + q
    return http_get_json(url)

def vm_query_range(base_url: str, query: str, start: int, end: int, step: int):
    params = urllib.parse.urlencode({"query": query, "start": str(start), "end": str(end), "step": str(step)})
    url = base_url.rstrip("/") + "/api/v1/query_range?" + params
    return http_get_json(url)

def clickhouse_query_url(base_url: str, sql: str):
    q = urllib.parse.quote(sql, safe="")
    url = base_url.rstrip("/") + "/?query=" + q
    headers = {}
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace")
            if text.lstrip().startswith("{") or text.lstrip().startswith("["):
                return json.loads(text)
            return text
    except Exception as e:
        LOG.error("clickhouse query error %s", e)
        raise

def build_time_bounds(range_minutes: int = DEFAULT_RANGE_MINUTES):
    end = int(time.time())
    start = end - int(range_minutes * 60)
    step = max(15, int((end - start) / 60))
    return start, end, step

def collect_payload(vm_base: str):
    start, end, step = build_time_bounds()
    out = {}
    try:
        out["discovery"] = vm_query_range(vm_base, 'vm_promscrape_discovery_kubernetes_objects{role="pod"}', start, end, step)
    except Exception as e:
        LOG.error("discovery query failed: %s", e)
        out["discovery"] = {}
    try:
        out["remote_write"] = vm_query_range(vm_base, 'increase(vm_persistentqueue_bytes_written_total[5m])', start, end, max(60, step))
    except Exception as e:
        LOG.error("remote-write query failed: %s", e)
        out["remote_write"] = {}
    try:
        out["retriever_error_rate"] = vm_query_range(vm_base, 'sum(rate(retrieval_errors_total[5m])) / sum(rate(retrieval_requests_total[5m]))', start, end, step)
    except Exception as e:
        LOG.error("retriever_error_rate failed: %s", e)
        out["retriever_error_rate"] = {}
    try:
        out["retriever_p95"] = vm_query_range(vm_base, f'histogram_quantile({SLO_LATENCY_QUANTILE}, sum(rate(retrieval_request_duration_seconds_bucket[5m])) by (le))', start, end, step)
    except Exception as e:
        LOG.error("retriever_p95 failed: %s", e)
        out["retriever_p95"] = {}
    try:
        out["qdrant_qps"] = vm_query_range(vm_base, 'sum(rate(qdrant_query_total[1m]))', start, end, step)
    except Exception as e:
        LOG.error("qdrant_qps failed: %s", e)
        out["qdrant_qps"] = {}
    try:
        out["qdrant_p95"] = vm_query_range(vm_base, 'histogram_quantile(0.95, sum(rate(qdrant_query_duration_seconds_bucket[5m])) by (le))', start, end, step)
    except Exception as e:
        LOG.error("qdrant_p95 failed: %s", e)
        out["qdrant_p95"] = {}
    return {"start": start, "end": end, "step": step, "series": out}

app = Flask("selfdashboard_final")

@app.route("/api/metrics")
def api_metrics():
    try:
        vm_base = ensure_service_url("monitoring", "victoria-metrics", 8428, DATASOURCE_URL)
    except Exception as e:
        LOG.error("cannot reach Victoria: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 502
    payload = collect_payload(vm_base)
    return jsonify({"ok": True, "payload": payload})

@app.route("/api/logs")
def api_logs():
    svc = request.args.get("service", "")
    ns = request.args.get("namespace", "")
    limit = int(request.args.get("limit", "100"))
    try:
        click_base = ensure_service_url(CLICKHOUSE_NAMESPACE, CLICKHOUSE_SERVICE, CLICKHOUSE_HTTP_PORT, None)
    except Exception as e:
        LOG.error("cannot reach ClickHouse: %s", e)
        return jsonify({"ok": False, "error": "clickhouse unreachable: " + str(e)}), 502
    svc_clause = f"AND service = '{svc}'" if svc else ""
    ns_clause = f"AND namespace = '{ns}'" if ns else ""
    sql = f"SELECT ts, level, message, service, namespace, pod FROM {os.getenv('VECTOR_CLICKHOUSE_DATABASE','logs')}.{os.getenv('VECTOR_CLICKHOUSE_TABLE','kube_logs')} WHERE ts >= now() - INTERVAL 1 HOUR {svc_clause} {ns_clause} ORDER BY ts DESC LIMIT {limit} FORMAT JSON"
    try:
        res = clickhouse_query_url(click_base, sql)
        return jsonify({"ok": True, "rows": res})
    except Exception as e:
        LOG.error("clickhouse logs query failed: %s", e)
        return jsonify({"ok": False, "error": str(e)}), 502

@app.route("/api/health")
def api_health():
    checks = {"ok": True}
    try:
        vm_base = ensure_service_url("monitoring", "victoria-metrics", 8428, DATASOURCE_URL)
        r = vm_query_instant = vm_query_instant if False else None
        q = vm_query_instant = (lambda base, q: http_get_lambda(base, q))(vm_base, 'count({__name__=~"retrieval_requests_total|retrieval_errors_total|retrieval_request_duration_seconds"})') if False else None
    except Exception as e:
        checks["victoria_ok"] = False
        checks["victoria_err"] = str(e)
        checks["ok"] = False
    try:
        click_base = ensure_service_url(CLICKHOUSE_NAMESPACE, CLICKHOUSE_SERVICE, CLICKHOUSE_HTTP_PORT, None)
        clickhouse_query_url(click_base, "SELECT 1 FORMAT JSON")
    except Exception as e:
        checks["clickhouse_ok"] = False
        checks["clickhouse_err"] = str(e)
        checks["ok"] = False
    return jsonify(checks)

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Platform Dashboard</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script><style>body{font-family:Arial,Helvetica,sans-serif;margin:16px} .panel{margin-bottom:24px;padding:12px;border:1px solid #ddd;border-radius:6px}</style></head><body><h2>Platform — Ingestion & Service Overview</h2><div class="panel"><h3>Ingestion: vmagent discovery</h3><canvas id="c_ingest" height="120"></canvas></div><div class="panel"><h3>Retriever: error rate / p95 / rps</h3><canvas id="c_retriever" height="160"></canvas></div><div class="panel"><h3>Qdrant: qps / p95</h3><canvas id="c_qdrant" height="140"></canvas></div><div class="panel"><h3>Logs (last 100)</h3><form id="logsform">Service: <input id="svc" name="service" value="retriever"> Namespace: <input id="ns" name="namespace" value="inference"> <button>Load</button></form><pre id="logs" style="max-height:240px;overflow:auto;background:#111;color:#efe;padding:8px"></pre></div><script>function seriesToPoints(s){ if(!s || !s.data || !s.data.result) return []; return s.data.result.map(r=>({name:r.metric.__name__ || JSON.stringify(r.metric), values:r.values || []})) }async function loadMetrics(){ const resp=await fetch('/api/metrics'); const j=await resp.json(); if(!j.ok){document.getElementById('logs').textContent='metrics error: '+(j.error||'unknown');return} const p=j.payload; const s=p.series; const ing=seriesToPoints(s.discovery); if(ing.length>0){ const vals=ing[0].values.map(v=>v[1]); new Chart(document.getElementById('c_ingest').getContext('2d'),{type:'line',data:{labels:ing[0].values.map(v=>new Date(v[0]*1000).toLocaleTimeString()),datasets:[{label:'discovery',data:vals,fill:true,borderColor:'green'}]},options:{scales:{y:{beginAtZero:true}}}}) } const ret_p95=seriesToPoints(s.retriever_p95); const ret_err=seriesToPoints(s.retriever_error_rate); const ret_rps=seriesToPoints(s.retriever_rps); const labels=(ret_p95[0]||ret_err[0]||ret_rps[0]||{values:[]}).values.map(v=>new Date(v[0]*1000).toLocaleTimeString()); const datasets=[]; if(ret_p95[0]) datasets.push({label:'retriever_p95',data:ret_p95[0].values.map(v=>v[1]),borderColor:'orange',yAxisID:'y1'}); if(ret_err[0]) datasets.push({label:'error_rate',data:ret_err[0].values.map(v=>v[1]),borderColor:'red'}); if(ret_rps[0]) datasets.push({label:'rps',data:ret_rps[0].values.map(v=>v[1]),borderColor:'blue'}); new Chart(document.getElementById('c_retriever').getContext('2d'),{type:'line',data:{labels:labels,datasets:datasets},options:{scales:{y:{beginAtZero:true},y1:{position:'right'}}}}); const q_qps=seriesToPoints(s.qdrant_qps); const q_p95=seriesToPoints(s.qdrant_p95); const qlabels=(q_qps[0]||q_p95[0]||{values:[]}).values.map(v=>new Date(v[0]*1000).toLocaleTimeString()); const qdatasets=[]; if(q_qps[0]) qdatasets.push({label:'qdrant_qps',data:q_qps[0].values.map(v=>v[1]),borderColor:'purple'}); if(q_p95[0]) qdatasets.push({label:'qdrant_p95',data:q_p95[0].values.map(v=>v[1]),borderColor:'teal'}); new Chart(document.getElementById('c_qdrant').getContext('2d'),{type:'line',data:{labels:qlabels,datasets:qdatasets},options:{scales:{y:{beginAtZero:true}}}});}document.getElementById('logsform').addEventListener('submit', async function(e){ e.preventDefault(); const svc=document.getElementById('svc').value; const ns=document.getElementById('ns').value; const r=await fetch('/api/logs?service='+encodeURIComponent(svc)+'&namespace='+encodeURIComponent(ns)+'&limit=100'); const j=await r.json(); if(!j.ok){ document.getElementById('logs').textContent='error: '+(j.error||'unknown'); return } document.getElementById('logs').textContent=JSON.stringify(j.rows, null, 2); }); loadMetrics(); setInterval(loadMetrics, 30000);</script></body></html>"""

@app.route("/")
def ui_index():
    return render_template_string(HTML)

def verify_mode():
    validate_envs()
    ok = True
    try:
        vm_base = ensure_service_url("monitoring", "victoria-metrics", 8428, DATASOURCE_URL)
        r = vm_query_instant = None
        try:
            qres = urllib.request.urlopen(vm_base + "/api/v1/query?query=" + urllib.parse.quote('count({__name__=~"retrieval_requests_total|retrieval_errors_total|retrieval_request_duration_seconds"})'), timeout=REQUEST_TIMEOUT)
            body = qres.read().decode()
            j = json.loads(body)
            if j.get("status") != "success" or not j.get("data", {}).get("result"):
                LOG.error("victoria query empty: %s", j)
                ok = False
            else:
                LOG.info("victoria query returned %d results", len(j.get("data", {}).get("result", [])))
        except Exception as e:
            LOG.error("victoria instant query failed: %s", e)
            ok = False
    except Exception as e:
        LOG.error("victoria access failed: %s", e)
        ok = False
    try:
        click_base = ensure_service_url(CLICKHOUSE_NAMESPACE, CLICKHOUSE_SERVICE, CLICKHOUSE_HTTP_PORT, None)
        try:
            res = clickhouse_query_url(click_base, "SELECT 1 FORMAT JSON")
            LOG.info("clickhouse query OK")
        except Exception as e:
            LOG.error("clickhouse query failed: %s", e)
            ok = False
    except Exception as e:
        LOG.error("clickhouse access failed: %s", e)
        ok = False
    if not ok:
        LOG.error("verification failed")
        sys.exit(2)
    LOG.info("verification succeeded")

def parse_args():
    p = argparse.ArgumentParser(description="Resilient single-dashboard service")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--serve", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    if args.verify:
        verify_mode()
    if args.serve:
        validate_envs()
        LOG.info("starting dashboard service on %s:%d", APP_HOST, APP_PORT)
        app.run(host=APP_HOST, port=APP_PORT)

if __name__ == "__main__":
    main()
