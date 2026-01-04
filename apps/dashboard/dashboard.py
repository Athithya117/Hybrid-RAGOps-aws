#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import time
import json
import socket
import tempfile
import logging
import asyncio
import signal
import urllib.parse
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any, List
import httpx
from cachetools import TTLCache
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Template

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("dashboard_singlefile")

DATASOURCE_URL = os.getenv("DATASOURCE_URL", "http://victoria-metrics.monitoring.svc:8428").rstrip("/")
CLICKHOUSE_SERVICE = os.getenv("CLICKHOUSE_SERVICE_NAME", "clickhouse")
CLICKHOUSE_NAMESPACE = os.getenv("CLICKHOUSE_NAMESPACE", "clickhouse")
CLICKHOUSE_HTTP_PORT = int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123"))
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "") or ""
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "") or ""
SLO_SUCCESS_TARGET = os.getenv("SLO_SUCCESS_TARGET", "0.999")
SLO_LATENCY_QUANTILE = os.getenv("SLO_LATENCY_QUANTILE", "0.95")
APP_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
REQUEST_TIMEOUT = int(os.getenv("DASHBOARD_HTTP_TIMEOUT_SEC", "8"))
DEFAULT_RANGE_MINUTES = int(os.getenv("DASHBOARD_RANGE_MINUTES", "60"))
PORTFWD_READY_TIMEOUT = int(os.getenv("PORTFWD_READY_TIMEOUT", "20"))
CACHE_TTL = int(os.getenv("DASHBOARD_CACHE_TTL_SEC", "25"))
KUBECTL_BIN = os.getenv("KUBECTL_BIN", "kubectl")
VM_NS = os.getenv("VM_NAMESPACE", "monitoring")
VM_SVC = os.getenv("VM_SERVICE", "victoria-metrics")
VM_PORT = int(os.getenv("VICTORIA_PORT", "8428"))
RETRIEVER_NAMESPACE = os.getenv("RETRIEVER_NAMESPACE", "inference")
RETRIEVER_POD_LABEL = os.getenv("RETRIEVER_POD_LABEL", "app.kubernetes.io/name=retrieval")
ALLOWED_QUANTILES = ("0.95", "0.99")
RETRY_BACKOFFS = [1, 2, 4]

metrics_cache = TTLCache(maxsize=64, ttl=CACHE_TTL)
app = FastAPI()

HTML_TMPL = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Platform Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/htmx.org@1.12.0"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>body{font-family:Inter,ui-sans-serif,system-ui,Arial,Helvetica,sans-serif;margin:16px}</style>
</head>
<body class="bg-gray-50 text-gray-900">
<div class="max-w-5xl mx-auto">
  <header class="mb-4">
    <h1 class="text-2xl font-semibold">Platform — Ingestion & Service Overview</h1>
    <div class="text-sm text-gray-600 mt-1">SLO target: {{ slo_success_target }} · p95 quantile: {{ slo_latency_quantile }}</div>
  </header>

  <div id="alerts" hx-get="/api/health" hx-trigger="load" hx-swap="outerHTML"></div>

  <section class="bg-white rounded shadow p-4 mb-4">
    <h2 class="text-lg font-medium mb-2">Ingestion: vmagent discovery</h2>
    <div id="ingest-panel" hx-get="/api/metrics" hx-trigger="load, every 30s" hx-swap="outerHTML">
      <canvas id="c_ingest" height="120"></canvas>
    </div>
  </section>

  <section class="bg-white rounded shadow p-4 mb-4">
    <h2 class="text-lg font-medium mb-2">Retriever: error rate / p95 / rps</h2>
    <div id="retriever-panel" hx-get="/api/metrics" hx-trigger="load, every 30s" hx-swap="outerHTML">
      <canvas id="c_retriever" height="160"></canvas>
    </div>
  </section>

  <section class="bg-white rounded shadow p-4 mb-4">
    <h2 class="text-lg font-medium mb-2">Qdrant: qps / p95</h2>
    <div id="qdrant-panel" hx-get="/api/metrics" hx-trigger="load, every 30s" hx-swap="outerHTML">
      <canvas id="c_qdrant" height="140"></canvas>
    </div>
  </section>

  <section class="bg-white rounded shadow p-4 mb-4">
    <h2 class="text-lg font-medium mb-2">Logs (last 100)</h2>
    <form id="logsform" class="mb-2" onsubmit="return loadLogs()">
      Service: <input id="svc" name="service" value="retriever" class="border px-2 py-1 rounded">
      Namespace: <input id="ns" name="namespace" value="{{ retriever_ns }}" class="border px-2 py-1 rounded">
      <button class="ml-2 bg-blue-600 text-white px-3 py-1 rounded">Load</button>
    </form>
    <pre id="logs" class="bg-gray-900 text-green-200 p-3 rounded max-h-64 overflow-auto"></pre>
  </section>
</div>

<script>
async function fetchJSON(path){ const r=await fetch(path); return r.json(); }
function renderLine(id,labels,dsets){ const ctx=document.getElementById(id).getContext('2d'); if(window[id]) window[id].destroy(); window[id]=new Chart(ctx,{type:'line',data:{labels:labels,datasets:dsets}}); }
function toLabels(values){ return values.map(v=>new Date(v[0]*1000).toLocaleTimeString()); }
function toSeriesVals(values){ return values.map(v=>+v[1]); }

async function updateAll(){ const j=await fetchJSON('/api/metrics'); if(!j.ok){ document.getElementById('logs').textContent='metrics error: '+(j.error||'unknown'); return; } const s=j.payload.series;
  const ing=(s.discovery && s.discovery.data && s.discovery.data.result && s.discovery.data.result[0])||null;
  if(ing){ renderLine('c_ingest',toLabels(ing.values),[{label:'discovery',data:toSeriesVals(ing.values),borderColor:'green',fill:true}]);}
  const r_p95=(s.retriever_p95 && s.retriever_p95.data && s.retriever_p95.data.result && s.retriever_p95.data.result[0])||null;
  const r_err=(s.retriever_error_rate && s.retriever_error_rate.data && s.retriever_error_rate.data.result && s.retriever_error_rate.data.result[0])||null;
  const labels=(r_p95||r_err||{values:[]}).values?((r_p95||r_err).values.map(v=>new Date(v[0]*1000).toLocaleTimeString())):[];  
  const ds=[];
  if(r_p95) ds.push({label:'retriever_p95',data:toSeriesVals(r_p95.values),borderColor:'orange',yAxisID:'y1',fill:false});
  if(r_err) ds.push({label:'error_rate',data:toSeriesVals(r_err.values),borderColor:'red',fill:false});
  renderLine('c_retriever',labels,ds);
  const q_qps=(s.qdrant_qps && s.qdrant_qps.data && s.qdrant_qps.data.result && s.qdrant_qps.data.result[0])||null;
  const q_p95=(s.qdrant_p95 && s.qdrant_p95.data && s.qdrant_p95.data.result && s.qdrant_p95.data.result[0])||null;
  const qlabels=(q_qps||q_p95||{values:[]}).values?((q_qps||q_p95).values.map(v=>new Date(v[0]*1000).toLocaleTimeString())):[];
  const qds=[];
  if(q_qps) qds.push({label:'qdrant_qps',data:toSeriesVals(q_qps.values),borderColor:'purple',fill:false});
  if(q_p95) qds.push({label:'qdrant_p95',data:toSeriesVals(q_p95.values),borderColor:'teal',fill:false});
  renderLine('c_qdrant',qlabels,qds);
}
async function loadLogs(){ const svc=document.getElementById('svc').value; const ns=document.getElementById('ns').value; document.getElementById('logs').textContent='loading logs...'; const r=await fetch('/api/logs?service='+encodeURIComponent(svc)+'&namespace='+encodeURIComponent(ns)+'&limit=100'); const j=await r.json(); if(!j.ok){ document.getElementById('logs').textContent='error: '+(j.error||'unknown'); return false; } document.getElementById('logs').textContent=JSON.stringify(j.rows,null,2); return false; }
async function updateHealth(){ const r=await fetch('/api/health'); const j=await r.json(); const target=document.getElementById('alerts'); if(!j.ok){ target.innerHTML=`<div class="p-3 mb-4 rounded bg-red-50 border-red-200 text-red-800">Health degraded: ${j.victoria_err||''} ${j.clickhouse_err||''}</div>`; } else { target.innerHTML=`<div class="p-2 mb-4 rounded bg-green-50 border-green-200 text-green-800">All systems reachable</div>`; } }
document.addEventListener("DOMContentLoaded", async function(){ await updateAll(); await updateHealth(); setInterval(updateAll,30000); setInterval(updateHealth,30000); });
</script>
</body>
</html>
"""

def parse_csv_to_list(s: str) -> List[str]:
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    uniq: List[str] = []
    for p in parts:
        if p not in uniq:
            uniq.append(p)
    return uniq

def ensure_envs_valid():
    try:
        s = float(SLO_SUCCESS_TARGET)
        if not (0.0 < s < 1.0):
            LOG.error("invalid SLO_SUCCESS_TARGET %s", SLO_SUCCESS_TARGET)
            raise SystemExit(2)
    except Exception:
        LOG.error("invalid SLO_SUCCESS_TARGET %s", SLO_SUCCESS_TARGET)
        raise SystemExit(2)
    if SLO_LATENCY_QUANTILE not in ALLOWED_QUANTILES:
        LOG.warning("SLO_LATENCY_QUANTILE '%s' not allowed; falling back to '0.95'", SLO_LATENCY_QUANTILE)
    LOG.info("env validation passed")

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p

async def spawn_port_forward(ns: str, svc_or_pod: str, remote_port: int, local_port: int, logfile: str, is_pod: bool = False):
    cmd = [KUBECTL_BIN, "-n", ns, "port-forward", ("pod/" if is_pod else "svc/") + svc_or_pod, f"{local_port}:{remote_port}"]
    LOG.info("starting port-forward: %s -> local:%d log=%s", svc_or_pod, local_port, logfile)
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=open(logfile, "ab"), stderr=open(logfile, "ab"))
    return proc

async def wait_http_ok(base: str, path: str = "/api/v1/query", params: Optional[Dict[str, str]] = None, timeout: int = 2) -> bool:
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.get(base.rstrip("/") + path, params=params)
            if r.status_code == 200:
                return True
    except Exception as e:
        LOG.debug("wait_http_ok probe failed %s %s", base, e)
    return False

def shutil_which(name: str) -> bool:
    import shutil
    return shutil.which(name) is not None

async def ensure_service_url(ns: str, svc: str, remote_port: int, hint: Optional[str] = None) -> Tuple[str, Optional[asyncio.subprocess.Process], Optional[str]]:
    if hint:
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.get(hint.rstrip("/") + "/api/v1/query", params={"query": "up"})
                if r.status_code == 200 and r.json().get("status") == "success":
                    LOG.info("service reachable at hint %s", hint)
                    return hint, None, None
        except Exception as e:
            LOG.info("hint unreachable %s -> %s", hint, e)
    cluster_url = f"http://{svc}.{ns}.svc:{remote_port}"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.get(cluster_url.rstrip("/") + "/api/v1/query", params={"query": "up"})
            if r.status_code == 200 and r.json().get("status") == "success":
                LOG.info("service reachable at cluster DNS %s", cluster_url)
                return cluster_url, None, None
    except Exception as e:
        LOG.info("cluster DNS unreachable or refused for %s: %s", cluster_url, e)
    if shutil_which(KUBECTL_BIN):
        local = free_port()
        logfile = tempfile.NamedTemporaryFile(prefix="pf_", suffix=".log", delete=False).name
        proc = await spawn_port_forward(ns, svc, remote_port, local, logfile)
        try:
            for _ in range(PORTFWD_READY_TIMEOUT):
                if await wait_http_ok(f"http://127.0.0.1:{local}", "/api/v1/query", {"query": "up"}):
                    LOG.info("service reachable via port-forward http://127.0.0.1:%d", local)
                    return f"http://127.0.0.1:{local}", proc, logfile
                await asyncio.sleep(1)
            LOG.error("port-forward started but service did not respond in time; see logs %s", logfile)
            raise RuntimeError("port-forward not answering")
        except Exception as e:
            try:
                proc.terminate()
            except Exception:
                pass
            LOG.error("port-forward failed: %s", e)
            raise
    else:
        LOG.error("kubectl not found and cluster DNS failed; cannot access %s.%s", svc, ns)
        raise RuntimeError("no access to cluster service and kubectl not available")

def build_time_bounds(range_minutes: int = DEFAULT_RANGE_MINUTES) -> Tuple[int, int, int]:
    end = int(time.time())
    start = end - int(range_minutes * 60)
    step = max(15, int((end - start) / 60))
    return start, end, step

async def vm_query_instant(base: str, query: str) -> Dict[str, Any]:
    params = {"query": query}
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.get(base.rstrip("/") + "/api/v1/query", params=params)
        r.raise_for_status()
        return r.json()

async def vm_query_range(base: str, query: str, start: int, end: int, step: int, timeout: int = 20) -> Dict[str, Any]:
    params = {"query": query, "start": str(start), "end": str(end), "step": str(step)}
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(base.rstrip("/") + "/api/v1/query_range", params=params)
        r.raise_for_status()
        return r.json()

async def clickhouse_query_http(base: str, sql: str) -> Any:
    url = base.rstrip("/") + "/?query=" + urllib.parse.quote(sql, safe="")
    auth = None
    if CLICKHOUSE_USER:
        auth = (CLICKHOUSE_USER, CLICKHOUSE_PASSWORD)
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.get(url, auth=auth)
        r.raise_for_status()
        text = r.text
        try:
            return json.loads(text)
        except Exception:
            return text

def cache_get(key: str):
    try:
        return metrics_cache[key]
    except KeyError:
        return None

def cache_set(key: str, val: Any):
    try:
        metrics_cache[key] = val
    except Exception as e:
        LOG.warning("cache set failed: %s", e)

def safe_name(v: str) -> str:
    if not v:
        return ""
    if not all(c.isalnum() or c in "_-." for c in v) or len(v) > 64:
        raise HTTPException(status_code=400, detail="invalid name")
    return v

def safe_limit(v: int) -> int:
    try:
        n = int(v)
        if n <= 0:
            raise ValueError()
        return min(1000, n)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid limit")

@app.get("/", response_class=HTMLResponse)
async def ui_index(request: Request):
    ctx = {"slo_success_target": SLO_SUCCESS_TARGET, "slo_latency_quantile": SLO_LATENCY_QUANTILE, "retriever_ns": RETRIEVER_NAMESPACE}
    return HTMLResponse(Template(HTML_TMPL).render(**ctx))

@app.get("/api/health")
async def api_health():
    checks = {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}
    vm_ok = False
    ch_ok = False
    proc_vm = None
    logfile_vm = None
    try:
        base, proc_vm, logfile_vm = await ensure_service_url(VM_NS, VM_SVC, VM_PORT, DATASOURCE_URL)
        try:
            res = await vm_query_instant(base, 'up')
            if res.get("status") == "success":
                vm_ok = True
        except Exception as e:
            checks["victoria_err"] = str(e)
            checks["ok"] = False
    except Exception as e:
        checks["victoria_err"] = str(e)
        checks["ok"] = False
    try:
        click_base, proc_ch, logfile_ch = await ensure_service_url(CLICKHOUSE_NAMESPACE, CLICKHOUSE_SERVICE, CLICKHOUSE_HTTP_PORT, None)
        try:
            await clickhouse_query_http(click_base, "SELECT 1 FORMAT JSON")
            ch_ok = True
        except Exception as e:
            checks["clickhouse_err"] = str(e)
    except Exception as e:
        checks["clickhouse_err"] = str(e)
    finally:
        try:
            if proc_vm:
                proc_vm.terminate()
        except Exception:
            pass
    checks["victoria_ok"] = vm_ok
    checks["clickhouse_ok"] = ch_ok
    checks["ok"] = checks.get("victoria_ok", False)
    status = 200 if checks["ok"] else 503
    return JSONResponse(status_code=status, content=checks)

@app.get("/api/metrics")
async def api_metrics(range_minutes: int = Query(DEFAULT_RANGE_MINUTES, ge=5, le=1440)):
    key = f"metrics:{range_minutes}"
    cached = cache_get(key)
    if cached:
        cached["cache_age_seconds"] = int(time.time()) - cached.get("_cached_at", int(time.time()))
        payload = cached["payload"]
        return JSONResponse(content={"ok": True, "payload": payload, "cache_age_seconds": cached["cache_age_seconds"]})
    proc = None
    logfile = None
    try:
        base, proc, logfile = await ensure_service_url(VM_NS, VM_SVC, VM_PORT, DATASOURCE_URL)
    except Exception as e:
        LOG.error("cannot reach Victoria: %s", e)
        return JSONResponse(status_code=502, content={"ok": False, "error": str(e)})
    start, end, step = build_time_bounds(range_minutes)
    payload = {"start": start, "end": end, "step": step, "series": {}}
    queries = {
        "discovery": os.getenv("Q_DISCOVERY", 'vm_promscrape_discovery_kubernetes_objects{role="pod"}'),
        "remote_write": os.getenv("Q_REMOTE_WRITE", 'increase(vmagent_remotewrite_bytes_sent_total[5m])'),
        "retriever_error_rate": os.getenv("Q_RETRIEVER_ERR_RATE", 'sum(increase(retrieval_errors_total[5m])) / max(sum(increase(retrieval_requests_total[5m])), 1)'),
        "retriever_p95": os.getenv("Q_RETRIEVER_P95", f'histogram_quantile({SLO_LATENCY_QUANTILE}, sum(increase(retrieval_request_duration_seconds_bucket[5m])) by (le))'),
        "qdrant_qps": os.getenv("Q_QDRANT_QPS", 'sum(increase(qdrant_query_total[1m]))'),
        "qdrant_p95": os.getenv("Q_QDRANT_P95", 'histogram_quantile(0.95, sum(increase(qdrant_query_duration_seconds_bucket[5m])) by (le))')
    }

    async def run_query(name, q, s=start, e=end, st=step):
        last_exc = None
        for back in RETRY_BACKOFFS:
            try:
                if name in ("discovery", "remote_write"):
                    res = await vm_query_range(base, q, s, e, max(60, st))
                else:
                    res = await vm_query_range(base, q, s, e, st)
                if not res:
                    return {"status": "no_data", "query": q, "data": {"result": []}}
                return res
            except Exception as ex:
                last_exc = ex
                LOG.info("query %s attempt failed: %s; backoff=%s", name, ex, back)
                await asyncio.sleep(back)
        LOG.error("query %s all attempts failed: %s", name, last_exc)
        return {"status": "error", "query": q, "error": str(last_exc)}
    tasks = {n: asyncio.create_task(run_query(n, qq)) for n, qq in queries.items()}
    results = {}
    for n, t in tasks.items():
        try:
            results[n] = await t
        except Exception as e:
            LOG.error("unexpected error awaiting task %s: %s", n, e)
            results[n] = {"status": "error", "query": queries.get(n), "error": str(e)}
    wrapped = {}
    for k, v in results.items():
        if not v or (isinstance(v, dict) and v.get("status") in ("error", "no_data")):
            wrapped[k] = v
        else:
            wrapped[k] = v
    payload["series"] = wrapped
    cache_set(key, {"_cached_at": int(time.time()), "payload": payload})
    try:
        if proc:
            proc.terminate()
    except Exception:
        pass
    return JSONResponse(content={"ok": True, "payload": payload})

@app.get("/api/logs")
async def api_logs(service: str = Query("", max_length=64), namespace: str = Query("", max_length=64), limit: int = Query(100)):
    svc = safe_name(service) if service else ""
    ns = safe_name(namespace) if namespace else ""
    n = safe_limit(limit)
    proc = None
    logfile = None
    try:
        click_base, proc, logfile = await ensure_service_url(CLICKHOUSE_NAMESPACE, CLICKHOUSE_SERVICE, CLICKHOUSE_HTTP_PORT, None)
    except Exception as e:
        LOG.error("cannot reach ClickHouse: %s", e)
        return JSONResponse(status_code=502, content={"ok": False, "error": "clickhouse unreachable: " + str(e)})
    db = os.getenv("VECTOR_CLICKHOUSE_DATABASE", "logs")
    table = os.getenv("VECTOR_CLICKHOUSE_TABLE", "kube_logs")
    svc_clause = f"AND service = '{svc}'" if svc else ""
    ns_clause = f"AND namespace = '{ns}'" if ns else ""
    sql = f"SELECT ts, level, message, service, namespace, pod FROM {db}.{table} WHERE ts >= now() - INTERVAL 1 HOUR {svc_clause} {ns_clause} ORDER BY ts DESC LIMIT {n} FORMAT JSON"
    try:
        res = await clickhouse_query_http(click_base, sql)
        rows = res if isinstance(res, list) else res.get("data") if isinstance(res, dict) else res
        return JSONResponse(content={"ok": True, "rows": rows})
    except Exception as e:
        LOG.error("clickhouse logs query failed: %s", e)
        return JSONResponse(status_code=502, content={"ok": False, "error": str(e)})
    finally:
        try:
            if proc:
                proc.terminate()
        except Exception:
            pass

async def verify_mode() -> None:
    ensure_envs_valid()
    ok = True
    proc_vm = None
    logfile_vm = None
    try:
        base, proc_vm, logfile_vm = await ensure_service_url(VM_NS, VM_SVC, VM_PORT, DATASOURCE_URL)
        try:
            qres = await vm_query_instant(base, 'count({__name__=~"retrieval_requests_total|retrieval_errors_total|retrieval_request_duration_seconds"})')
            if qres.get("status") != "success" or not qres.get("data", {}).get("result"):
                LOG.error("victoria query empty: %s", qres)
                ok = False
            else:
                LOG.info("victoria query returned %d results", len(qres.get("data", {}).get("result", [])))
        except Exception as e:
            LOG.error("victoria instant query failed: %s", e)
            ok = False
    except Exception as e:
        LOG.error("victoria access failed: %s", e)
        ok = False
    try:
        click_base, proc_ch, logfile_ch = await ensure_service_url(CLICKHOUSE_NAMESPACE, CLICKHOUSE_SERVICE, CLICKHOUSE_HTTP_PORT, None)
        try:
            await clickhouse_query_http(click_base, "SELECT 1 FORMAT JSON")
            LOG.info("clickhouse query OK")
        except Exception as e:
            LOG.error("clickhouse query failed: %s", e)
            ok = False
    except Exception as e:
        LOG.info("clickhouse not reachable during verify (non-fatal): %s", e)
    if not ok:
        LOG.error("verification failed")
        sys.exit(2)
    LOG.info("verification succeeded")
    sys.exit(0)

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Single-file dashboard (FastAPI + HTMX + Tailwind)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--serve", action="store_true")
    return p.parse_args()

def run_uvicorn():
    import uvicorn
    LOG.info("starting dashboard service on %s:%d", APP_HOST, APP_PORT)
    uvicorn.run("dashboard:app", host=APP_HOST, port=APP_PORT, log_level="info", reload=False)

if __name__ == "__main__":
    args = parse_args()
    if args.verify:
        try:
            asyncio.run(verify_mode())
        except Exception as e:
            LOG.error("verify mode failed: %s", e)
            sys.exit(3)
    if args.serve:
        ensure_envs_valid()
        try:
            import pathlib
            this = pathlib.Path(__file__).resolve()
            if this.stem != "dashboard":
                import uvicorn
                LOG.info("starting dashboard service on %s:%d", APP_HOST, APP_PORT)
                uvicorn.run(app, host=APP_HOST, port=APP_PORT, log_level="info")
            else:
                run_uvicorn()
        except Exception as e:
            LOG.error("serve failed: %s", e)
            raise
