# apps/inference/frontend/frontend_ui.py
import os
import io
import json
import time
import logging
import asyncio
import cgi
from typing import Any, Optional
from urllib.parse import urljoin, parse_qs

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Template
import httpx
from contextlib import asynccontextmanager

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("frontend_ui")

QUERY_URL = os.getenv("QUERY_URL", "http://retrieval-svc.inference.svc.cluster.local:8001").rstrip("/")

if not QUERY_URL:
    raise RuntimeError("QUERY_URL environment variable must be set")

SERVICE_NAME = "frontend"
ENV = os.getenv("ENV", "dev")
SKIP_PRE_CHECKS = os.getenv("SKIP_PRE_CHECKS", "false").lower() in ("1", "true", "yes")

# templates
tailwind_cdn = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"
INDEX_TEMPLATE = Template(
    """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<link href='{{ tailwind }}' rel='stylesheet'></head><body class='bg-gray-50 min-h-screen p-6'>
<div class='max-w-3xl mx-auto'><h1 class='text-2xl font-semibold mb-4'>RAG Query UI</h1>
<form id='qry' hx-post='/run' hx-target='#result' hx-swap='innerHTML' class='space-y-4 bg-white p-4 rounded shadow'>
<div><label class='block text-sm font-medium text-gray-700'>Query</label>
<textarea name='query' required rows='3' class='mt-1 block w-full border rounded p-2'></textarea></div>
<div class='flex items-center space-x-4'>
<label class='flex items-center space-x-2'><input type='checkbox' id='tracing' name='enable_tracing' value='true' class='form-checkbox'/>
<span class='text-sm'>Enable tracing</span></label>
<label class='text-sm'>Top K <input type='number' name='top_k' value='5' min='1' max='50' class='ml-2 w-20 border rounded p-1 text-sm'/></label>
</div><div><button type='submit' class='bg-blue-600 text-white px-4 py-2 rounded'>Ask</button></div></form>
<div id='result' class='mt-6'></div></div><script src='https://unpkg.com/htmx.org@1.9.2'></script></body></html>"""
)

RESULT_TEMPLATE = Template(
    """
<div class='bg-white p-4 rounded shadow'><h2 class='font-medium mb-2'>Answer</h2>
<div class='prose'><pre class='whitespace-pre-wrap'>{{ answer }}</pre></div>
{% if chunks %}
  <h3 class='mt-4 font-medium'>Sources</h3>
  <ul class='space-y-2'>
  {% for c in chunks %}
    <li class='p-2 border rounded flex justify-between items-start'>
      <div class='text-sm w-full'>
        <div class='font-medium'>[{{ c.index }}]</div>
        <div class='mt-1 text-xs text-gray-700'>
          <ul class='list-none p-0 m-0'>
            {% for it in c.meta_items %}
              {% if it.k == 'content' %}
                <li><details><summary class='cursor-pointer text-blue-600'>Show content</summary>
                <div class='mt-2 text-xs text-gray-800 whitespace-pre-wrap'>{{ it.v | e }}</div></details></li>
              {% elif it.k == 'source_url' %}
                <li><strong>{{ it.k }}:</strong> <a href='#' class='source-link text-blue-600 underline' data-s3='{{ c.source_url }}'>open</a></li>
              {% else %}
                <li><strong>{{ it.k }}:</strong> {{ it.v }}</li>
              {% endif %}
            {% endfor %}
          </ul>
        </div>
        <div class='mt-2 text-xs text-gray-500 presign-result' id='presign-{{ loop.index0 }}'></div>
      </div>
    </li>
  {% endfor %}
  </ul>
{% endif %}
</div>
<script>
document.addEventListener('click',function(ev){
  var t=ev.target;
  if(t.classList.contains('source-link')){
    ev.preventDefault();
    var s3=t.getAttribute('data-s3');
    var links=Array.from(document.querySelectorAll('.source-link'));
    var idx=links.indexOf(t);
    var presignDiv=document.getElementById('presign-'+idx);
    presignDiv.textContent='Fetching presigned URL...';
    fetch('/presign',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({s3_path:s3,expires:3600,inline:true})})
      .then(r=>r.json()).then(j=>{
        if(j.url){
          presignDiv.innerHTML="<a href='"+j.url+"' target='_blank' class='text-green-600 underline'>Open presigned URL</a><div class='text-xs text-gray-600 break-words'>"+j.url+"</div>";
        } else {
          presignDiv.textContent='presign failed: '+(j.detail||j.error||JSON.stringify(j));
        }
      }).catch(e=>{presignDiv.textContent='presign error: '+e;});
  }
});
</script>
"""
)

# Prometheus (dedicated registry to avoid duplication)
_registry = CollectorRegistry()

def _c(name, doc, labelnames=None):
    return Counter(name, doc, labelnames or ["service", "env", "endpoint", "status_code"], registry=_registry)

def _h(name, doc, labelnames=None, buckets=None):
    return Histogram(name, doc, labelnames or ["service", "env", "endpoint", "status_code"], registry=_registry, buckets=buckets) if buckets else Histogram(name, doc, labelnames or ["service", "env", "endpoint", "status_code"], registry=_registry)

def _g(name, doc, labelnames=None):
    return Gauge(name, doc, labelnames or ["service", "env"], registry=_registry)

REQUEST_COUNT = _c("frontend_requests_total", "Total HTTP requests served by frontend")
REQUEST_LATENCY = _h(
    "frontend_request_duration_seconds",
    "Request latency (seconds) observed by frontend",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
BACKEND_LATENCY = _h(
    "frontend_backend_latency_seconds",
    "Time waiting for retrieval backend (seconds)",
    labelnames=["service", "env", "endpoint", "dependency"],
)
PRESIGN_COUNT = _c("frontend_presign_requests_total", "Presign requests proxied by frontend")
FRONTEND_ERRORS = _c("frontend_errors_total", "Frontend error counts", labelnames=["service", "env", "endpoint", "error_type"])
ERROR_RATE = _g("frontend_error_rate", "Error rate (errors/requests)", labelnames=["service", "env"])
SERVICE_READY = _g("service_ready", "Service readiness gauge", labelnames=["service", "env"])
SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(0)

# prober config
_prober_task = None
_stop_prober = False
PROBE_INTERVAL = int(os.getenv("FRONTEND_PROBE_INTERVAL", "10"))
PROBE_TIMEOUT = float(os.getenv("FRONTEND_PROBE_TIMEOUT", "3.0"))

async def probe_once() -> bool:
    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            r = await client.get(urljoin(QUERY_URL, "/healthz"))
            return r.status_code == 200
    except Exception:
        return False

async def prober_loop():
    global _stop_prober
    while not _stop_prober:
        try:
            ok = await probe_once()
            SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(1 if ok else 0)
        except Exception:
            SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(0)
        for _ in range(PROBE_INTERVAL):
            if _stop_prober:
                break
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _prober_task, _stop_prober
    _stop_prober = False
    if SKIP_PRE_CHECKS:
        SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(1)
        log.info("SKIP_PRE_CHECKS enabled: skipping initial probe and background prober (QUERY_URL=%s)", QUERY_URL)
    else:
        try:
            ok = await probe_once()
            SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(1 if ok else 0)
            log.info("lifespan initial probe ok=%s query_url=%s", ok, QUERY_URL)
        except Exception as e:
            SERVICE_READY.labels(service=SERVICE_NAME, env=ENV).set(0)
            log.warning("lifespan initial probe failed: %s", e)
        _prober_task = asyncio.create_task(prober_loop())
    try:
        yield
    finally:
        _stop_prober = True
        if _prober_task:
            _prober_task.cancel()
            try:
                await _prober_task
            except asyncio.CancelledError:
                pass
        log.info("lifespan prober stopped")

app = FastAPI(title="frontend-ui", lifespan=lifespan)

# robust form parsing (optional python-multipart; fallback to urlencoded/multipart)
class SimpleForm(dict):
    def get(self, k: str, default: Optional[Any] = None):
        v = super().get(k, default)
        if isinstance(v, (list, tuple)):
            return v[0] if v else default
        return v

async def parse_form(request: Request) -> SimpleForm:
    try:
        fr = await request.form()
        out = SimpleForm()
        if hasattr(fr, "multi_items"):
            for k, v in fr.multi_items():
                out.setdefault(k, []).append(v if not hasattr(v, "file") else v)
        else:
            for k, v in fr.items():
                out.setdefault(k, []).append(v)
        return out
    except Exception as e:
        log.debug("request.form() fallback: %s", e)

    ct = request.headers.get("content-type", "")
    raw = await request.body()
    if not raw:
        return SimpleForm()

    if "application/x-www-form-urlencoded" in ct:
        try:
            decoded = raw.decode("utf-8")
            qs = parse_qs(decoded, keep_blank_values=True)
            return SimpleForm({k: v for k, v in qs.items()})
        except Exception as ex:
            log.warning("urlencoded parse failed: %s", ex)
            return SimpleForm()

    if "multipart/form-data" in ct:
        try:
            fp = io.BytesIO(raw)
            environ = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": ct, "CONTENT_LENGTH": str(len(raw))}
            fs = cgi.FieldStorage(fp=fp, environ=environ, keep_blank_values=True)
            out = SimpleForm()
            if fs.list:
                for f in fs.list:
                    out.setdefault(f.name, []).append(f.value)
            return out
        except Exception as ex:
            log.warning("multipart parse failed: %s", ex)
            return SimpleForm()

    try:
        j = json.loads(raw.decode("utf-8"))
        if isinstance(j, dict):
            return SimpleForm(j)
    except Exception:
        pass

    return SimpleForm()

# helpers
def _update_error_rate():
    try:
        total_requests = 0.0
        total_errors = 0.0
        try:
            for m in REQUEST_COUNT.collect():
                for s in m.samples:
                    if s.name.startswith("frontend_requests_total"):
                        total_requests += float(s.value)
            for m in FRONTEND_ERRORS.collect():
                for s in m.samples:
                    if s.name.startswith("frontend_errors_total"):
                        total_errors += float(s.value)
        except Exception:
            try:
                r = REQUEST_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint="/", status_code="200")
                total_requests = float(r._value.get())
            except Exception:
                total_requests = 0.0
            try:
                e = FRONTEND_ERRORS.labels(service=SERVICE_NAME, env=ENV, endpoint="/", error_type="500")
                total_errors = float(e._value.get())
            except Exception:
                total_errors = 0.0

        val = (total_errors / total_requests) if total_requests > 0 else 0.0
        ERROR_RATE.labels(service=SERVICE_NAME, env=ENV).set(val)
    except Exception:
        log.debug("error rate compute failed", exc_info=True)

def metric_labels(endpoint: str, status_code: int):
    return {"service": SERVICE_NAME, "env": ENV, "endpoint": endpoint, "status_code": str(status_code)}

# routes
@app.get("/", response_class=HTMLResponse)
async def index():
    endpoint = "/"
    start = time.time()
    status_code = 200
    try:
        return HTMLResponse(INDEX_TEMPLATE.render(tailwind=tailwind_cdn))
    finally:
        elapsed = max(time.time() - start, 1e-6)
        try:
            REQUEST_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
            REQUEST_LATENCY.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).observe(elapsed)
            _update_error_rate()
        except Exception:
            log.debug("metric write failed in index", exc_info=True)

@app.post("/run")
async def run(request: Request):
    endpoint = "/run"
    start = time.time()
    status_code = 200
    try:
        form = await parse_form(request)
        query_text = (form.get("query") or "").strip()
        if not query_text:
            status_code = 400
            return HTMLResponse("<div class='bg-red-100 p-3 rounded'>Query required</div>", status_code=status_code)

        enable_tracing = True if str(form.get("enable_tracing") or "").lower() == "true" else False
        try:
            top_k = int(form.get("top_k") or 5)
        except Exception:
            top_k = 5

        payload = {"query": query_text, "enable_tracing": enable_tracing, "top_k": top_k, "return_chunks": enable_tracing}
        bk_start = time.time()
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(urljoin(QUERY_URL, "/generate"), json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                j = resp.json()
        except httpx.HTTPStatusError as e:
            status_code = 502
            try:
                FRONTEND_ERRORS.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, error_type=str(status_code)).inc()
                _update_error_rate()
            except Exception:
                pass
            return HTMLResponse(f"<div class='bg-red-100 p-3 rounded'>Backend error: {e.response.text}</div>", status_code=status_code)
        except Exception as e:
            status_code = 502
            try:
                FRONTEND_ERRORS.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, error_type="exception").inc()
                _update_error_rate()
            except Exception:
                pass
            log.exception("backend call failed")
            return HTMLResponse(f"<div class='bg-red-100 p-3 rounded'>Query failed: {e}</div>", status_code=status_code)
        finally:
            bk_elapsed = max(time.time() - bk_start, 1e-6)
            try:
                BACKEND_LATENCY.labels(service=SERVICE_NAME, env=ENV, endpoint="/generate", dependency="retrieval").observe(bk_elapsed)
            except Exception:
                log.debug("backend metric write failed", exc_info=True)

        answer = j.get("answer", "")
        chunks = j.get("chunks") or []
        for c in chunks:
            c_meta = c.get("meta_items") or []
            normalized = []
            for it in c_meta:
                if isinstance(it, dict) and "k" in it and "v" in it:
                    normalized.append(it)
                elif isinstance(it, (list, tuple)) and len(it) == 2:
                    normalized.append({"k": it[0], "v": it[1]})
                else:
                    normalized.append({"k": str(it[0]) if isinstance(it, (list, tuple)) else "meta", "v": str(it[1]) if isinstance(it, (list, tuple)) else str(it)})
            c["meta_items"] = normalized

        return HTMLResponse(RESULT_TEMPLATE.render(answer=answer, chunks=chunks))
    finally:
        elapsed = max(time.time() - start, 1e-6)
        try:
            REQUEST_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
            REQUEST_LATENCY.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).observe(elapsed)
            if status_code >= 400:
                FRONTEND_ERRORS.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, error_type=str(status_code)).inc()
            _update_error_rate()
        except Exception:
            log.debug("metric write failed in /run finally", exc_info=True)

@app.post("/presign")
async def presign(request: Request):
    endpoint = "/presign"
    start = time.time()
    status_code = 200
    try:
        j = await request.json()
        s3_path = j.get("s3_path")
        if not s3_path:
            status_code = 400
            return JSONResponse({"error": "s3_path required"}, status_code=status_code)
        inline = bool(j.get("inline", True))
        payload = {"s3_path": s3_path, "expires": int(j.get("expires", 3600)), "inline": inline}
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                resp = await client.post(urljoin(QUERY_URL, "/presign"), json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                return JSONResponse(resp.json())
            except httpx.HTTPStatusError as e:
                status_code = 502
                FRONTEND_ERRORS.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, error_type=str(status_code)).inc()
                _update_error_rate()
                return JSONResponse({"error": e.response.text}, status_code=status_code)
            except Exception as e:
                status_code = 500
                FRONTEND_ERRORS.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, error_type="exception").inc()
                _update_error_rate()
                return JSONResponse({"error": str(e)}, status_code=status_code)
    finally:
        elapsed = max(time.time() - start, 1e-6)
        try:
            PRESIGN_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
            REQUEST_COUNT.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).inc()
            REQUEST_LATENCY.labels(service=SERVICE_NAME, env=ENV, endpoint=endpoint, status_code=str(status_code)).observe(elapsed)
            _update_error_rate()
        except Exception:
            log.debug("metric write failed in /presign finally", exc_info=True)

@app.get("/metrics")
def metrics():
    try:
        return Response(generate_latest(_registry), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        log.exception("metrics generation failed")
        raise HTTPException(status_code=500, detail="metrics generation failed")

@app.get("/health")
def health():
    try:
        ready_val = bool(SERVICE_READY.labels(service=SERVICE_NAME, env=ENV)._value.get())
    except Exception:
        ready_val = False
    return {"status": "ok", "service": SERVICE_NAME, "env": ENV, "ready": ready_val}

@app.get("/readyz")
def readyz():
    try:
        ready_val = bool(SERVICE_READY.labels(service=SERVICE_NAME, env=ENV)._value.get())
    except Exception:
        ready_val = False
    return {
        "status": "ready" if ready_val else "not_ready",
        "service_ready": ready_val,
        "qdrant": None,
        "dense": None,
        "sparse": None,
        "reranker": None,
    }

if __name__ == "__main__":
    import uvicorn
    # directory self-contained: import string 'frontend_ui:app'
    uvicorn.run("frontend_ui:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level=os.getenv("LOG_LEVEL", "info"))
