#!/usr/bin/env python3
from __future__ import annotations
import os
import logging
import json
from typing import Optional
from urllib.parse import urljoin
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, BaseLoader, select_autoescape

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("frontend_noauth")

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() != "" else default

QUERY_URL = _env("QUERY_URL")
FRONTEND_URL = _env("FRONTEND_URL") or "http://localhost:8000"

missing = []
if not QUERY_URL:
    missing.append("QUERY_URL")
if missing:
    raise RuntimeError("Missing required env vars: " + ", ".join(missing))

def _ensure_url(u: str, name: str) -> None:
    if not u.startswith("http://") and not u.startswith("https://"):
        raise RuntimeError(f"{name} must be an http(s) URL: got {u}")

_ensure_url(QUERY_URL, "QUERY_URL")
_ensure_url(FRONTEND_URL, "FRONTEND_URL")

app = FastAPI(title="frontend-noauth", docs_url=None, redoc_url=None)
ENABLE_CORS = os.getenv("ENABLE_CORS", "false").lower() in ("1", "true", "yes")
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*")
if ENABLE_CORS:
    origins = ["*"] if CORS_ALLOWED_ORIGINS == "*" else [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    log.info("CORS enabled for origins: %s", origins)

# index template (same minimal UI)
INDEX_TEMPLATE = r"""
<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG UI (No Auth)</title>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-50 min-h-screen p-6">
<div class="max-w-4xl mx-auto">
  <div class="flex justify-between items-center mb-6">
    <h1 class="text-2xl font-semibold">RAG UI (No Auth)</h1>
  </div>

  <form id="qry" class="space-y-4 bg-white p-4 rounded shadow" onsubmit="return false;">
    <label class="block text-sm font-medium">Query</label>
    <textarea id="query" rows="3" class="mt-1 block w-full border rounded p-2" placeholder="Ask your question..."></textarea>
    <div class="flex items-center space-x-4">
      <label class="flex items-center space-x-2"><input id="enable_tracing" type="checkbox"/><span class="text-sm">Enable tracing</span></label>
      <label class="text-sm">Top K <input id="top_k" type="number" value="5" min="1" max="50" class="ml-2 w-20 border rounded p-1 text-sm"/></label>
    </div>
    <div><button id="ask" type="button" class="bg-blue-600 text-white px-4 py-2 rounded">Ask</button></div>
  </form>
  <div id="result" class="mt-6"></div>
  <div class="mt-6 text-xs text-gray-500">This frontend forwards requests to the backend configured by QUERY_URL on the server.</div>
</div>

<script>
function renderResult(json){
  const res = document.getElementById('result');
  if(!json || typeof json !== 'object'){
    res.innerHTML = '<pre class="whitespace-pre-wrap">'+(JSON.stringify(json,null,2)||'')+'</pre>';
    return;
  }
  // answer
  let out = '<div class="bg-white p-4 rounded shadow"><h2 class="font-medium mb-2">Answer</h2>';
  out += '<div class="prose"><pre class="whitespace-pre-wrap">'+(json.answer||'')+'</pre></div>';
  // chunks
  const chunks = json.chunks || [];
  if(chunks.length){
    out += '<h3 class="mt-4 font-medium">Sources</h3><ul class="space-y-2">';
    chunks.forEach((c, idx) => {
      out += '<li class="p-2 border rounded"><div class="text-sm w-full">';
      out += '<div class="font-medium">['+ (c.index || (idx+1)) +']</div>';
      out += '<div class="mt-1 text-xs text-gray-700"><ul class="list-none p-0 m-0">';
      const meta = c.meta_items || [];
      meta.forEach(it => {
        if(it.k === 'content'){
          out += '<li><details><summary class="cursor-pointer text-blue-600">Show content</summary><div class="mt-2 text-xs text-gray-800 whitespace-pre-wrap">'+escapeHtml(it.v)+'</div></details></li>';
        } else if(it.k === 'source_url'){
          // source link will call /presign to fetch a temporary URL
          out += '<li><strong>'+escapeHtml(it.k)+':</strong> <a href="#" class="source-link text-blue-600 underline" data-s3="'+escapeHtml(it.v)+'">open</a></li>';
        } else {
          out += '<li><strong>'+escapeHtml(it.k)+':</strong> '+escapeHtml(String(it.v))+'</li>';
        }
      });
      out += '</ul></div>';
      out += '<div class="mt-2 text-xs text-gray-500 presign-result" id="presign-'+idx+'"></div>';
      out += '</div></li>';
    });
    out += '</ul>';
  }
  out += '</div>';
  res.innerHTML = out;

  // attach click handlers for presign links
  document.querySelectorAll('.source-link').forEach((el, i) => {
    el.addEventListener('click', async function(ev){
      ev.preventDefault();
      const s3 = el.getAttribute('data-s3');
      const presignDiv = document.getElementById('presign-'+i);
      presignDiv.textContent = 'Fetching presigned URL...';
      try{
        const r = await fetch('/presign', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ s3_path: s3, expires:3600, inline:true })});
        const j = await r.json();
        if(r.ok && j.url){
          presignDiv.innerHTML = "<a href='"+escapeAttr(j.url)+"' target='_blank' class='text-green-600 underline'>Open presigned URL</a><div class='text-xs text-gray-600 break-words'>"+escapeHtml(j.url)+"</div>";
        } else {
          presignDiv.textContent = 'presign failed: ' + (j.detail || j.error || JSON.stringify(j));
        }
      }catch(e){
        presignDiv.textContent = 'presign error: ' + String(e);
      }
    });
  });
}

function escapeHtml(s){
  if(!s) return '';
  return s.replace(/[&<>"']/g, function(m){ return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]); });
}
function escapeAttr(s){ return escapeHtml(s).replace(/"/g,'&quot;'); }

async function submit(){
  const q=document.getElementById('query').value.trim();
  if(!q){ document.getElementById('result').innerHTML='<div class="bg-red-100 p-3 rounded">Query required</div>'; return; }
  const top_k=parseInt(document.getElementById('top_k').value||'5',10);
  const enable_tracing=document.getElementById('enable_tracing').checked===true;
  const payload={ query: q, top_k, enable_tracing, return_chunks: enable_tracing };
  document.getElementById('ask').disabled=true; document.getElementById('ask').innerText='Asking...';
  try{
    const resp = await fetch('/run', { method:'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify(payload) });
    const text = await resp.text();
    const ct = resp.headers.get('content-type') || '';
    if(resp.ok){
      try{
        if(ct.includes('application/json')){
          const j = JSON.parse(text);
          renderResult(j);
        } else {
          document.getElementById('result').innerHTML = '<pre class="whitespace-pre-wrap">'+escapeHtml(text)+'</pre>';
        }
      }catch(e){
        document.getElementById('result').innerHTML = '<pre class="whitespace-pre-wrap">'+escapeHtml(text)+'</pre>';
      }
    } else {
      document.getElementById('result').innerHTML = '<div class="bg-red-100 p-3 rounded">Error: '+resp.status+' — '+escapeHtml(text)+'</div>';
    }
  } catch(e){
    document.getElementById('result').innerHTML='<div class="bg-red-100 p-3 rounded">Request failed: '+String(e)+'</div>';
  } finally{
    document.getElementById('ask').disabled=false; document.getElementById('ask').innerText='Ask';
  }
}

document.addEventListener('DOMContentLoaded', function(){
  document.getElementById('ask').addEventListener('click', submit);
});
</script>
</body>
</html>
"""

env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
tmpl = env.from_string(INDEX_TEMPLATE)
INDEX_HTML = tmpl.render()

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

@app.get("/health")
async def health():
    return {"status": "ok", "query_url": QUERY_URL}

@app.post("/run")
async def run(request: Request):
    """
    Forwards the JSON body to backend /generate. If backend returns JSON, return it as JSON so the SPA can render it.
    If backend returns non-JSON, return the raw text.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict) or not body.get("query"):
        raise HTTPException(status_code=400, detail="Missing 'query'")

    target = urljoin(QUERY_URL.rstrip("/") + "/", "generate")
    headers = {"Content-Type": "application/json"}
    client_host = request.client.host if request.client else None
    if client_host:
        headers["X-Forwarded-For"] = client_host

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(target, json=body, headers=headers)
            content_type = resp.headers.get("content-type", "")
            if "application/json" in (content_type or ""):
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            else:
                return PlainTextResponse(content=resp.text, status_code=resp.status_code)
    except httpx.HTTPStatusError as e:
        log.error("Upstream returned non-200: %s", str(e))
        raise HTTPException(status_code=502, detail=f"Upstream error: {getattr(e.response, 'status_code', 'unknown')}")
    except Exception:
        log.exception("Upstream call failed")
        raise HTTPException(status_code=502, detail="Upstream call failed")

@app.post("/presign")
async def presign(request: Request):
    """
    Proxy to backend /presign. Accepts {"s3_path": "...", "expires": 3600, "inline": true}
    Returns backend JSON response (expected { "url": "..." }).
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    target = urljoin(QUERY_URL.rstrip("/") + "/", "presign")
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(target, json=body, headers={"Content-Type": "application/json"})
            try:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except Exception:
                return PlainTextResponse(content=resp.text, status_code=resp.status_code)
    except Exception:
        log.exception("Presign proxy failed")
        raise HTTPException(status_code=502, detail="Presign proxy failed")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("frontend:app", host=host, port=port, reload=True)
