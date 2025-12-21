# apps/inference/frontend/frontend_ui.py
import logging, os, sys
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from jinja2 import Environment, BaseLoader, select_autoescape

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import QUERY_URL, EXTERNAL_BASE, DISPLAY_SOURCES_IN_UI, DISPLAY_TOPK_IN_UI, REQUIRE_AUTH

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger("frontend_noauth")

def _ensure_url(u: str, name: str) -> str:
    if not isinstance(u, str) or (not u.startswith("http://") and not u.startswith("https://")):
        log.warning("%s must be http(s). Falling back to default for %s", name, u)
        return "http://retrieval-svc.inference.svc.cluster.local:8001" if name == "QUERY_URL" else "http://localhost:8000"
    return u

QUERY_URL = _ensure_url(QUERY_URL, "QUERY_URL")
FRONTEND_URL = _ensure_url(EXTERNAL_BASE, "FRONTEND_URL")

app = FastAPI(title="frontend-noauth", docs_url=None, redoc_url=None)
ENABLE_CORS = os.getenv("ENABLE_CORS", "false").lower() in ("1", "true", "yes")
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*")
if ENABLE_CORS:
    origins = ["*"] if CORS_ALLOWED_ORIGINS == "*" else [o.strip() for o in CORS_ALLOWED_ORIGINS.split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])
    log.info("CORS enabled for origins: %s", origins)

DISPLAY_SOURCES = bool(DISPLAY_SOURCES_IN_UI)
DISPLAY_TOPK = bool(DISPLAY_TOPK_IN_UI)
REQUIRE_AUTH_UI = bool(REQUIRE_AUTH)

INDEX_TEMPLATE = r"""<!doctype html>
<html>
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG UI</title>
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-50 min-h-screen p-6">
<div class="max-w-4xl mx-auto">
  <div class="flex justify-between items-center mb-6">
    <h1 class="text-2xl font-semibold">RAG UI</h1>
    <div id="auth-controls" class="text-sm"></div>
  </div>
  <form id="qry" class="space-y-4 bg-white p-4 rounded shadow" onsubmit="return false;">
    <label class="block text-sm font-medium">Query</label>
    <textarea id="query" rows="3" class="mt-1 block w-full border rounded p-2" placeholder="Ask your question..."></textarea>
    <div class="flex items-center space-x-4">
      {% if display_sources %}
      <label class="flex items-center space-x-2"><input id="enable_tracing" type="checkbox"/><span class="text-sm">Enable tracing</span></label>
      {% endif %}
      {% if display_topk %}
      <label class="text-sm">Top K <input id="top_k" type="number" value="5" min="1" max="50" class="ml-2 w-20 border rounded p-1 text-sm"/></label>
      {% endif %}
    </div>
    <div><button id="ask" type="button" class="bg-blue-600 text-white px-4 py-2 rounded" {% if require_auth %}disabled{% endif %}>{% if require_auth %}Login required{% else %}Ask{% endif %}</button></div>
  </form>
  <div id="result" class="mt-6"></div>
</div>
<script>
const DISPLAY_SOURCES = {{ 'true' if display_sources else 'false' }};
const DISPLAY_TOPK = {{ 'true' if display_topk else 'false' }};
const REQUIRE_AUTH = {{ 'true' if require_auth else 'false' }};
async function checkAuth(){
  const ctrl = document.getElementById('auth-controls');
  ctrl.innerHTML = '<span class="text-gray-500">Checking auth…</span>';
  const tok = localStorage.getItem('app_jwt');
  if(!tok){
    ctrl.innerHTML = '<a href="/auth/login" class="text-sm text-blue-600 underline">Login</a>';
    if(REQUIRE_AUTH){ disableQuery(); } else { enableQuery(); }
    return;
  }
  try{
    const resp = await fetch('/auth/me',{headers:{'Authorization':'Bearer '+tok}});
    if(!resp.ok){
      localStorage.removeItem('app_jwt');
      ctrl.innerHTML = '<a href="/auth/login" class="text-sm text-blue-600 underline">Login</a>';
      if(REQUIRE_AUTH){ disableQuery(); } else { enableQuery(); }
      return;
    }
    const j = await resp.json();
    const name = j.user && (j.user.name || j.user.email || j.user.sub) || 'user';
    ctrl.innerHTML = '<span class="mr-4 text-sm text-gray-700">Signed in as '+escapeHtml(name)+'</span><button id="logout-btn" class="text-sm text-red-600 underline">Logout</button>';
    document.getElementById('logout-btn').addEventListener('click', async function(){ try{ await fetch('/auth/logout'); }catch(e){} localStorage.removeItem('app_jwt'); window.location.reload(); });
    enableQuery();
  }catch(e){
    localStorage.removeItem('app_jwt');
    ctrl.innerHTML = '<a href="/auth/login" class="text-sm text-blue-600 underline">Login</a>';
    if(REQUIRE_AUTH){ disableQuery(); } else { enableQuery(); }
  }
}
function disableQuery(){
  const q = document.getElementById('query');
  const btn = document.getElementById('ask');
  if(q) q.disabled = true;
  if(btn){ btn.disabled = true; btn.innerText = 'Login required'; }
}
function enableQuery(){
  const q = document.getElementById('query');
  const btn = document.getElementById('ask');
  if(q) q.disabled = false;
  if(btn){ btn.disabled = false; btn.innerText = 'Ask'; }
}
function renderResult(json){
  const res = document.getElementById('result');
  if(!json || typeof json !== 'object'){
    res.innerHTML = '<pre class="whitespace-pre-wrap">'+(JSON.stringify(json,null,2)||'')+'</pre>';
    return;
  }
  let out = '<div class="bg-white p-4 rounded shadow"><h2 class="font-medium mb-2">Answer</h2>';
  out += '<div class="prose"><pre class="whitespace-pre-wrap">'+(json.answer||'')+'</pre></div>';
  const chunks = json.chunks || [];
  if(DISPLAY_SOURCES && chunks.length){
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
  document.querySelectorAll('.source-link').forEach((el, i) => {
    el.addEventListener('click', async function(ev){
      ev.preventDefault();
      const s3 = el.getAttribute('data-s3');
      const presignDiv = document.getElementById('presign-'+i);
      presignDiv.textContent = 'Fetching presigned URL...';
      try{
        const tok = localStorage.getItem('app_jwt');
        const headers = {'Content-Type':'application/json'};
        if(tok){ headers['Authorization'] = 'Bearer '+tok; }
        const r = await fetch('/presign', { method:'POST', headers: headers, body: JSON.stringify({ s3_path: s3, expires:3600, inline:true })});
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
function escapeAttr(s){ return escapeHtml(s).replace(/"/g,'&quot'); }
async function submit(){
  const q=document.getElementById('query').value.trim();
  if(!q){ document.getElementById('result').innerHTML='<div class="bg-red-100 p-3 rounded">Query required</div>'; return; }
  let top_k = 5;
  if(DISPLAY_TOPK){
    top_k = parseInt(document.getElementById('top_k').value||'5',10);
  }
  const enable_tracing = DISPLAY_SOURCES ? document.getElementById('enable_tracing').checked===true : false;
  const payload={ query: q, top_k, enable_tracing, return_chunks: enable_tracing };
  document.getElementById('ask').disabled=true; document.getElementById('ask').innerText='Asking...';
  try{
    const tok = localStorage.getItem('app_jwt');
    const headers = {'Content-Type':'application/json'};
    if(tok){ headers['Authorization'] = 'Bearer '+tok; }
    const resp = await fetch('/run', { method:'POST', headers: headers, body: JSON.stringify(payload) });
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
  checkAuth();
  document.getElementById('ask').addEventListener('click', submit);
});
</script>
</body>
</html>
"""
env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
tmpl = env.from_string(INDEX_TEMPLATE)
INDEX_HTML = tmpl.render(display_sources=DISPLAY_SOURCES, display_topk=DISPLAY_TOPK, require_auth=REQUIRE_AUTH_UI)

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML)

@app.get("/health")
async def health():
    return {"status":"ok","query_url":QUERY_URL}

@app.post("/run")
async def run(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(body, dict) or not body.get("query"):
        raise HTTPException(status_code=400, detail="Missing 'query'")
    target = QUERY_URL.rstrip("/") + "/generate"
    headers = {"Content-Type":"application/json"}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    client_host = request.client.host if request.client else None
    if client_host:
        headers["X-Forwarded-For"] = client_host
    import httpx
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(target, json=body, headers=headers)
            content_type = resp.headers.get("content-type","")
            if "application/json" in (content_type or ""):
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            else:
                return PlainTextResponse(content=resp.text, status_code=resp.status_code)
    except httpx.HTTPStatusError as e:
        log.error("Upstream returned non-200: %s", str(e))
        raise HTTPException(status_code=502, detail=f"Upstream error: {getattr(e.response,'status_code','unknown')}")
    except Exception:
        log.exception("Upstream call failed")
        raise HTTPException(status_code=502, detail="Upstream call failed")

@app.post("/presign")
async def presign(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    target = QUERY_URL.rstrip("/") + "/presign"
    headers = {"Content-Type":"application/json"}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    import httpx
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(target, json=body, headers=headers)
            try:
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
            except Exception:
                return PlainTextResponse(content=resp.text, status_code=resp.status_code)
    except Exception:
        log.exception("Presign proxy failed")
        raise HTTPException(status_code=502, detail="Presign proxy failed")
