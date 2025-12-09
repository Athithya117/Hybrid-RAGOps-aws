# apps/inference/frontend/frontend_ui.py
import os
import logging
import jwt
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from jinja2 import Template
import httpx

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("frontend_ui")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("Missing required environment variable: JWT_SECRET")

JWT_ISS = os.getenv("JWT_ISS", "stateless-openid-auth")
JWT_AUD = os.getenv("JWT_AUD", "rag-ui")
QUERY_URL = os.getenv("QUERY_URL", "").rstrip("/")
TAILWIND = "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"

INDEX_HTML = Template("""<!doctype html>
<html>
<head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<link href='{{ tailwind }}' rel='stylesheet'>
<title>RAG Query UI</title>
</head>
<body class='bg-gray-50 min-h-screen p-6 font-sans'>
<div class='max-w-4xl mx-auto relative'>
  <div class='flex justify-between items-center mb-4'>
    <h1 class='text-2xl font-semibold'>RAG Query UI</h1>
    <div id='auth-top' class='text-sm'></div>
  </div>

  <div id='main-hero' class='mb-4'>
    <div id='login-cta' class='hidden bg-white p-6 rounded shadow flex items-center justify-between'>
      <div><h2 class='text-lg font-medium'>Sign in required</h2><p class='text-sm text-gray-600'>Sign in to ask questions and view sources.</p></div>
      <div><button id='hero-signin' class='bg-blue-600 text-white px-4 py-2 rounded'>Sign in</button></div>
    </div>
  </div>

  <form id='qry' class='space-y-4 bg-white p-4 rounded shadow'>
    <div><label class='block text-sm font-medium text-gray-700'>Query</label>
    <textarea id='query' name='query' required rows='3' class='mt-1 block w-full border rounded p-2'></textarea></div>
    <div class='flex items-center space-x-4'>
      <label class='flex items-center space-x-2'><input type='checkbox' id='enable_tracing' name='enable_tracing' value='true' class='form-checkbox'/>
      <span class='text-sm'>Enable tracing</span></label>
      <label class='text-sm'>Top K <input id='top_k' type='number' name='top_k' value='5' min='1' max='50' class='ml-2 w-20 border rounded p-1 text-sm'/></label>
    </div>
    <div><button id='ask' type='button' class='bg-blue-600 text-white px-4 py-2 rounded'>Ask</button></div>
  </form>

  <div id='result' class='mt-6'></div>
</div>

<style>
  #logout-fab { position: fixed; left: 12px; bottom: 12px; z-index: 1000; display:none; }
</style>

<script>
const LOGIN_PAGE = '/auth/login';
const LOGOUT_PATH = '/auth/logout';
function esc(s){return (s+'').replace(/[&<>"']/g, function(m){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[m];});}
function parseJwt(t){ try{ const p=t.split('.')[1]; const b=p.replace(/-/g,'+').replace(/_/g,'/'); const j=decodeURIComponent(Array.prototype.map.call(atob(b),c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)).join('')); return JSON.parse(j);}catch(e){return null;}}

// show top-right signed-in UI and wire logout
function showTop(user){
  const el=document.getElementById('auth-top');
  if(!el) return;
  if(user){
    el.innerHTML = '<div class="flex items-center space-x-3"><div class="text-sm text-gray-800">Signed in as <strong>'+esc(user)+'</strong></div><button id="top-logout" class="ml-2 inline-flex items-center bg-gray-100 px-2 py-1 rounded text-sm">Sign out</button></div>';
    const b=document.getElementById('top-logout'); if(b) b.addEventListener('click', signOut);
  } else {
    el.innerHTML = '<button id="top-signin" class="bg-blue-600 text-white px-3 py-1 rounded text-sm">Sign in</button>';
    const b=document.getElementById('top-signin'); if(b) b.addEventListener('click', ()=> window.location.href = LOGIN_PAGE);
  }
}

// Validate token server-side and refresh UI accordingly
async function validateToken(token){
  try{
    const resp = await fetch('/auth/me', { headers: { 'Authorization': 'Bearer ' + token } });
    if(resp.ok){
      const j = await resp.json();
      const user = j.user || {};
      const display = user.email || user.name || user.sub || 'User';
      showTop(display);
      const cta = document.getElementById('login-cta'); if(cta) cta.classList.add('hidden');
      return true;
    }
  }catch(e){
    console.debug('validateToken failed', e);
  }
  // invalid or expired token -> clear and show sign-in
  try{ localStorage.removeItem('app_jwt'); }catch(e){}
  showTop(null);
  const cta = document.getElementById('login-cta'); if(cta) cta.classList.remove('hidden');
  return false;
}

function refreshUi(){
  const t = localStorage.getItem('app_jwt');
  if(!t){
    showTop(null);
    const cta = document.getElementById('login-cta'); if(cta) cta.classList.remove('hidden');
  } else {
    // parse locally first for instant UX, then validate with server
    const p = parseJwt(t) || {};
    const display = p.email || p.name || p.sub || 'User';
    showTop(display);
    // now validate with server and react if invalid
    validateToken(t).then(ok => {
      if(!ok){
        // already handled in validateToken (clears token)
      }
    });
  }
}

function signOut(){
  try{ localStorage.removeItem('app_jwt'); }catch(e){}
  // Navigate to /auth/logout which will also clear and redirect to root.
  window.location.replace(LOGOUT_PATH);
}

async function submit(){
  const q=document.getElementById('query').value.trim();
  if(!q){ document.getElementById('result').innerHTML = '<div class="bg-red-100 p-3 rounded">Query required</div>'; return; }
  const token = localStorage.getItem('app_jwt');
  if(!token){ window.location.href = LOGIN_PAGE; return; }
  const top_k = parseInt(document.getElementById('top_k').value || '5',10);
  if(Number.isNaN(top_k) || top_k < 1 || top_k > 100){ document.getElementById('result').innerHTML = '<div class="bg-red-100 p-3 rounded">top_k must be between 1 and 100</div>'; return; }
  const enable_tracing = document.getElementById('enable_tracing').checked === true;
  const payload = { query: q, top_k: top_k, enable_tracing: enable_tracing, return_chunks: enable_tracing };
  document.getElementById('ask').disabled = true; document.getElementById('ask').textContent = 'Asking...';
  try {
    const resp = await fetch('/run', { method:'POST', headers:{ 'Content-Type':'application/json', 'Authorization':'Bearer ' + token }, body: JSON.stringify(payload), credentials:'same-origin' });
    if(resp.status === 401){ try{ localStorage.removeItem('app_jwt'); }catch(e){} window.location.href = LOGIN_PAGE; return; }
    if(!resp.ok){
      const txt = await resp.text().catch(()=>resp.statusText);
      document.getElementById('result').innerHTML = '<div class="bg-red-100 p-3 rounded">Query failed: '+esc(String(txt))+'</div>';
      return;
    }
    const html = await resp.text();
    document.getElementById('result').innerHTML = html;
  } catch(err){
    document.getElementById('result').innerHTML = '<div class="bg-red-100 p-3 rounded">Query failed: '+esc(String(err))+'</div>';
  } finally {
    document.getElementById('ask').disabled = false; document.getElementById('ask').textContent = 'Ask';
  }
}

document.addEventListener('DOMContentLoaded', function(){
  refreshUi();
  const askBtn = document.getElementById('ask'); if(askBtn) askBtn.addEventListener('click', submit);
  const heroBtn = document.getElementById('hero-signin'); if(heroBtn) heroBtn.addEventListener('click', ()=> window.location.href = LOGIN_PAGE);
});
</script>
</body>
</html>
""")

RESULT_HTML = Template("""
<div class='bg-white p-4 rounded shadow'><h2 class='font-medium mb-2'>Answer</h2>
<div class='prose'><pre class='whitespace-pre-wrap'>{{ answer }}</pre></div>
{% if chunks %}
  <h3 class='mt-4 font-medium'>Sources</h3>
  <ul class='space-y-2'>
  {% for c in chunks %}
    <li class='p-2 border rounded'>
      <div class='text-sm font-medium'>[{{ c.index }}] {{ c.title or 'source' }}</div>
      <div class='mt-1 text-xs text-gray-700 whitespace-pre-wrap'>{{ c.preview | e }}</div>
    </li>
  {% endfor %}
  </ul>
{% endif %}
</div>
""")

app = FastAPI(title="frontend-ui")

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(INDEX_HTML.render(tailwind=TAILWIND))

@app.post("/run")
async def run(request: Request):
    auth = request.headers.get("authorization", "")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = auth.split(" ", 1)[1].strip()
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"], audience=JWT_AUD, issuer=JWT_ISS)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    body = await request.json()
    q = (body.get("query") or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query required")
    try:
        top_k = int(body.get("top_k", 5))
    except Exception:
        raise HTTPException(status_code=400, detail="top_k must be an integer")
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k out of allowed range (1-100)")

    if not QUERY_URL:
        raise HTTPException(status_code=502, detail="QUERY_URL not configured")

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(f"{QUERY_URL}/generate", json=body, headers={"Authorization": f"Bearer {token}"})
            resp.raise_for_status()
            j = resp.json()
    except httpx.HTTPStatusError as e:
        log.error("Backend error: %s", e.response.text[:1000])
        raise HTTPException(status_code=502, detail=f"Backend error: {e.response.status_code}")
    except Exception as e:
        log.exception("backend call failed")
        raise HTTPException(status_code=502, detail="Backend call failed")

    answer = j.get("answer", "")
    chunks = j.get("chunks") or []
    normalized = []
    for idx, c in enumerate(chunks):
        preview = ""
        title = None
        if isinstance(c, dict):
            meta = c.get("meta") or {}
            preview = meta.get("content") or c.get("text") or c.get("content") or ""
            title = meta.get("title") or c.get("title")
        else:
            preview = str(c)
        normalized.append({"index": idx + 1, "preview": preview, "title": title})
    return HTMLResponse(RESULT_HTML.render(answer=answer, chunks=normalized))

@app.get("/auth/fragment/providers", response_class=HTMLResponse)
async def providers_fragment():
    return HTMLResponse("<div class='space-x-2'><a class='inline-block bg-white border rounded px-3 py-1 mr-2' href='/auth/login'>Sign in</a></div>")

@app.get("/auth/fragment/status", response_class=HTMLResponse)
async def status_fragment():
    return HTMLResponse("<div class='text-sm text-gray-700'>Not signed in. <a href='/auth/login' class='text-blue-600 underline'>Sign in</a></div>")

@app.get("/health")
async def health():
    return {"status": "ok", "service": "frontend"}
