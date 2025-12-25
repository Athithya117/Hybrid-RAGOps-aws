# apps/inference/frontend/stateless_openid_auth.py
import os,time,logging,json,html
from typing import Any,List,Optional,Dict
from fastapi import FastAPI,Request,HTTPException
from fastapi.responses import HTMLResponse,RedirectResponse,JSONResponse
from authlib.integrations.starlette_client import OAuth,OAuthError
from authlib.jose.errors import InvalidClaimError
import httpx
import jwt
from config import (get_redirect,enabled_flags,enabled_providers_effective,GOOGLE_CLIENT_ID,GOOGLE_CLIENT_SECRET,MS_CLIENT_ID,MS_CLIENT_SECRET,GITHUB_CLIENT_ID,GITHUB_CLIENT_SECRET,GOOGLE_ALLOWED_DOMAINS,GITHUB_ALLOWED_ORGS,MS_TENANT_ID,MICROSOFT_ALLOWED_TENANT_IDS,MICROSOFT_ALLOWED_DOMAINS)
logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"))
log = logging.getLogger("stateless_openid_auth")
JWT_SECRET = os.getenv("JWT_SECRET") or ""
if not JWT_SECRET:
    log.warning("JWT_SECRET not set; generating ephemeral secret (NOT for production).")
    JWT_SECRET = os.urandom(32).hex()
JWT_EXP_SECONDS = int(os.getenv("JWT_EXP_SECONDS","1800"))
JWT_ISS = os.getenv("JWT_ISS","stateless-openid-auth")
JWT_AUD = os.getenv("JWT_AUD","rag-ui")
flags = enabled_flags()
effective = enabled_providers_effective()
if any(flags.values()) and not effective:
    log.info("Provider flags set but no provider had both CLIENT_ID and CLIENT_SECRET. Providers will remain disabled until credentials provided.")
_GOOGLE_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg"><path fill="#EA4335" d="M12 10.2v3.6h5.2c-.2 1.2-1.4 3.6-5.2 3.6-3.1 0-5.6-2.6-5.6-5.8S8.9 6.8 12 6.8c1.8 0 2.9.8 3.6 1.5l2.4-2.3C17.2 4 14.8 3 12 3 7.6 3 4 6.6 4 11s3.6 8 8 8c4.6 0 7-3.2 7-7.7 0-.5 0-.9-.1-1.1H12z"/></svg>'
_MICROSOFT_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="2" width="9" height="9" fill="#F35325"/><rect x="13" y="2" width="9" height="9" fill="#81BC06"/><rect x="2" y="13" width="9" height="9" fill="#05A6F0"/><rect x="13" y="13" width="9" height="9" fill="#FFBA08"/></svg>'
_GITHUB_SVG = '<svg viewBox="0 0 24 24" width="18" height="18" xmlns="http://www.w3.org/2000/svg"><path fill="#111" d="M12 .5C5.6.5.5 5.6.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.3.8-.6v-2.2c-3.2.7-3.9-1.4-3.9-1.4-.5-1.1-1.2-1.4-1.2-1.4-1-.7.1-.7.1-.7 1.1.1 1.7 1.1 1.7 1.1 1 .1 1.6.8 2 .6.1-.8.4-1.4.7-1.8-2.6-.3-5.4-1.3-5.4-5.8 0-1.3.5-2.4 1.3-3.2-.1-.3-.6-1.6.1-3.3 0 0 1-.3 3.3 1.3.9-.3 1.9-.5 2.9-.5s2 .2 2.9.5c2.3-1.6 3.3-1.3 3.3-1.3.7 1.7.2 3 .1 3.3.8.8 1.3 1.9 1.3 3.2 0 4.5-2.8 5.5-5.5 5.8.5.4.8 1.1.8 2.3v3.4c0 .3.2.8.8.6 4.6-1.5 7.9-5.8 7.9-10.9C23.5 5.6 18.4.5 12 .5z"/></svg>'
oauth = OAuth()
if "google" in effective:
    oauth.register(name="google",client_id=GOOGLE_CLIENT_ID,client_secret=GOOGLE_CLIENT_SECRET,server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",client_kwargs={"scope":"openid email profile"})
if "microsoft" in effective:
    ms_tenant = MS_TENANT_ID or "common"
    server_metadata = f"https://login.microsoftonline.com/{ms_tenant}/v2.0/.well-known/openid-configuration"
    oauth.register(name="microsoft",client_id=MS_CLIENT_ID,client_secret=MS_CLIENT_SECRET,server_metadata_url=server_metadata,client_kwargs={"scope":"openid email profile offline_access User.Read"})
if "github" in effective:
    oauth.register(name="github",client_id=GITHUB_CLIENT_ID,client_secret=GITHUB_CLIENT_SECRET,access_token_url="https://github.com/login/oauth/access_token",authorize_url="https://github.com/login/oauth/authorize",api_base_url="https://api.github.com/",client_kwargs={"scope":"user:email read:org"})
app = FastAPI(title="stateless-openid-auth")
def _enabled_providers() -> List[str]:
    return enabled_providers_effective()
def _redirect_uri(provider: str) -> str:
    return get_redirect(provider)
@app.get("/redirects",response_class=HTMLResponse)
async def redirects_page():
    provs = _enabled_providers()
    if not provs:
        body = "<div>No providers enabled. Set ENABLE_*_AUTH and provide client credentials.</div>"
    else:
        rows = [f"<li><strong>{p}</strong>: <code>{_redirect_uri(p)}</code></li>" for p in provs]
        body = "<ul>" + "\n".join(rows) + "</ul>"
    html_body = "<!doctype html><html><head><meta charset='utf-8'><title>Redirect URIs</title></head><body><h2>Redirect URIs to register</h2>" + body + "</body></html>"
    return HTMLResponse(html_body)
@app.get("/login",response_class=HTMLResponse)
async def login_page():
    providers = _enabled_providers()
    icons = {"google":_GOOGLE_SVG,"microsoft":_MICROSOFT_SVG,"github":_GITHUB_SVG}
    if not providers:
        btns_html = "<div class='text-sm text-gray-500'>No providers enabled.</div>"
    else:
        btns = []
        for p in providers:
            btns.append("<a href='/auth/login/start/{p}' class='w-full inline-flex items-center justify-center border rounded py-2 px-3 mb-3' aria-label='Continue with {cap}'>{svg}<span style='margin-left:8px'>Continue with {cap}</span></a>".format(p=p,cap=p.capitalize(),svg=icons.get(p,"")))
        btns_html = "\n".join(btns)
    expected_hint = "/auth/callback/<provider>"
    allowed_hint_parts = []
    if "google" in providers and GOOGLE_ALLOWED_DOMAINS:
        allowed_hint_parts.append(f"Google allowed domains: {', '.join(sorted(GOOGLE_ALLOWED_DOMAINS))}")
    if "github" in providers and GITHUB_ALLOWED_ORGS:
        allowed_hint_parts.append(f"GitHub allowed orgs: {', '.join(sorted(GITHUB_ALLOWED_ORGS))}")
    if "microsoft" in providers and MICROSOFT_ALLOWED_TENANT_IDS:
        allowed_hint_parts.append(f"Microsoft allowed tenant ids: {', '.join(sorted(MICROSOFT_ALLOWED_TENANT_IDS))}")
    if "microsoft" in providers and MICROSOFT_ALLOWED_DOMAINS:
        allowed_hint_parts.append(f"Microsoft allowed domains: {', '.join(sorted(MICROSOFT_ALLOWED_DOMAINS))}")
    allowed_hint_html = ""
    if allowed_hint_parts:
        allowed_hint_html = "<div class='mt-3 text-xs text-gray-500'>" + "<br/>".join(html.escape(p) for p in allowed_hint_parts) + "</div>"
    html_page = ("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
                 "<link href='https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css' rel='stylesheet'>"
                 "<title>Sign in</title></head><body class='bg-gray-50 min-h-screen flex items-center justify-center'>"
                 "<div class='max-w-md w-full p-6'><div class='bg-white p-6 rounded shadow'>"
                 "<h1 class='text-xl font-semibold mb-3'>Sign in</h1>"
                 f"{btns_html}"
                 f"<div class='mt-6 text-xs text-gray-400'>Expected redirect URIs follow pattern: <code>{expected_hint}</code></div>"
                 f"{allowed_hint_html}"
                 "</div></div></body></html>")
    return HTMLResponse(html_page)
@app.get("/login/start/{provider}")
async def login_start(request: Request, provider: str):
    provider = provider.lower()
    if provider not in _enabled_providers():
        raise HTTPException(status_code=404,detail="Provider not enabled")
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=500,detail="OAuth client not available")
    redirect_uri = _redirect_uri(provider)
    try:
        return await client.authorize_redirect(request, redirect_uri)
    except AssertionError:
        return HTMLResponse("<h2>Server misconfiguration: SessionMiddleware required</h2>",status_code=500)
    except Exception:
        return HTMLResponse("<h2>OAuth redirect initiation failed — see server logs</h2>",status_code=500)
async def _fetch_userinfo_with_token(provider: str, access_token: str, client: Any = None) -> Dict[str, Any]:
    if not access_token:
        return {}
    if client:
        try:
            resp = await client.get("userinfo", token={"access_token": access_token})
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    try:
        async with httpx.AsyncClient(timeout=10.0) as h:
            headers = {"Authorization":f"Bearer {access_token}","Accept":"application/json"}
            if provider == "microsoft":
                resp = await h.get("https://graph.microsoft.com/v1.0/me?$select=id,displayName,mail,userPrincipalName,tenantId",headers=headers)
                if resp.status_code == 200:
                    return resp.json()
            elif provider == "github":
                resp = await h.get("https://api.github.com/user",headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    email = data.get("email")
                    if not email:
                        resp2 = await h.get("https://api.github.com/user/emails",headers=headers)
                        if resp2.status_code == 200:
                            try:
                                emails = resp2.json()
                                for e in emails:
                                    if e.get("primary") and e.get("verified"):
                                        email = e.get("email")
                                        break
                                if not email:
                                    for e in emails:
                                        if e.get("verified"):
                                            email = e.get("email")
                                            break
                            except Exception:
                                email = email
                    result = {"id":data.get("id"),"node_id":data.get("node_id"),"login":data.get("login"),"name":data.get("name"),"email":email}
                    return result
    except Exception:
        pass
    return {}
def _render_access_denied(title: str, message: str, details: Optional[str] = None, allowed: Optional[str] = None) -> HTMLResponse:
    allowed_html = f"<p>Allowed: <code>{html.escape(allowed)}</code></p>" if allowed else ""
    details_html = f"<div style='margin-top:8px;font-size:90%;color:#666'>{html.escape(details)}</div>" if details else ""
    safe_front = get_redirect("google").rsplit("/auth",1)[0]
    body = ("<!doctype html><html><head><meta charset='utf-8'><title>Access denied</title></head><body>"
            f"<div style='font-family:system-ui,Segoe UI,Roboto,Arial;margin:32px'>"
            f"<h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(message)}</p>"
            f"{allowed_html}"
            f"{details_html}"
            f"<p><a href='{html.escape(safe_front)}'>Return to application</a></p>"
            "</div></body></html>")
    return HTMLResponse(content=body,status_code=403)
@app.get("/callback/{provider}")
async def callback(request: Request, provider: str):
    provider = provider.lower()
    if provider not in _enabled_providers():
        raise HTTPException(status_code=404,detail="Provider not enabled")
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=500,detail="OAuth client not available")
    token = None
    try:
        token = await client.authorize_access_token(request)
    except InvalidClaimError as err:
        log.warning("InvalidClaimError during authorize_access_token: %s -- will attempt manual exchange for provider=%s",err,provider)
        token = None
    except OAuthError as err:
        log.error("OAuthError during authorize_access_token: %s",err)
        token = None
    except Exception as err:
        log.exception("authorize_access_token failed unexpectedly")
        token = None
    if not token or (isinstance(token,dict) and not token.get("access_token") and not token.get("id_token")):
        code = request.query_params.get("code") or ""
        if not code:
            return RedirectResponse(url=f"{get_redirect(provider).rsplit('/auth',1)[0]}/auth/success?error=oauth",status_code=302)
        token_endpoint = None
        try:
            token_endpoint = client.server_metadata.get("token_endpoint")
        except Exception:
            token_endpoint = None
        if not token_endpoint and provider == "microsoft":
            tenant = MS_TENANT_ID or "common"
            token_endpoint = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
        if not token_endpoint:
            log.error("No token endpoint available for manual exchange provider=%s",provider)
            return RedirectResponse(url=f"{get_redirect(provider).rsplit('/auth',1)[0]}/auth/success?error=oauth",status_code=302)
        try:
            async with httpx.AsyncClient(timeout=15.0) as h:
                data = {"grant_type":"authorization_code","code":code,"redirect_uri":_redirect_uri(provider)}
                if provider == "microsoft":
                    data["client_id"] = MS_CLIENT_ID
                    if MS_CLIENT_SECRET:
                        data["client_secret"] = MS_CLIENT_SECRET
                elif provider == "google":
                    data["client_id"] = GOOGLE_CLIENT_ID
                    if GOOGLE_CLIENT_SECRET:
                        data["client_secret"] = GOOGLE_CLIENT_SECRET
                resp = await h.post(token_endpoint,data=data,headers={"Accept":"application/json"})
                resp.raise_for_status()
                token = resp.json()
                log.debug("Manual token exchange result keys: %s",list(token.keys()))
        except Exception as e:
            log.exception("Manual token exchange for %s failed",provider)
            return RedirectResponse(url=f"{get_redirect(provider).rsplit('/auth',1)[0]}/auth/success?error=oauth",status_code=302)
    access_token = token.get("access_token") if isinstance(token,dict) else None
    id_token = token.get("id_token") if isinstance(token,dict) else None
    userinfo = {}
    if id_token:
        try:
            try:
                if client:
                    userinfo = await client.parse_id_token(request,token)
                else:
                    claims = jwt.decode(id_token,options={"verify_signature":False})
                    userinfo = claims
            except Exception:
                claims = jwt.decode(id_token,options={"verify_signature":False})
                userinfo = claims
        except Exception as e:
            log.warning("Failed to parse id_token: %s",e)
            userinfo = {}
    if not userinfo:
        try:
            if client and access_token:
                try:
                    resp = await client.get("userinfo",token={"access_token":access_token})
                    if resp and getattr(resp,"status_code",None) == 200:
                        userinfo = resp.json()
                except Exception:
                    pass
            if not userinfo and access_token:
                userinfo = await _fetch_userinfo_with_token(provider,access_token,client)
        except Exception:
            log.exception("userinfo fetch failed")
            userinfo = {}
    if not userinfo:
        log.warning("No userinfo/claims extracted; aborting oauth callback provider=%s",provider)
        if provider == "github":
            allowed = ", ".join(sorted(GITHUB_ALLOWED_ORGS)) if GITHUB_ALLOWED_ORGS else None
            return _render_access_denied("Access denied","Unable to fetch GitHub profile information from provider.",details="Ensure the OAuth app has correct scopes and the provider is reachable.",allowed=allowed)
        return RedirectResponse(url=f"{get_redirect(provider).rsplit('/auth',1)[0]}/auth/success?error=oauth",status_code=302)
    sub = userinfo.get("sub") or userinfo.get("id") or userinfo.get("node_id")
    email = userinfo.get("email") or userinfo.get("mail") or userinfo.get("userPrincipalName")
    name = userinfo.get("name") or userinfo.get("displayName") or userinfo.get("login") or userinfo.get("preferred_username")
    tenant = None
    try:
        tenant = (userinfo.get("tid") or userinfo.get("tenantId") or (id_token and (jwt.decode(id_token,options={"verify_signature":False}).get("tid"))))
    except Exception:
        tenant = tenant or None
    tenant = tenant.lower() if isinstance(tenant,str) else tenant
    if not sub or not email:
        log.warning("Essential identity fields missing in userinfo: sub=%s email=%s",sub,email)
        if provider == "github":
            allowed = ", ".join(sorted(GITHUB_ALLOWED_ORGS)) if GITHUB_ALLOWED_ORGS else None
            return _render_access_denied("Access denied","GitHub identity did not provide an email address.",details="A verified email is required. Ensure GitHub user has a verified primary email or grant access to user:email scope.",allowed=allowed)
        return RedirectResponse(url=f"{get_redirect(provider).rsplit('/auth',1)[0]}/auth/success?error=oauth",status_code=302)
    safe_front = get_redirect(provider).rsplit("/auth",1)[0]
    if provider == "google" and GOOGLE_ALLOWED_DOMAINS and ("@" in (email or "") and (email.split("@",1)[1].lower() not in GOOGLE_ALLOWED_DOMAINS)):
        allowed = ", ".join(sorted(GOOGLE_ALLOWED_DOMAINS))
        return _render_access_denied("Access denied",f"Your account {email} (domain {email.split('@',1)[1].lower()}) is not permitted to sign in.",details="Contact your administrator or use an allowed account.",allowed=allowed)
    if provider == "github" and GITHUB_ALLOWED_ORGS and access_token:
        try:
            resp = await client.get("user/orgs",token={"access_token":access_token})
            if resp.status_code == 200:
                orgs = resp.json()
                org_names = [o.get("login","").lower() for o in orgs if isinstance(o,dict)]
                if not any(o in GITHUB_ALLOWED_ORGS for o in org_names):
                    allowed = ", ".join(sorted(GITHUB_ALLOWED_ORGS))
                    return _render_access_denied("Access denied",f"Your GitHub account {html.escape(name or email or sub or 'unknown')} is not a member of any allowed organizations.",allowed=allowed)
        except Exception:
            pass
    if provider == "microsoft":
        if MICROSOFT_ALLOWED_TENANT_IDS:
            if not tenant:
                return _render_access_denied("Access denied",f"Your account {html.escape(email or 'unknown')} did not return a tenant id; cannot verify tenant allowlist.",details="Ensure the app registration includes proper scopes and an ID token or Graph returns tenantId.",allowed=", ".join(sorted(MICROSOFT_ALLOWED_TENANT_IDS)))
            if tenant.lower() not in MICROSOFT_ALLOWED_TENANT_IDS:
                return _render_access_denied("Access denied",f"Your Microsoft account tenant {html.escape(str(tenant))} is not allowed.",allowed=", ".join(sorted(MICROSOFT_ALLOWED_TENANT_IDS)))
        if MICROSOFT_ALLOWED_DOMAINS:
            if "@" not in (email or ""):
                return _render_access_denied("Access denied",f"Your Microsoft account {html.escape(email or 'unknown')} has no email-like identifier; cannot verify domain.",allowed=", ".join(sorted(MICROSOFT_ALLOWED_DOMAINS)))
            dom = email.split("@",1)[1].lower()
            if dom not in MICROSOFT_ALLOWED_DOMAINS:
                return _render_access_denied("Access denied",f"Your account {html.escape(email)} (domain {html.escape(dom)}) is not permitted to sign in.",allowed=", ".join(sorted(MICROSOFT_ALLOWED_DOMAINS)))
    payload = {"iss":JWT_ISS,"aud":JWT_AUD,"sub":str(sub),"provider":provider,"email":email,"name":name,"iat":int(time.time()),"exp":int(time.time())+JWT_EXP_SECONDS}
    jwt_token = jwt.encode(payload,JWT_SECRET,algorithm="HS256")
    token_js = json.dumps(jwt_token)
    html_body = ("<!doctype html><html><head><meta charset='utf-8'></head><body>"
                 "<script>try{ var tok = "+token_js+"; try{ localStorage.setItem('app_jwt', tok); }catch(e){} setTimeout(function(){ window.location.replace('"+safe_front+"'); },50);}catch(e){document.body.innerText='Sign-in failed';}</script></body></html>")
    return HTMLResponse(content=html_body)
@app.get("/success",response_class=HTMLResponse)
async def success_page():
    safe_front = get_redirect("google").rsplit("/auth",1)[0]
    return HTMLResponse("<!doctype html><html><head><meta charset='utf-8'></head><body><script>try{window.location.replace('"+safe_front+"');}catch(e){document.body.innerText='Return to app';}</script></body></html>")
@app.get("/me")
async def me(request: Request):
    auth = request.headers.get("authorization","")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401,detail="Missing Authorization header")
    token = auth.split(" ",1)[1].strip()
    try:
        data = jwt.decode(token,JWT_SECRET,algorithms=["HS256"],audience=JWT_AUD,issuer=JWT_ISS)
        return {"authenticated":True,"user":data}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401,detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401,detail="Invalid token")
@app.get("/logout",response_class=HTMLResponse)
async def logout():
    safe_front = get_redirect("google").rsplit("/auth",1)[0]
    return HTMLResponse("<!doctype html><html><head><meta charset='utf-8'></head><body><script>try{localStorage.removeItem('app_jwt')}catch(e){}window.location.replace('"+safe_front+"');</script></body></html>")
@app.get("/health")
async def health():
    return JSONResponse({"status":"ok","service":"stateless-openid-auth","providers":_enabled_providers()})
