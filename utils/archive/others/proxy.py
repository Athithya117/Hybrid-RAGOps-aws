import os,sys,json,logging
from fastapi import FastAPI, Request, HTTPException
import httpx
import uvicorn
logging.basicConfig(stream=sys.stdout, level=os.getenv("LOG_LEVEL","INFO"))
logger=logging.getLogger("mistral_proxy")
MISTRAL_KEY=os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_KEY:
    logger.warning("MISTRAL_API_KEY not set; proxy will return 401 for upstream calls")
UPSTREAM_BASE="https://api.mistral.ai/v1"
app=FastAPI()
@app.get("/health")
async def health():
    return {"ok": True}
def _to_mistral_payload(body:dict)->dict:
    model=body.get("model") or body.get("model_id") or body.get("model_name") or "mistral-small"
    prompt=body.get("prompt")
    messages=body.get("messages")
    if messages:
        return {"model": model, "messages": messages}
    if prompt is None:
        # try other fields
        if "input" in body: prompt=body.get("input")
    if prompt is None:
        raise ValueError("provide 'model' and either 'prompt' or 'messages'")
    return {"model": model, "messages":[{"role":"user","content": str(prompt)}]}
def _extract_text_from_mistral(resp_json:dict)->str:
    # handle common shapes; be conservative
    if not isinstance(resp_json, dict):
        return json.dumps(resp_json)
    # chat completions style
    choices=resp_json.get("choices") or resp_json.get("outputs") or resp_json.get("output")
    if isinstance(choices, list) and len(choices)>0:
        first=choices[0]
        # new style: {'message': {'content': '...'}}
        if isinstance(first, dict):
            m=first.get("message")
            if isinstance(m, dict):
                c=m.get("content")
                if isinstance(c, str): return c
            # sometimes: {'content': '...'} or {'text':'...'}
            if "content" in first and isinstance(first.get("content"), str): return first.get("content")
            if "text" in first and isinstance(first.get("text"), str): return first.get("text")
            # some providers nest 'delta' -> 'content'
            if first.get("delta") and isinstance(first["delta"], dict) and isinstance(first["delta"].get("content"), str):
                return first["delta"]["content"]
    # fallback for other keys
    if "output" in resp_json and isinstance(resp_json["output"], str): return resp_json["output"]
    if "text" in resp_json and isinstance(resp_json["text"], str): return resp_json["text"]
    # best-effort: join any string fields in top-level
    strings=[]
    for k,v in resp_json.items():
        if isinstance(v,str): strings.append(v)
    if strings:
        return "\n".join(strings)
    return json.dumps(resp_json)
@app.post("/generate")
async def generate(request: Request):
    if not MISTRAL_KEY:
        raise HTTPException(status_code=401,detail="MISTRAL_API_KEY not configured in proxy")
    try:
        body=await request.json()
    except Exception:
        raise HTTPException(status_code=400,detail="invalid json")
    try:
        payload=_to_mistral_payload(body)
    except ValueError as e:
        raise HTTPException(status_code=400,detail=str(e))
    headers={"Authorization": f"Bearer {MISTRAL_KEY}", "Content-Type":"application/json"}
    url=f"{UPSTREAM_BASE}/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r=await client.post(url, json=payload, headers=headers)
        except httpx.RequestError as e:
            logger.exception("upstream request error: %s", e)
            raise HTTPException(status_code=502,detail="upstream request failed")
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError:
        txt=r.text or r.content.decode(errors="ignore")
        logger.warning("upstream returned status %s: %s", r.status_code, txt[:1000])
        raise HTTPException(status_code=502,detail=f"upstream {r.status_code}: {txt[:200]}")
    try:
        j=r.json()
    except Exception:
        return {"text": r.text}
    out_text=_extract_text_from_mistral(j)
    return {"text": out_text, "mistral_raw": j}
if __name__=="__main__":
    uvicorn.run("proxy:app", host="0.0.0.0", port=int(os.getenv("PORT","9000")), log_level="info")
