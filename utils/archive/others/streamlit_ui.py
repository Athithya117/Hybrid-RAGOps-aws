import os,streamlit as st
from streamlit.components.v1 import html
API_URL=os.getenv("INFERENCE_API","/retrieve")
st.set_page_config(page_title="RAG8s Console",page_icon="💬",layout="wide")
st.title("RAG8s Console")
component_html=f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<style>
body{{font-family:system-ui,Arial;margin:0;padding:0}}
.container{{padding:16px;max-width:1100px;margin:0 auto}}
.header{{display:flex;gap:8px;align-items:center;margin-bottom:12px}}
.input{{flex:1;display:flex;gap:8px}}
textarea{{width:100%;height:84px;padding:8px;font-size:14px}}
button{{padding:8px 12px;font-size:14px;cursor:pointer}}
.chat{{margin-top:12px}}
.message{{border-radius:8px;padding:10px;margin-bottom:8px}}
.user{{background:#e6f0ff;border:1px solid #cde0ff}}
.assistant{{background:#f3f3f3;border:1px solid #ddd}}
.docs{{margin-top:8px;padding:8px;border-left:3px solid #ddd;background:#fff}}
.meta{{font-size:12px;color:#666;margin-bottom:6px}}
</style>
</head>
<body>
<div class="container">
<div class="header">
<div style="font-weight:600">RAG8s Console (client-side)</div>
</div>
<div class="input">
<textarea id="prompt" placeholder="Ask me anything..."></textarea>
<button id="send">Ask</button>
</div>
<div id="status" style="margin-top:8px;font-size:13px;color:#333"></div>
<div class="chat" id="chat"></div>
</div>
<script>
const apiUrl=decodeURIComponent("{{API}}");
function escapeHtml(s){return s.replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
document.getElementById("send").onclick=async function(){
  const prompt=document.getElementById("prompt").value.trim();
  if(!prompt) return;
  const chat=document.getElementById("chat");
  const userNode=document.createElement("div");userNode.className="message user";userNode.innerText=prompt;chat.prepend(userNode);
  document.getElementById("status").innerText="Retrieving...";
  try{
    const resp=await fetch(apiUrl,{method:"POST",credentials:"include",headers:{"Content-Type":"application/json"},body:JSON.stringify({query:prompt,do_presign:true})});
    if(!resp.ok){
      const txt=await resp.text();
      throw new Error(resp.status+" "+resp.statusText+": "+txt.slice(0,200));
    }
    const data=await resp.json();
    const ans=data.answer||"(no answer)";
    const aNode=document.createElement("div");aNode.className="message assistant";aNode.innerText=ans;chat.prepend(aNode);
    const docs=data.docs||[];
    if(docs.length){
      const dwrap=document.createElement("div");dwrap.className="docs";
      docs.forEach(function(d,i){
        const meta=document.createElement("div");meta.className="meta";
        const m=d.metadata||{};
        const title=m.title||m.file_name||("doc_"+i);
        const parts=[title];
        if(m.source_url) parts.push("[source]");
        if(m.signed_url && m.signed_url!==m.source_url) parts.push("[signed]");
        meta.innerText=parts.join(" • ");
        const txt=document.createElement("pre");txt.style.whiteSpace="pre-wrap";txt.style.fontFamily="inherit";txt.style.margin="6px 0";txt.innerText=d.text||"";
        dwrap.appendChild(meta);dwrap.appendChild(txt);
      });
      chat.prepend(dwrap);
    } else {
      const meta=document.createElement("div");meta.className="meta";meta.innerText="Retrieved counts: "+JSON.stringify(data.retrieval||{});
      chat.prepend(meta);
    }
  }catch(e){
    const errNode=document.createElement("div");errNode.className="message assistant";errNode.style.borderColor="#f5c6cb";errNode.style.background="#fff1f0";errNode.innerText="Error: "+e.message;
    document.getElementById("chat").prepend(errNode);
  }finally{
    document.getElementById("status").innerText="";
    document.getElementById("prompt").value="";
  }
};
(function(){const el=document.getElementById("chat");if(!el.children.length){const info=document.createElement("div");info.className="meta";info.innerText="You are authenticated in the browser; requests use browser credentials and will be routed through oauth2-proxy.";el.appendChild(info);}})();
</script>
</body>
</html>
""".replace("{{API}}",API_URL)
html(component_html,height=700)
