"""Upload page: drop an NDA, pick a matter, watch rules land."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .bindings import load_bindings
from .extract import extractor_from_env
from .governance import maybe_govern
from .graph import Services, build_graph, compile_nda
from .jev import judge_from_env
from .openbox_api import backend_from_env

app = FastAPI(title="NDA → OpenBox")
BINDINGS_DIR = Path(os.environ.get("BINDINGS_DIR", "bindings"))

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>NDA → OpenBox</title>
<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#e5e5e5;--ok:#1a7f37;--warn:#b35900;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#999;--line:#333}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui}
main{max-width:1000px;margin:auto}h1{font-size:22px}form{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:14px}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.mut{color:var(--mut)}code{font-size:12px}
#cov{margin-top:16px;padding:12px;border:1px solid var(--line);border-radius:8px}
</style></head><body><main>
<h1>NDA → OpenBox controls</h1>
<form id="f"><input type="file" name="nda" required><select name="matter">__MATTERS__</select>
<button>Compile</button><span id="t" class="mut"></span></form>
<div id="cov"></div><table id="r"></table>
<script>
const f=document.getElementById('f'),t=document.getElementById('t'),r=document.getElementById('r'),cov=document.getElementById('cov');
f.onsubmit=async e=>{e.preventDefault();t.textContent='compiling…';r.innerHTML='';cov.innerHTML='';
const s=performance.now();const res=await fetch('/compile',{method:'POST',body:new FormData(f)});const d=await res.json();
t.textContent=`${Math.round(performance.now()-s)} ms end to end · `+d.timings.map(x=>`${x.stage} ${Math.round(x.ms)}ms`).join(' · ');
const c=d.coverage;cov.innerHTML=`<b>${c.clauses}</b> clauses · <b class=ok>${c.enforced}</b> enforced at runtime · <b>${c.judgement}</b> under judgement · <b class=mut>${c.not_enforceable}</b> outside runtime scope · <b class=warn>${c.review}</b> for review`
+(d.review.length?'<ul>'+d.review.map(x=>`<li class=warn>${x}</li>`).join('')+'</ul>':'');
const q={};d.clauses.forEach(x=>q[x.id]=x);
r.innerHTML='<tr><th>Clause</th><th>Control</th><th>Judge</th><th>Status</th></tr>'+d.controls.map(x=>{
const cl=q[x.clause_id];const p=x.payload;const cls=x.status==='active'?'ok':x.status==='review'?'warn':x.status==='failed'?'bad':'';
return `<tr><td><b>§${x.clause_id} ${cl.heading}</b><br><span class=mut>${cl.text.slice(0,220)}…</span></td>
<td><b>${x.type}</b> · ${p.rule_name||p.name}<br><code>${JSON.stringify(p.conditions||p.states||p.params||p.question).slice(0,260)}</code></td>
<td>${x.verify_probability?.toFixed(2)??'-'}</td><td class=${cls}>${x.status}${x.note?'<br><span class=mut>'+x.note+'</span>':''}</td></tr>`}).join('')};
</script></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    options = "".join(
        f'<option value="{p.stem}">{p.stem}</option>' for p in sorted(BINDINGS_DIR.glob("*.yaml"))
    )
    return _PAGE.replace("__MATTERS__", options)


@app.post("/compile")
async def compile_endpoint(nda: UploadFile = File(...), matter: str = Form(...)) -> JSONResponse:
    bindings = load_bindings(BINDINGS_DIR / f"{matter}.yaml")
    suffix = Path(nda.filename or "nda.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(await nda.read())
        source = Path(handle.name)
    services = Services(judge_from_env(), extractor_from_env(), backend_from_env(), bindings)
    try:
        report = await compile_nda(source, services, maybe_govern(build_graph(services)))
    finally:
        await services.backend.close()
        source.unlink(missing_ok=True)
    body = report.model_dump(mode="json")
    body["coverage"] = report.coverage
    return JSONResponse(body)


def run() -> None:
    load_dotenv()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("NDA_WEB_PORT", "8010")))
