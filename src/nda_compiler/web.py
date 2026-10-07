"""Compliance officer's page: upload an NDA, pick the policies, apply them.

Two steps, deliberately. `/propose` compiles the NDA and returns every control
the NDA supports, in plain English next to the clause it came from, with the
judge's confidence — but creates nothing. `/apply` takes the officer's
selection and creates exactly those on OpenBox. The officer's choice is the
approval step; nothing reaches an agent without it.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .bindings import load_bindings
from .extract import extractor_from_env
from .governance import maybe_govern
from .graph import Services, build_graph, compile_nda
from .jev import judge_from_env
from .models import CompileReport
from .openbox_api import RecordingBackend, apply_all, backend_from_env
from .templates import describe

app = FastAPI(title="NDA → OpenBox")
BINDINGS_DIR = Path(os.environ.get("BINDINGS_DIR", "bindings"))

# Drafts live for the life of the process; this is a single-officer trial.
_DRAFTS: dict[str, tuple[CompileReport, str]] = {}


class ApplyRequest(BaseModel):
    draft_id: str
    selected: list[int]


_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>NDA Policies</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fff;--fg:#111;--mut:#666;--line:#e5e5e5;--ok:#1a7f37;--warn:#b35900;--bad:#b3261e;--acc:#1d4ed8}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mut:#999;--line:#333;--acc:#60a5fa}}
body{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui}
main{max-width:1100px;margin:auto}h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--mut);margin:0 0 16px}
form{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font:inherit;cursor:pointer}
button[disabled]{opacity:.5;cursor:default}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:14px}
td,th{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}.mut{color:var(--mut)}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:0 8px;font-size:12px;color:var(--mut)}
#cov{margin:16px 0;padding:12px;border:1px solid var(--line);border-radius:8px}
#bar{display:flex;gap:12px;align-items:center;margin-top:12px}
</style></head><body><main>
<h1>NDA → OpenBox policies</h1>
<p class="sub">Upload the NDA. Review what it would enforce. Tick what you want. Apply.</p>
<form id="f"><input type="file" name="nda" required>
<select name="matter">__MATTERS__</select><button>Read NDA</button><span id="t" class="mut"></span></form>
<div id="cov" hidden></div>
<table id="r"></table>
<div id="bar" hidden><button id="apply">Apply selected</button><span id="s" class="mut"></span></div>
<script>
const $=id=>document.getElementById(id);let draft=null;
$('f').onsubmit=async e=>{e.preventDefault();$('t').textContent='reading…';$('r').innerHTML='';$('cov').hidden=true;$('bar').hidden=true;
const s=performance.now();const res=await fetch('/propose',{method:'POST',body:new FormData($('f'))});const d=await res.json();
if(!res.ok){$('t').textContent=d.detail||'failed';return}
draft=d;$('t').textContent=`${Math.round(performance.now()-s)} ms · `+d.timings.map(x=>`${x.stage} ${Math.round(x.ms)}ms`).join(' · ');
const c=d.coverage;$('cov').hidden=false;$('cov').innerHTML=`<b>${c.clauses}</b> clauses · <b>${d.controls.length}</b> policies available · <b class=mut>${c.not_enforceable}</b> clauses outside runtime scope (term, return/destroy, governing law)`;
const q={};d.clauses.forEach(x=>q[x.id]=x);
$('r').innerHTML='<tr><th></th><th>Clause</th><th>Policy</th><th>Kind</th><th>Judge</th><th>Status</th></tr>'+d.controls.map((x,i)=>{
const cl=q[x.clause_id];const chk=x.verify_probability>=0.8?'checked':'';
return `<tr><td><input type=checkbox data-i=${i} ${chk}></td>
<td><b>§${x.clause_id} ${cl.heading}</b><br><span class=mut>${cl.text}</span></td>
<td>${x.description}</td><td><span class=pill>${x.type.replace('_',' ')}</span></td>
<td>${x.verify_probability!=null?x.verify_probability.toFixed(2):'-'}</td><td id=st${i} class=mut>proposed</td></tr>`}).join('');
$('bar').hidden=false;$('s').textContent='';};
$('apply').onclick=async()=>{const sel=[...document.querySelectorAll('input[type=checkbox]:checked')].map(b=>+b.dataset.i);
if(!sel.length){$('s').textContent='nothing selected';return}
$('apply').disabled=true;$('s').textContent=`applying ${sel.length}…`;const s=performance.now();
const res=await fetch('/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({draft_id:draft.draft_id,selected:sel})});
const d=await res.json();$('apply').disabled=false;
if(!res.ok){$('s').textContent=d.detail||'failed';return}
d.results.forEach(r=>{const el=$('st'+r.index);el.textContent=r.status+(r.note?' · '+r.note:'');el.className=r.status==='active'?'ok':r.status==='failed'?'bad':'warn'});
$('s').textContent=`${d.results.filter(r=>r.status==='active').length}/${sel.length} active on OpenBox · ${Math.round(performance.now()-s)} ms`};
</script></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    options = "".join(
        f'<option value="{p.stem}">{p.stem}</option>' for p in sorted(BINDINGS_DIR.glob("*.yaml"))
    )
    return _PAGE.replace("__MATTERS__", options)


def _report_json(report: CompileReport, bindings) -> dict[str, Any]:
    body = report.model_dump(mode="json")
    body["coverage"] = report.coverage
    for control, raw in zip(report.controls, body["controls"], strict=True):
        raw["description"] = describe(control, bindings)
    return body


@app.post("/propose")
async def propose(nda: UploadFile = File(...), matter: str = Form(...)) -> JSONResponse:
    bindings_path = BINDINGS_DIR / f"{matter}.yaml"
    if not bindings_path.exists():
        raise HTTPException(404, f"unknown matter {matter}")
    bindings = load_bindings(bindings_path)
    suffix = Path(nda.filename or "nda.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(await nda.read())
        source = Path(handle.name)
    # Propose only: a recording backend means nothing is created yet.
    services = Services(judge_from_env(), extractor_from_env(), RecordingBackend(), bindings)
    try:
        report = await compile_nda(source, services, maybe_govern(build_graph(services)))
    finally:
        source.unlink(missing_ok=True)
    # Reset the recording backend's pretend statuses; the officer decides.
    report.controls = [
        c.model_copy(update={"status": "draft", "remote_id": None}) for c in report.controls
    ]
    draft_id = uuid.uuid4().hex[:12]
    _DRAFTS[draft_id] = (report, matter)
    body = _report_json(report, bindings)
    body["draft_id"] = draft_id
    return JSONResponse(body)


@app.post("/apply")
async def apply(req: ApplyRequest) -> JSONResponse:
    entry = _DRAFTS.get(req.draft_id)
    if entry is None:
        raise HTTPException(404, "draft expired; read the NDA again")
    report, _matter = entry
    chosen = [(i, report.controls[i]) for i in req.selected if 0 <= i < len(report.controls)]
    backend = backend_from_env()
    try:
        applied = await apply_all(backend, [c for _, c in chosen])
    finally:
        await backend.close()
    results = []
    for (index, _), control in zip(chosen, applied, strict=True):
        report.controls[index] = control
        results.append(
            {
                "index": index,
                "status": control.status,
                "remote_id": control.remote_id,
                "note": control.note,
            }
        )
    return JSONResponse({"results": results})


def run() -> None:
    load_dotenv()
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("NDA_WEB_PORT", "8010")))
