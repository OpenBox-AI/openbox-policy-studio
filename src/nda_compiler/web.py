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

import httpx
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


_STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/openbox/state")
async def openbox_state(matter: str = "trial") -> JSONResponse:
    """What is enforced on the matter's agent(s) right now, straight from OpenBox."""

    bindings = load_bindings(BINDINGS_DIR / f"{matter}.yaml")
    base = os.environ.get("OPENBOX_BACKEND_URL", "http://localhost:3000").rstrip("/")
    key = os.environ.get("OPENBOX_ORG_API_KEY", "").strip()
    opa = os.environ.get("OPA_URL", "http://localhost:8181").rstrip("/")
    items: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10) as client:
        for agent in bindings.all_agents:
            opa_raw = ""
            try:
                policies = (await client.get(f"{opa}/v1/policies")).json().get("result", [])
                opa_raw = "".join(
                    p.get("raw", "")
                    for p in policies
                    if agent.id.replace("-", "") in p.get("id", "")
                )
            except Exception:
                pass
            for kind, path, name_key in (
                ("access rule", "policy-rule", "rule_name"),
                ("sequence rule", "behavior-rule", "rule_name"),
                ("output scan", "guardrails", "name"),
            ):
                if not key:
                    continue
                try:
                    r = await client.get(
                        f"{base}/agent/{agent.id}/{path}",
                        headers={"X-API-Key": key},
                        params={"limit": 200},
                    )
                    page = r.json().get("data", {})
                    rows = page.get("data", page) if isinstance(page, dict) else page
                except Exception:
                    rows = []
                for row in rows or []:
                    if row.get("is_active") is False or row.get("is_current_version") is False:
                        continue
                    name = row.get(name_key, "")
                    clause = name.split("§")[1].split(" ")[0] if "§" in name else ""
                    items.append(
                        {
                            "id": row.get("id"),
                            "name": name,
                            "kind": kind,
                            "clause": f"§{clause}" if clause else "",
                            "agent": agent.name,
                            "opa_loaded": bool(row.get("id") and row["id"] in opa_raw),
                        }
                    )
    return JSONResponse(
        {
            "agent_name": ", ".join(a.name for a in bindings.all_agents),
            "backend": base.replace("http://", "").replace("https://", ""),
            "items": items,
        }
    )


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
