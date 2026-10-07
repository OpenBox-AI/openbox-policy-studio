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
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import re

from .bindings import load_bindings
from .extract import extractor_from_env
from .firms import FirmProfile, profile_from_text, write_bindings
from .graph import Services, build_graph, compile_nda
from .jev import judge_from_env
from .models import CompileReport
from .openbox_api import RecordingBackend, apply_all, backend_from_env, fetch_activity_events
from .platform_context import ExistingRule, load_context
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


@app.get("/firms")
async def firms() -> JSONResponse:
    """The firms this Studio knows, one bindings file each."""

    out = []
    for path in sorted(BINDINGS_DIR.glob("*.yaml")):
        try:
            b = load_bindings(path)
            out.append({"id": path.stem, "label": f"{path.stem} · {b.disclosing_party}"})
        except Exception:
            out.append({"id": path.stem, "label": path.stem})
    return JSONResponse(out)


@app.get("/openbox/state")
async def openbox_state(firm: str = "trial") -> JSONResponse:
    """What is enforced on the firm's agent right now, straight from OpenBox."""

    bindings = load_bindings(BINDINGS_DIR / f"{firm}.yaml")
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
            for kind, path, name_key in (("policy", "policy-rule", "rule_name"),):
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
                            "decision": row.get("decision", ""),
                            "version": row.get("version") or row.get("version_number"),
                            "created_at": row.get("created_at"),
                            "agent_id": agent.id,
                            "clause": f"§{clause}" if clause else "",
                            "agent": agent.name,
                            "opa_loaded": bool(row.get("id") and row["id"] in opa_raw),
                        }
                    )
    # A rule being replaced can briefly list two current versions; show one.
    seen: set[tuple[str, str]] = set()
    items = [
        it
        for it in items
        if not ((it["agent"], it["name"]) in seen or seen.add((it["agent"], it["name"])))
    ]
    return JSONResponse(
        {
            "agent_name": ", ".join(a.name for a in bindings.all_agents),
            "backend": base.replace("http://", "").replace("https://", ""),
            "items": items,
        }
    )


def _event_source():
    """Live activity events per agent, when an org key is configured."""

    base = os.environ.get("OPENBOX_BACKEND_URL", "http://localhost:3000")
    key = os.environ.get("OPENBOX_ORG_API_KEY", "").strip()
    if not key:
        return None

    async def events(agent_id: str) -> list[dict[str, Any]]:
        cached = _EVENTS.get(agent_id)
        if cached and time.monotonic() - cached[0] < 120:
            return cached[1]
        rows = await fetch_activity_events(base, key, agent_id)
        _EVENTS[agent_id] = (time.monotonic(), rows)
        return rows

    return events


# Observed events per agent, kept two minutes: reading the log is paged ten
# at a time, and the officer's page asks for the graph on load and on upload.
_EVENTS: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _graph_json(bindings) -> list[dict[str, Any]]:
    names = {a.id: a.name for a in bindings.all_agents}
    out = []
    for agent_id, graph in bindings.graphs.items():
        body = graph.model_dump(mode="json")
        body["name"] = names.get(agent_id, graph.name)
        body["order"] = graph.ordered_nodes()
        for tool in body["tools"]:
            spec = graph.tool(tool["name"])
            tool["document_arg"] = spec.document_arg if spec else None
        out.append(body)
    return out


def _platform_json(services: Services) -> dict[str, Any]:
    p = services.platform
    return {
        "synced_at": p.synced_at,
        "fields": len(p.fields),
        "operators": p.operators,
        "decisions": p.decisions,
        "span_types": len(p.span_fields),
        "existing_rules": {a: len(r) for a, r in p.existing_rules.items()},
    }


def _report_json(report: CompileReport, bindings, services: Services) -> dict[str, Any]:
    body = report.model_dump(mode="json")
    body["coverage"] = report.coverage
    body["graphs"] = _graph_json(bindings)
    body["platform"] = _platform_json(services)
    for control, raw in zip(report.controls, body["controls"], strict=True):
        raw["description"] = describe(control, bindings)
    return body


# One judge for the process, so tool roles are judged once rather than per upload.
_JUDGE = None


_RULES: dict[str, tuple[float, list[ExistingRule]]] = {}


async def _current_rules(agent_id: str) -> list[ExistingRule]:
    """The agent's current policy rules, live from OpenBox (cached a minute).

    The conflict check compares a new firm's rules with what is already on
    the agent, so it must see what is there now, not a snapshot.
    """

    base = os.environ.get("OPENBOX_BACKEND_URL", "http://localhost:3000").rstrip("/")
    key = os.environ.get("OPENBOX_ORG_API_KEY", "").strip()
    cached = _RULES.get(agent_id)
    if cached and time.monotonic() - cached[0] < 60:
        return cached[1]
    rules: list[ExistingRule] = []
    if key:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(
                    f"{base}/agent/{agent_id}/policy-rule",
                    headers={"X-API-Key": key},
                    params={"limit": 200},
                )
            page = r.json().get("data", {})
            rows = page.get("data", page) if isinstance(page, dict) else page
            rules = [
                ExistingRule(
                    id=x["id"],
                    base_rule_id=x.get("base_rule_id", ""),
                    rule_name=x["rule_name"],
                    decision=x["decision"],
                    priority=x.get("priority", 0),
                    match_mode=x.get("match_mode", "all"),
                    conditions=x.get("conditions", []),
                    is_active=x.get("is_active", False),
                    created_at=x.get("created_at", ""),
                )
                for x in rows or []
                if x.get("is_current_version", True)
            ]
        except Exception:
            rules = []
    _RULES[agent_id] = (time.monotonic(), rules)
    return rules


async def _services(bindings) -> Services:
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = judge_from_env()
    platform = load_context(Path(os.environ.get("PLATFORM_DIR", "platform")))
    for agent in bindings.all_agents:
        live = await _current_rules(agent.id)
        if live:
            platform.existing_rules[agent.id] = live
    # Propose only: a recording backend means nothing is created yet.
    return Services(
        _JUDGE,
        extractor_from_env(),
        RecordingBackend(),
        bindings,
        events=_event_source(),
        platform=platform,
    )


@app.get("/agent-graph")
async def agent_graph(firm: str = "trial") -> JSONResponse:
    """The firm's agent graph with live observations folded in and tool roles judged."""

    bindings_path = BINDINGS_DIR / f"{firm}.yaml"
    if not bindings_path.exists():
        raise HTTPException(404, f"unknown firm {firm}")
    bindings = load_bindings(bindings_path)
    services = await _services(bindings)
    for agent_id, graph in list(bindings.graphs.items()):
        bindings.graphs[agent_id] = await services.map_agent(graph)
    return JSONResponse({"graphs": _graph_json(bindings), "platform": _platform_json(services)})


def _detect_firm(text: str, chosen: str) -> str:
    """The firm whose Disclosing Party the document names, else the chosen one.

    Compiling an NDA under the wrong firm binds it to the wrong folders and
    codenames, so the document's own party names win over the dropdown.
    """

    haystack = " ".join(text.split()).lower()
    hits = []
    for path in sorted(BINDINGS_DIR.glob("*.yaml")):
        try:
            b = load_bindings(path)
        except Exception:
            continue
        names = [b.disclosing_party, *b.disclosing_party_aliases, *b.codenames]
        score = sum(len(n) for n in names if n and n.lower() in haystack)
        if score:
            hits.append((score, path.stem))
    if not hits:
        return chosen
    if any(stem == chosen for _, stem in hits):
        return chosen  # the document does name the chosen firm; the officer's choice stands
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return chosen  # a tie (two firms for one party) is the officer's call
    return hits[0][1]


async def _save_upload(nda: UploadFile) -> Path:
    suffix = Path(nda.filename or "nda.txt").suffix or ".txt"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(await nda.read())
        return Path(handle.name)


@app.get("/agents")
async def agents() -> JSONResponse:
    """The organisation's agents on OpenBox, for the firm form's agent picker."""

    base = os.environ.get("OPENBOX_BACKEND_URL", "http://localhost:3000").rstrip("/")
    key = os.environ.get("OPENBOX_ORG_API_KEY", "").strip()
    if not key:
        return JSONResponse([])
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(
            f"{base}/agent/list", params={"page": 0, "perPage": 100}, headers={"X-API-Key": key}
        )
    page = r.json().get("data", [])
    rows = page.get("data", page) if isinstance(page, dict) else page
    known = {a.id: a.name for p in BINDINGS_DIR.glob("*.yaml") for a in load_bindings(p).all_agents}
    return JSONResponse(
        [
            {
                "id": a["id"],
                "name": a.get("agent_name") or known.get(a["id"]) or a["id"][:8],
                "graph": (Path("graphs") / f"{a['id']}.json").exists(),
            }
            for a in rows or []
        ]
    )


# Contracts uploaded for a firm that does not exist yet, until the form is saved.
_FIRM_DRAFTS: dict[str, Path] = {}


@app.post("/firms/draft")
async def firm_draft(contract: UploadFile = File(...)) -> JSONResponse:
    """Read the firm out of its contract; nothing is written yet."""

    from . import pdf as _pdf

    source = await _save_upload(contract)
    text = _pdf.read_text(source)
    profile = await profile_from_text(text)
    token = uuid.uuid4().hex[:12]
    _FIRM_DRAFTS[token] = source
    existing = {p.stem for p in BINDINGS_DIR.glob("*.yaml")}
    if profile.slug in existing:
        profile.slug = f"{profile.slug}-{token[:4]}"
    return JSONResponse({"token": token, "profile": profile.model_dump(), "clauses": len(_pdf.split_clauses(text))})


class FirmRequest(BaseModel):
    token: str
    slug: str
    disclosing_party: str
    disclosing_party_aliases: list[str] = []
    codenames: list[str] = []
    purpose: str = ""
    covered_folders: list[str]
    agent_id: str
    agent_name: str


@app.post("/firms")
async def create_firm(req: FirmRequest) -> JSONResponse:
    """Write the firm's bindings and compile the contract it was created from."""

    source = _FIRM_DRAFTS.pop(req.token, None)
    if source is None or not source.exists():
        raise HTTPException(404, "contract upload expired; upload it again")
    if not req.covered_folders:
        raise HTTPException(422, "choose the folder the firm's material lives in")
    profile = FirmProfile(
        disclosing_party=req.disclosing_party,
        disclosing_party_aliases=req.disclosing_party_aliases,
        codenames=req.codenames,
        purpose=req.purpose,
        slug=re.sub(r"[^a-z0-9]+", "-", req.slug.lower()).strip("-") or "firm",
    )
    write_bindings(
        BINDINGS_DIR,
        profile,
        agent_id=req.agent_id,
        agent_name=req.agent_name,
        covered_folders=[f.strip().rstrip("/") + "/" for f in req.covered_folders if f.strip()],
    )
    return await _compile(source, profile.slug, profile.slug)


@app.delete("/firms/{slug}")
async def delete_firm(slug: str) -> JSONResponse:
    """Remove a firm's bindings. Rules already on OpenBox are left as they are."""

    path = BINDINGS_DIR / f"{slug}.yaml"
    if not path.exists() or "/" in slug or slug.startswith("."):
        raise HTTPException(404, f"unknown firm {slug}")
    path.unlink()
    return JSONResponse({"deleted": slug})


@app.post("/propose")
async def propose(nda: UploadFile = File(...), firm: str = Form(...)) -> JSONResponse:
    if not (BINDINGS_DIR / f"{firm}.yaml").exists():
        raise HTTPException(404, f"unknown firm {firm}")
    source = await _save_upload(nda)
    from . import pdf as _pdf

    return await _compile(source, _detect_firm(_pdf.read_text(source), firm), firm)


async def _compile(source: Path, detected: str, firm: str) -> JSONResponse:
    bindings = load_bindings(BINDINGS_DIR / f"{detected}.yaml")
    services = await _services(bindings)
    try:
        # Propose is read-only, so the graph runs bare here. Governing the compiler
        # itself (the CLI path) re-instruments the process per handler, which does
        # not suit a long-lived server.
        report = await compile_nda(source, services, build_graph(services))
    finally:
        source.unlink(missing_ok=True)
    # Reset the recording backend's pretend statuses; the officer decides.
    # Rows the build or verify step put in review keep that mark and its note.
    report.controls = [
        c.model_copy(update={"status": "draft" if c.status != "review" else "review", "remote_id": None})
        for c in report.controls
    ]
    draft_id = uuid.uuid4().hex[:12]
    _DRAFTS[draft_id] = (report, detected)
    body = _report_json(report, bindings, services)
    body["draft_id"] = draft_id
    body["firm"] = detected
    body["firm_switched"] = detected != firm
    return JSONResponse(body)


@app.post("/apply")
async def apply(req: ApplyRequest) -> JSONResponse:
    entry = _DRAFTS.get(req.draft_id)
    if entry is None:
        raise HTTPException(404, "draft expired; read the NDA again")
    report, _firm = entry
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
