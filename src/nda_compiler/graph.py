"""The compiler as a LangGraph graph.

    map → parse → classify → extract → build → verify → apply

`map` reads the governed agent's own graph (its tools, their arguments, what
each does) and folds in what OpenBox has already observed the agent calling,
so every later template binds to real tool calls. The platform context (what
a policy rule can express, which rules already exist) is handed to the judge
once and rides along with every question. Each node fans out internally
(asyncio.gather) over clauses or controls, so wall-clock is one round trip
per stage, not one per clause. Every node records its own timing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import pdf
from .agent_graph import AgentGraph, observe, with_roles
from .bindings import Bindings
from .extract import Extractor, extract_all
from .jev import Judge, verify_all
from .models import (
    ENFORCEABLE_KINDS,
    Classification,
    Clause,
    CompileReport,
    Control,
    Obligation,
    StageTiming,
)
from .openbox_api import OpenBoxBackend, apply_all
from .platform_context import PlatformContext, load_context
from .templates import build_controls, describe, not_applicable, summarize

EventSource = Callable[[str], Awaitable[list[dict[str, Any]]]]


class CompileState(TypedDict, total=False):
    source: str
    nda_text: str
    clauses: list[Clause]
    definitions: str
    classifications: list[Classification]
    obligations: list[Obligation]
    controls: list[Control]
    review: list[str]
    not_applicable: list[str]
    timings: list[StageTiming]


class Services:
    def __init__(
        self,
        judge: Judge,
        extractor: Extractor,
        backend: OpenBoxBackend,
        bindings: Bindings,
        verify_threshold: float | None = None,
        events: EventSource | None = None,
        platform: PlatformContext | None = None,
    ) -> None:
        self.judge = judge
        self.extractor = extractor
        self.backend = backend
        self.bindings = bindings
        self.events = events
        self.platform = platform if platform is not None else load_context(
            Path(os.environ.get("PLATFORM_DIR", "platform"))
        )
        self.verify_threshold = (
            verify_threshold
            if verify_threshold is not None
            else float(os.environ.get("VERIFY_THRESHOLD", "0.8"))
        )
        # The judge sees the platform the same way on every question.
        self.judge.platform = self.platform.summary()
        self.judge.existing = {
            a.id: self.platform.existing_summary(a.id) for a in bindings.all_agents
        }
        # Tool roles depend on the tool, not the NDA; judged once per process.
        self._roles: dict[str, tuple[str, float]] = {}

    async def map_agent(self, graph: AgentGraph) -> AgentGraph:
        """One agent's graph with live observations folded in and roles judged."""

        if self.events is not None:
            try:
                graph = observe(await self.events(graph.agent_id), graph)
            except Exception:
                pass  # the static graph still stands; observation is a bonus
        missing = [t for t in graph.tools if t.name not in self._roles]
        if missing:
            self._roles.update(await self.judge.classify_tools(missing))
        return with_roles(graph, self._roles)


def _timed(stage: str, state: CompileState, started: float, update: dict[str, Any]) -> dict:
    timings = [
        *state.get("timings", []),
        StageTiming(stage=stage, ms=(time.perf_counter() - started) * 1000),
    ]
    return {**update, "timings": timings}


def build_graph(services: Services):
    async def map_tools(state: CompileState) -> dict:
        started = time.perf_counter()
        graphs = services.bindings.graphs
        mapped = await asyncio.gather(*(services.map_agent(g) for g in graphs.values()))
        for graph in mapped:
            graphs[graph.agent_id] = graph
        return _timed("map", state, started, {})

    async def parse(state: CompileState) -> dict:
        started = time.perf_counter()
        text, clauses = pdf.load(Path(state["source"]))
        return _timed(
            "parse",
            state,
            started,
            {"nda_text": text, "clauses": clauses, "definitions": pdf.definitions_text(clauses)},
        )

    async def classify(state: CompileState) -> dict:
        started = time.perf_counter()
        classifications = await services.judge.classify(state["nda_text"], state["clauses"])
        return _timed("classify", state, started, {"classifications": classifications})

    async def extract(state: CompileState) -> dict:
        started = time.perf_counter()
        by_id = {c.id: c for c in state["clauses"]}
        items = [
            (by_id[c.clause_id], c.kind)
            for c in state["classifications"]
            if c.kind in ENFORCEABLE_KINDS
        ]
        obligations = await extract_all(
            services.extractor, items, state["definitions"], state["nda_text"]
        )
        review = [
            f"§{o.clause_id}: ungrounded values {o.ungrounded}" for o in obligations if o.ungrounded
        ]
        # Which platform decision each clause calls for, from its wording.
        grounded = [o for o in obligations if not o.ungrounded]
        decisions = await services.judge.decide(grounded, services.platform.decisions)
        obligations = [
            o.model_copy(update={"decision": decisions.get(o.clause_id, "BLOCK")})
            for o in obligations
        ]
        return _timed("extract", state, started, {"obligations": obligations, "review": review})

    async def build(state: CompileState) -> dict:
        started = time.perf_counter()
        controls: list[Control] = []
        skipped: list[str] = []
        # Sub-clauses often restate one duty (§2.1 and §2.2 both limiting access);
        # an identical control is proposed once and credits every clause.
        seen: dict[str, Control] = {}
        for obligation in state["obligations"]:
            if obligation.ungrounded:
                continue
            for note in not_applicable(obligation, services.bindings):
                if note not in skipped:
                    skipped.append(note)
            for control in build_controls(obligation, services.bindings):
                body = {
                    k: v
                    for k, v in control.payload.items()
                    if k not in ("rule_name", "description", "reason", "reject_message")
                }
                key = json.dumps([control.type, control.agent_id, body], sort_keys=True)
                if key in seen:
                    seen[key].note = (
                        seen[key].note + ", " if seen[key].note else "also "
                    ) + f"§{control.clause_id}"
                    continue
                seen[key] = control
                controls.append(control)
        # Every field a rule uses must be one OPA actually sees.
        for control in controls:
            unknown = [
                c["left"]["field"]
                for c in control.payload["conditions"]
                if services.platform.fields and services.platform.field(c["left"]["field"]) is None
            ]
            if unknown:
                control.status = "review"
                control.note = f"fields not in the platform catalog: {', '.join(unknown)}"
        return _timed("build", state, started, {"controls": controls, "not_applicable": skipped})

    async def verify(state: CompileState) -> dict:
        started = time.perf_counter()
        by_id = {c.id: c for c in state["clauses"]}
        controls = state["controls"]
        probabilities = await verify_all(
            services.judge,
            [
                (by_id[c.clause_id], {**c.payload, "agent_id": c.agent_id}, describe(c, services.bindings))
                for c in controls
            ],
        )
        verified: list[Control] = []
        review = list(state.get("review", []))
        for control, probability in zip(controls, probabilities, strict=True):
            if control.status == "review":
                verified.append(control.model_copy(update={"verify_probability": probability}))
                review.append(f"§{control.clause_id}: {control.note}")
            elif probability >= services.verify_threshold:
                verified.append(
                    control.model_copy(
                        update={"verify_probability": probability, "status": "verified"}
                    )
                )
            else:
                verified.append(
                    control.model_copy(
                        update={"verify_probability": probability, "status": "review"}
                    )
                )
                review.append(
                    f"§{control.clause_id}: judge gave {probability:.2f} for "
                    f"{summarize(control, services.bindings.all_agents)}"
                )
        return _timed("verify", state, started, {"controls": verified, "review": review})

    async def apply(state: CompileState) -> dict:
        started = time.perf_counter()
        ready = [c for c in state["controls"] if c.status == "verified"]
        held = [c for c in state["controls"] if c.status != "verified"]
        applied = await apply_all(services.backend, ready)
        return _timed("apply", state, started, {"controls": [*applied, *held]})

    builder = StateGraph(CompileState)
    for name, node in (
        ("map", map_tools),
        ("parse", parse),
        ("classify", classify),
        ("extract", extract),
        ("build", build),
        ("verify", verify),
        ("apply", apply),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "map")
    builder.add_edge("map", "parse")
    builder.add_edge("parse", "classify")
    builder.add_edge("classify", "extract")
    builder.add_edge("extract", "build")
    builder.add_edge("build", "verify")
    builder.add_edge("verify", "apply")
    builder.add_edge("apply", END)
    return builder.compile()


async def compile_nda(source: Path, services: Services, graph=None) -> CompileReport:
    graph = graph or build_graph(services)
    state = await graph.ainvoke({"source": str(source)})
    return CompileReport(
        matter=services.bindings.matter,
        source=str(source),
        clauses=state["clauses"],
        classifications=state["classifications"],
        obligations=state["obligations"],
        controls=state["controls"],
        review=state.get("review", []),
        not_applicable=state.get("not_applicable", []),
        timings=state.get("timings", []),
        models={"judge": services.judge.model, "extractor": services.extractor.model},
    )
