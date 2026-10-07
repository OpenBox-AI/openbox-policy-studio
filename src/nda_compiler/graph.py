"""The compiler as a LangGraph graph.

    parse → classify → extract → build → verify → apply

Each node fans out internally (asyncio.gather) over clauses or controls, so
wall-clock is one round trip per stage, not one per clause. Every node records
its own timing; the report shows where the seconds went.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import pdf
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
from .templates import build_controls, summarize


class CompileState(TypedDict, total=False):
    source: str
    nda_text: str
    clauses: list[Clause]
    definitions: str
    classifications: list[Classification]
    obligations: list[Obligation]
    controls: list[Control]
    review: list[str]
    timings: list[StageTiming]


class Services:
    def __init__(
        self,
        judge: Judge,
        extractor: Extractor,
        backend: OpenBoxBackend,
        bindings: Bindings,
        verify_threshold: float | None = None,
    ) -> None:
        self.judge = judge
        self.extractor = extractor
        self.backend = backend
        self.bindings = bindings
        self.verify_threshold = (
            verify_threshold
            if verify_threshold is not None
            else float(os.environ.get("VERIFY_THRESHOLD", "0.8"))
        )


def _timed(stage: str, state: CompileState, started: float, update: dict[str, Any]) -> dict:
    timings = [
        *state.get("timings", []),
        StageTiming(stage=stage, ms=(time.perf_counter() - started) * 1000),
    ]
    return {**update, "timings": timings}


def build_graph(services: Services):
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
        return _timed("extract", state, started, {"obligations": obligations, "review": review})

    async def build(state: CompileState) -> dict:
        started = time.perf_counter()
        controls: list[Control] = []
        for obligation in state["obligations"]:
            if obligation.ungrounded:
                continue
            controls.extend(build_controls(obligation, services.bindings))
        return _timed("build", state, started, {"controls": controls})

    async def verify(state: CompileState) -> dict:
        started = time.perf_counter()
        by_id = {c.id: c for c in state["clauses"]}
        controls = state["controls"]
        probabilities = await verify_all(
            services.judge,
            [
                (by_id[c.clause_id], c.payload, summarize(c, services.bindings.all_agents))
                for c in controls
            ],
        )
        verified: list[Control] = []
        review = list(state.get("review", []))
        for control, probability in zip(controls, probabilities, strict=True):
            if probability >= services.verify_threshold:
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
        ("parse", parse),
        ("classify", classify),
        ("extract", extract),
        ("build", build),
        ("verify", verify),
        ("apply", apply),
    ):
        builder.add_node(name, node)
    builder.add_edge(START, "parse")
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
        timings=state.get("timings", []),
        models={"judge": services.judge.model, "extractor": services.extractor.model},
    )
