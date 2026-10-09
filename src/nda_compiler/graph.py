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

from . import conflicts, pdf
from .agent_graph import AgentGraph, observe, with_roles
from .bindings import Bindings
from .extract import Extractor, extract_all, in_clause, outsider_names
from .jev import RECIPIENT_CRITERIA, Judge, clause_nouls, decision_from_wording, verify_all
from .models import (
    ENFORCEABLE_KINDS,
    Classification,
    Clause,
    CompileReport,
    Control,
    ControlKind,
    Obligation,
    StageTiming,
)
from .openbox_api import OpenBoxBackend, apply_all
from .platform_context import PlatformContext, load_context
from .templates import TEMPLATES, build_controls, describe, not_applicable, summarize

EventSource = Callable[[str], Awaitable[list[dict[str, Any]]]]

# Least to most restrictive, the platform's decisions.
STRICTNESS = ["ALLOW", "CONSTRAIN", "REQUIRE_APPROVAL", "BLOCK", "HALT"]


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
        # Recipient check (RECIPIENT_CHECK, on by default): a clause the
        # classifier filed under another kind may still limit who may receive
        # the material ("use it only for the deal and disclose it only to
        # Representatives"). The judge answers that as its own yes/no; the
        # nominated clauses are read again as recipient limits, and the
        # extractor's "applies" answer drops the ones that are not.
        if os.environ.get("RECIPIENT_CHECK", "1").strip() != "0":
            threshold = float(os.environ.get("RECIPIENT_CHECK_THRESHOLD", "0.8"))
            others = [
                by_id[c.clause_id]
                for c in state["classifications"]
                if c.kind not in (ControlKind.PERMITTED_RECIPIENTS, ControlKind.DEFINITION)
            ]
            probs = await clause_nouls(
                services.judge,
                state["nda_text"],
                others,
                "Does clause {id} limit who may receive the confidential information?",
                RECIPIENT_CRITERIA,
            )
            items += [
                (by_id[cid], ControlKind.PERMITTED_RECIPIENTS) for cid, p in probs.items() if p >= threshold
            ]
        obligations = await extract_all(
            services.extractor, items, state["definitions"], state["nda_text"]
        )
        # Recipients must come from the clause itself, not from elsewhere in
        # the NDA: a competitor clause that borrows "Representatives" from §2
        # would otherwise also produce §2's folder rules.
        obligations = [
            o.model_copy(
                update={
                    "permitted_recipients": in_clause(o.permitted_recipients, by_id[o.clause_id]),
                    "prohibited_recipients": in_clause(o.prohibited_recipients, by_id[o.clause_id]),
                }
            )
            for o in obligations
        ]
        # A paraphrase only matters where it would have shaped a rule; clauses
        # with no template are reported as not applicable regardless.
        review = [
            f"§{o.clause_id}: ungrounded values {o.ungrounded}"
            for o in obligations
            if o.ungrounded and o.kind in TEMPLATES
        ]
        # Which platform decision each clause calls for comes from its literal
        # wording: "prior written consent" -> REQUIRE_APPROVAL, "material
        # breach" / "injunctive relief" -> HALT, otherwise the prohibition is
        # a BLOCK. Deterministic on purpose: the same clause must always give
        # the same rule, and a model asked to choose between BLOCK and HALT on
        # a clause that says neither answered differently run to run.
        # With the v2 prompt the model answers two plain questions about the
        # clause (does consent unlock it, is its breach material); the keyword
        # check stays as the fallback when it was not asked.
        def decision(o: Obligation) -> str:
            if o.breach_is_material:
                return "HALT"
            if o.consent_unlocks is not None:
                return "REQUIRE_APPROVAL" if o.consent_unlocks else (
                    "HALT" if decision_from_wording(by_id[o.clause_id].text) == "HALT" else "BLOCK"
                )
            return decision_from_wording(by_id[o.clause_id].text) or "BLOCK"

        # A clause the model reads as imposing no duty of the classified kind
        # (a standstill or no-contact clause classified as an access limit)
        # yields no rule.
        dropped = [o for o in obligations if o.applies is False]
        obligations = [
            o.model_copy(update={"decision": decision(o)}) for o in obligations if o.applies is not False
        ]
        # Named organisations are never permitted recipients of the counterparty's
        # material: a clause that names bidders or competitors forbids them, in
        # whichever list the model put them.
        obligations = [
            o.model_copy(
                update={
                    "permitted_recipients": [p for p in o.permitted_recipients if p not in named],
                    "prohibited_recipients": [*o.prohibited_recipients, *(n for n in named if n not in o.prohibited_recipients)],
                }
            )
            if o.kind == ControlKind.PERMITTED_RECIPIENTS
            and (named := outsider_names(o.permitted_recipients, state["nda_text"]))
            else o
            for o in obligations
        ]
        # "Any transmission in breach of clause 4.1 shall be a material breach":
        # a sub-clause that only states the consequence of breaching another
        # raises that clause's decision, even though it is not a duty itself.
        enforced = {o.clause_id for o in obligations}
        raised: dict[str, str] = {}
        for c in state["classifications"]:
            clause = by_id[c.clause_id]
            if c.kind in ENFORCEABLE_KINDS or "breach" not in clause.text.lower():
                continue
            if decision_from_wording(clause.text) != "HALT":
                continue
            for ref in clause.references:
                if ref in enforced:
                    raised[ref] = "HALT"
        obligations = [
            o.model_copy(update={"decision": "HALT"})
            if o.clause_id in raised and STRICTNESS.index("HALT") > STRICTNESS.index(o.decision)
            else o
            for o in obligations
        ]
        review = [*review, *(f"§{o.clause_id}: read as imposing no {o.kind.value} duty; no rule proposed" for o in dropped)]
        return _timed("extract", state, started, {"obligations": obligations, "review": review})

    async def build(state: CompileState) -> dict:
        started = time.perf_counter()
        controls: list[Control] = []
        skipped: list[str] = []
        # Several clauses often bite on the same call (§2 "only Representatives",
        # §4 "sub-advisers only with consent"). The platform takes the first
        # match by priority, so two rules with the same conditions and different
        # decisions would be settled by accident. One rule is kept per set of
        # conditions: the strictest decision, crediting every clause behind it.
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
                    if k not in ("rule_name", "description", "reason", "decision", "constraints", "trust_impact")
                }
                key = json.dumps([control.agent_id, body], sort_keys=True)
                if key in seen:
                    kept = seen[key]
                    if STRICTNESS.index(control.payload["decision"]) > STRICTNESS.index(kept.payload["decision"]):
                        control.note = (f"also §{kept.clause_id}" if not kept.note else kept.note + f", §{kept.clause_id}")
                        controls[controls.index(kept)] = control
                        seen[key] = control
                    else:
                        kept.note = (kept.note + ", " if kept.note else "also ") + f"§{control.clause_id}"
                    continue
                seen[key] = control
                controls.append(control)
        # Other firms' rules already on the agent: overlapping folders, opposite
        # decisions on the same conditions, generic markings.
        by_agent: dict[str, list[Control]] = {}
        for control in controls:
            by_agent.setdefault(control.agent_id, []).append(control)
        controls = [
            c
            for agent_id, group in by_agent.items()
            for c in conflicts.check(group, services.platform.rules_for(agent_id), services.bindings.firm)
        ]
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
                continue
            elif probability >= services.verify_threshold:
                verified.append(
                    control.model_copy(
                        update={"verify_probability": probability, "status": "verified"}
                    )
                )
            else:
                # Below the threshold the rule is still proposed; it is simply
                # not auto-applied by the CLI, and the score is recorded.
                verified.append(control.model_copy(update={"verify_probability": probability}))
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
        firm=services.bindings.firm,
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
