"""Shapes shared by every stage of the compiler.

A clause is read once, classified once, extracted once, and becomes zero or
more OpenBox controls. Every control keeps the clause id it came from so the
report can show the compliance team which sentence produced which rule.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ControlKind(StrEnum):
    """What kind of control a clause asks for. Chosen by JEV."""

    PERMITTED_RECIPIENTS = "permitted_recipients"
    THIRD_PARTY_DISCLOSURE = "third_party_disclosure"
    USE_RESTRICTION = "use_restriction"
    MARKED_MATERIAL = "marked_material"
    PERSONAL_DATA = "personal_data"
    NOT_ENFORCEABLE = "not_enforceable"
    DEFINITION = "definition"
    BOILERPLATE = "boilerplate"


ENFORCEABLE_KINDS = frozenset(
    {
        ControlKind.PERMITTED_RECIPIENTS,
        ControlKind.THIRD_PARTY_DISCLOSURE,
        ControlKind.USE_RESTRICTION,
        ControlKind.MARKED_MATERIAL,
        ControlKind.PERSONAL_DATA,
    }
)


class Clause(BaseModel):
    id: str
    heading: str = ""
    text: str
    references: list[str] = Field(default_factory=list)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:10]


class Classification(BaseModel):
    clause_id: str
    kind: ControlKind
    confidence: float
    probabilities: dict[str, float] = Field(default_factory=dict)


class Obligation(BaseModel):
    """One enforceable duty, with every literal grounded in the NDA text."""

    clause_id: str
    kind: ControlKind
    bound_party: str
    action: str
    subject: str
    permitted_recipients: list[str] = Field(default_factory=list)
    prohibited_recipients: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    marked_terms: list[str] = Field(default_factory=list)
    purpose: str | None = None
    duration: str | None = None
    source_quote: str
    ungrounded: list[str] = Field(default_factory=list)


ControlType = Literal["policy_rule", "behavior_rule", "guardrail", "judgement"]


class TestCase(BaseModel):
    label: str
    input: dict[str, Any]
    expect: str


class Control(BaseModel):
    clause_id: str
    kind: ControlKind
    type: ControlType
    agent_id: str
    payload: dict[str, Any]
    tests: list[TestCase] = Field(default_factory=list)
    verify_probability: float | None = None
    status: Literal["draft", "verified", "review", "created", "evaluated", "active", "failed"] = (
        "draft"
    )
    remote_id: str | None = None
    note: str = ""


class StageTiming(BaseModel):
    stage: str
    ms: float


class CompileReport(BaseModel):
    matter: str
    source: str
    clauses: list[Clause]
    classifications: list[Classification]
    obligations: list[Obligation]
    controls: list[Control]
    review: list[str] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    models: dict[str, str] = Field(default_factory=dict)

    @property
    def coverage(self) -> dict[str, int]:
        by_kind: dict[str, int] = {}
        for c in self.classifications:
            by_kind[c.kind.value] = by_kind.get(c.kind.value, 0) + 1
        return {
            "clauses": len(self.clauses),
            "enforced": sum(1 for c in self.controls if c.type != "judgement"),
            "judgement": sum(1 for c in self.controls if c.type == "judgement"),
            "not_enforceable": by_kind.get(ControlKind.NOT_ENFORCEABLE.value, 0),
            "review": len(self.review),
        }

    @property
    def total_ms(self) -> float:
        return sum(t.ms for t in self.timings)
