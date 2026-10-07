"""TypeSafe System One (JEV) for the two decisions the compiler makes.

classify: one call, the whole NDA as state, one choice question per clause.
verify:   per generated control, one noul: does this rule faithfully enforce
          the clause it came from?

JEV answers with calibrated probabilities, not text, so both results are
numbers the pipeline can threshold. It reads criteria literally, so every
option spells out what qualifies, what does not, and an example.

With no TYPESAFE_API_KEY a deterministic keyword classifier stands in, so the
pipeline runs offline and tests stay hermetic.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any, Protocol

from .models import Classification, Clause, ControlKind

KIND_CRITERIA: dict[str, dict[str, Any]] = {
    ControlKind.PERMITTED_RECIPIENTS.value: {
        "what": "Limits who inside the receiving organisation may see the information "
        "(need to know, named representatives, specific teams), or forbids sharing it with "
        "named outsiders such as competitors.",
        "not_for": "Clauses about sending data to external systems, which are "
        "third_party_disclosure.",
        "example": "shall disclose only to its Representatives who have a need to know",
    },
    ControlKind.THIRD_PARTY_DISCLOSURE.value: {
        "what": "Forbids transmitting, emailing, uploading or otherwise sending the information "
        "to third parties or outside the receiving party's systems.",
        "not_for": "Clauses that only limit which internal people may see it.",
        "example": "shall not transmit or email Confidential Information to any third party",
    },
    ControlKind.USE_RESTRICTION.value: {
        "what": "Limits the purposes for which the information may be used.",
        "not_for": "Clauses about who may see it or where it may be sent.",
        "example": "shall use Confidential Information solely for the Purpose",
    },
    ControlKind.MARKED_MATERIAL.value: {
        "what": "Defines markings, codenames or labels that identify confidential material, "
        "or forbids reproducing marked material.",
        "example": "anything marked Confidential or Project Fizz",
    },
    ControlKind.PERSONAL_DATA.value: {
        "what": "Concerns personal data of individuals (employees, customers) and limits "
        "its reproduction or processing.",
        "example": "shall not reproduce such personal data in any report",
    },
    ControlKind.NOT_ENFORCEABLE.value: {
        "what": "A real obligation that cannot be checked on an agent's tool calls: term, "
        "survival, return or destruction, notice, remedies, governing law, assignment.",
        "example": "This Agreement remains in force for three years",
    },
    ControlKind.DEFINITION.value: {
        "what": "Defines terms used elsewhere and imposes no obligation itself.",
        "example": "'Confidential Information' means all non-public information",
    },
    ControlKind.BOILERPLATE.value: {
        "what": "Recitals, signatures, headings, counterparts, or an exceptions clause that "
        "only narrows other obligations.",
        "example": "The obligations in Sections 2 to 5 do not apply to information that is public",
    },
}

_FAKE_RULES: list[tuple[ControlKind, re.Pattern[str]]] = [
    (ControlKind.DEFINITION, re.compile(r"\bmeans\b.*\bmeans\b", re.I | re.S)),
    (ControlKind.PERSONAL_DATA, re.compile(r"personal data", re.I)),
    (ControlKind.THIRD_PARTY_DISCLOSURE, re.compile(r"transmit|email|send|third part", re.I)),
    (
        ControlKind.PERMITTED_RECIPIENTS,
        re.compile(r"need to know|Representatives|competitor", re.I),
    ),
    (ControlKind.USE_RESTRICTION, re.compile(r"solely for|only for the Purpose", re.I)),
    (ControlKind.MARKED_MATERIAL, re.compile(r"\bmarked\b", re.I)),
    (ControlKind.BOILERPLATE, re.compile(r"do not apply|exception", re.I)),
    (
        ControlKind.NOT_ENFORCEABLE,
        re.compile(r"\bterm\b|years|return|destroy|governing law|governed by|survive", re.I),
    ),
]


class Judge(Protocol):
    model: str

    async def classify(self, nda_text: str, clauses: list[Clause]) -> list[Classification]: ...

    async def verify(
        self, clause: Clause, control_payload: dict[str, Any], summary: str
    ) -> float: ...


class FakeJudge:
    """Keyword stand-in used when no TypeSafe key is set."""

    model = "fake-judge"

    async def classify(self, nda_text: str, clauses: list[Clause]) -> list[Classification]:
        out = []
        for clause in clauses:
            kind = ControlKind.BOILERPLATE
            for candidate, pattern in _FAKE_RULES:
                if pattern.search(f"{clause.heading}. {clause.text}"):
                    kind = candidate
                    break
            out.append(Classification(clause_id=clause.id, kind=kind, confidence=0.99))
        return out

    async def verify(self, clause: Clause, control_payload: dict[str, Any], summary: str) -> float:
        return 0.95


class TypeSafeJudge:
    def __init__(self, api_key: str, model: str = "jev-latest") -> None:
        from typesafe_sdk import AsyncTypeSafeClient

        self._client = AsyncTypeSafeClient(api_key=api_key)
        self.model = model

    async def classify(self, nda_text: str, clauses: list[Clause]) -> list[Classification]:
        from typesafe_sdk import Choice

        questions = {
            f"clause_{clause.id.replace('.', '_')}": Choice(
                instructions={
                    "what": f"Which kind of control does clause {clause.id} ask for?",
                    "clause": clause.text,
                },
                criteria=KIND_CRITERIA,
            )
            for clause in clauses
        }
        result = await self._client.system_one(
            {"document": nda_text, "task": "classify each numbered clause"},
            questions,
            model=self.model,
        )
        self.model = getattr(result, "model", self.model)
        out = []
        for clause in clauses:
            answer = result.choices[f"clause_{clause.id.replace('.', '_')}"]
            out.append(
                Classification(
                    clause_id=clause.id,
                    kind=ControlKind(answer.choice),
                    confidence=float(answer.confidence),
                    probabilities={k: float(v) for k, v in answer.probabilities.items()},
                )
            )
        return out

    async def verify(self, clause: Clause, control_payload: dict[str, Any], summary: str) -> float:
        from typesafe_sdk import Noul

        result = await self._client.system_one(
            {
                "nda_clause": clause.text,
                "proposed_control_in_plain_english": summary,
                "note": "The control is one of several derived from this clause and is enforced "
                "automatically on an AI agent's tool calls. It is not expected to cover the "
                "whole clause by itself.",
            },
            {
                "wrong": Noul(
                    instructions={
                        "what": "Would enforcing this control be a mistake under this clause?",
                    },
                    criteria={
                        "true": {
                            "what": "The control forbids something the clause permits, targets "
                            "material or a party the clause does not cover, or restricts an "
                            "action unrelated to what the clause restricts.",
                            "example": "Clause forbids emailing documents to third parties; "
                            "control blocks an agent from reading a public website.",
                        },
                        "false": {
                            "what": "The control restricts behaviour of the kind the clause "
                            "restricts, for the material and parties the clause covers, even "
                            "if it only addresses part of the clause.",
                            "example": "Clause limits disclosure to Representatives; control "
                            "blocks one non-Representative agent from reading the covered folder.",
                        },
                    },
                )
            },
            model=self.model,
        )
        # Reported as P(faithful) so the threshold reads the same way everywhere.
        return 1.0 - float(result.nouls["wrong"].noul)


def judge_from_env() -> Judge:
    key = os.environ.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        return FakeJudge()
    return TypeSafeJudge(
        key, os.environ.get("TYPESAFE_MODEL", "jev-latest").strip() or "jev-latest"
    )


async def verify_all(judge: Judge, items: list[tuple[Clause, dict[str, Any], str]]) -> list[float]:
    return list(await asyncio.gather(*(judge.verify(c, p, s) for c, p, s in items)))
