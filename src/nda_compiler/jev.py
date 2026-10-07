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

from .agent_graph import ROLE_CRITERIA, ToolSpec, fake_roles
from .models import Classification, Clause, ControlKind, Obligation

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

    async def classify_tools(self, tools: list[ToolSpec]) -> dict[str, tuple[str, float]]: ...

    async def decide(self, obligations: list[Obligation], decisions: list[str]) -> dict[str, str]: ...

    platform: dict[str, Any]
    existing: dict[str, list[dict[str, Any]]]


DECISION_CRITERIA: dict[str, dict[str, str]] = {
    "BLOCK": {
        "what": "The clause forbids the action outright: shall not, may not, in no event, "
        "is prohibited.",
        "example": "shall not disclose Confidential Information to any third party",
    },
    "REQUIRE_APPROVAL": {
        "what": "The clause allows the action only with the other party's consent, approval "
        "or written permission, or on notice to them.",
        "example": "may disclose to advisers only with the prior written consent of the "
        "Disclosing Party",
    },
    "HALT": {
        "what": "The clause treats the action as so serious that any attempt should stop "
        "the agent entirely: material breach, immediate termination, injunctive relief "
        "named for this act.",
        "example": "any disclosure to a competitor constitutes a material breach entitling "
        "the Disclosing Party to immediate injunctive relief",
    },
}

_FAKE_DECISION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("REQUIRE_APPROVAL", re.compile(r"prior written consent|with the consent|approval of|permission", re.I)),
    ("HALT", re.compile(r"material breach|injunctive|immediate(ly)? terminat", re.I)),
]


def fake_decisions(obligations: list[Obligation]) -> dict[str, str]:
    out = {}
    for o in obligations:
        out[o.clause_id] = next(
            (d for d, p in _FAKE_DECISION_RULES if p.search(o.source_quote)), "BLOCK"
        )
    return out


class FakeJudge:
    """Keyword stand-in used when no TypeSafe key is set."""

    model = "fake-judge"

    def __init__(self) -> None:
        self.platform: dict[str, Any] = {}
        self.existing: dict[str, list[dict[str, Any]]] = {}

    async def classify_tools(self, tools: list[ToolSpec]) -> dict[str, tuple[str, float]]:
        return fake_roles(tools)

    async def decide(self, obligations: list[Obligation], decisions: list[str]) -> dict[str, str]:
        return fake_decisions(obligations)

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
        # What a policy rule can express and which rules already exist, from
        # platform_context; set by Services so every question carries it.
        self.platform: dict[str, Any] = {}
        self.existing: dict[str, list[dict[str, Any]]] = {}

    async def decide(self, obligations: list[Obligation], decisions: list[str]) -> dict[str, str]:
        """Which platform decision each clause calls for, from its own wording."""

        from typesafe_sdk import Choice

        if not obligations:
            return {}
        criteria = {d: DECISION_CRITERIA[d] for d in decisions if d in DECISION_CRITERIA}
        questions = {
            f"clause_{i}": Choice(
                instructions={
                    "what": f"Which decision should OpenBox apply when the agent attempts "
                    f"what clause {o.clause_id} restricts?",
                    "clause": o.source_quote,
                },
                criteria=criteria,
            )
            for i, o in enumerate(obligations)
        }
        result = await self._client.system_one(
            {"platform_decisions": self.platform.get("decisions", {})},
            questions,
            model=self.model,
        )
        return {
            o.clause_id: result.choices[f"clause_{i}"].choice for i, o in enumerate(obligations)
        }

    async def classify_tools(self, tools: list[ToolSpec]) -> dict[str, tuple[str, float]]:
        """One choice per tool: what does calling it do with the material?

        Judged from the tool's name, description and argument names, which is
        all an operator reviewing the agent's graph would have too.
        """

        from typesafe_sdk import Choice

        if not tools:
            return {}
        questions = {
            f"tool_{i}": Choice(
                instructions={
                    "what": f"What does the tool '{tool.name}' do with the material it handles?",
                    "tool": f"{tool.name}({', '.join(tool.args)}): {tool.description}",
                },
                criteria=ROLE_CRITERIA,
            )
            for i, tool in enumerate(tools)
        }
        result = await self._client.system_one(
            {"agent_tools": [f"{t.name}: {t.description}" for t in tools]},
            questions,
            model=self.model,
        )
        return {
            tool.name: (
                result.choices[f"tool_{i}"].choice,
                float(result.choices[f"tool_{i}"].confidence),
            )
            for i, tool in enumerate(tools)
        }

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
            {
                "document": nda_text,
                "task": "classify each numbered clause",
                "what_a_policy_rule_can_check": self.platform,
            },
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
                "what_a_policy_rule_can_check": self.platform,
                "rules_already_on_this_agent": self.existing.get(
                    str(control_payload.get("agent_id", "")), []
                ),
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
