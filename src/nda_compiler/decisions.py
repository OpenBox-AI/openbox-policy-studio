"""OpenAI Decisions API (gpt-6-luna) for the decisions the compiler makes.

A drop-in for the TypeSafe judge: the same questions with the same criteria,
asked as Decisions API question types.

classify        one `choice` per clause, the whole NDA as shared input
classify_tools  one `choice` per tool
verify          one `predicate` per control: would enforcing it be a mistake?
predicates      one `predicate` per clause (the recipient check)

POST /v1/decisions takes `model`, `input` (shared evidence) and `questions`;
each answer comes back by `name`, as a probability (predicate) or a choice
with per-value probabilities (choice), or as a refusal.

The default judge whenever OPENAI_API_KEY is set (see jev.judge_from_env).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx

from .agent_graph import ROLE_CRITERIA, ToolSpec
from .models import Classification, Clause, ControlKind

URL = os.environ.get("OPENAI_DECISIONS_URL", "https://api.openai.com/v1/decisions")
# Questions per request; the API does not publish a limit, so long NDAs are split.
BATCH = int(os.environ.get("OPENAI_DECISIONS_BATCH", "40"))


def _describe(criteria: dict[str, Any]) -> str:
    parts = [criteria.get("what", "")]
    if criteria.get("not_for"):
        parts.append(f"Not for: {criteria['not_for']}")
    if criteria.get("example"):
        parts.append(f"Example: {criteria['example']}")
    return " ".join(p for p in parts if p)


def _choices(criteria: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [{"value": value, "description": _describe(c)} for value, c in criteria.items()]


class Refused(RuntimeError):
    pass


class DecisionsJudge:
    def __init__(self, api_key: str, model: str = "gpt-6-luna") -> None:
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=httpx.Timeout(120.0)
        )
        self.model = model
        self.platform: dict[str, Any] = {}
        self.existing: dict[str, list[dict[str, Any]]] = {}

    async def _ask(self, evidence: str, questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Answers by question name, across as many requests as the batch size needs."""

        async def one(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
            for attempt in range(4):
                response = await self._http.post(
                    URL, json={"model": self.model, "input": evidence, "questions": batch}
                )
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response.json().get("answers", [])
            return []

        batches = [questions[i : i + BATCH] for i in range(0, len(questions), BATCH)]
        answers = [a for chunk in await asyncio.gather(*(one(b) for b in batches)) for a in chunk]
        return {a["name"]: a for a in answers}

    async def classify(self, nda_text: str, clauses: list[Clause]) -> list[Classification]:
        from .jev import KIND_CRITERIA

        choices = _choices(KIND_CRITERIA)
        questions = [
            {
                "type": "choice",
                "name": f"clause_{i}",
                "instructions": f"Which kind of control does clause {clause.id} of the agreement ask for? "
                f"Clause {clause.id}: {clause.text}",
                "choices": choices,
            }
            for i, clause in enumerate(clauses)
        ]
        answers = await self._ask(nda_text, questions)
        out = []
        for i, clause in enumerate(clauses):
            a = answers.get(f"clause_{i}", {})
            if a.get("type") != "choice":
                # A refusal or a missing answer: no rule from this clause, and visible as such.
                out.append(Classification(clause_id=clause.id, kind=ControlKind.BOILERPLATE, confidence=0.0))
                continue
            out.append(
                Classification(
                    clause_id=clause.id,
                    kind=ControlKind(a["choice"]),
                    confidence=float(a.get("confidence") or 0.0),
                    probabilities={p["value"]: float(p["probability"]) for p in a.get("probabilities", [])},
                )
            )
        return out

    async def classify_tools(self, tools: list[ToolSpec]) -> dict[str, tuple[str, float]]:
        if not tools:
            return {}
        evidence = "The agent's tools:\n" + "\n".join(
            f"- {t.name}({', '.join(t.args)}): {t.description}" for t in tools
        )
        questions = [
            {
                "type": "choice",
                "name": f"tool_{i}",
                "instructions": f"What does the tool '{tool.name}' do with the material it handles?",
                "choices": _choices(ROLE_CRITERIA),
            }
            for i, tool in enumerate(tools)
        ]
        answers = await self._ask(evidence, questions)
        out = {}
        for i, tool in enumerate(tools):
            a = answers.get(f"tool_{i}", {})
            if a.get("type") == "choice":
                out[tool.name] = (a["choice"], float(a.get("confidence") or 0.0))
            else:
                out[tool.name] = ("other", 0.0)
        return out

    async def verify(self, clause: Clause, control_payload: dict[str, Any], summary: str) -> float:
        evidence = (
            f"NDA clause: {clause.text}\n\nProposed control, in plain English: {summary}\n\n"
            "The control is one of several derived from this clause and is enforced automatically "
            "on an AI agent's tool calls. It is not expected to cover the whole clause by itself."
        )
        answers = await self._ask(
            evidence,
            [
                {
                    "type": "predicate",
                    "name": "wrong",
                    "instructions": "Would enforcing this control be a mistake under this clause? True if the "
                    "control forbids something the clause permits, targets material or a party the clause "
                    "does not cover, or restricts an action unrelated to what the clause restricts. False if "
                    "it restricts behaviour of the kind the clause restricts, for the material and parties "
                    "the clause covers, even if it only addresses part of the clause.",
                }
            ],
        )
        a = answers.get("wrong", {})
        return 1.0 - float(a["probability"]) if a.get("type") == "predicate" else 0.0

    async def predicates(self, nda_text: str, clauses: list[Clause], what: str, criteria: dict[str, Any]) -> dict[str, float]:
        """P(true) per clause for one yes/no question (the recipient check)."""

        instructions = f"{what} True: {_describe(criteria['true'])} False: {_describe(criteria['false'])}"
        questions = [
            {"type": "predicate", "name": f"q_{i}", "instructions": f"{instructions.format(id=c.id)} Clause {c.id}: {c.text}"}
            for i, c in enumerate(clauses)
        ]
        answers = await self._ask(nda_text, questions)
        return {
            c.id: float(answers[f"q_{i}"]["probability"])
            for i, c in enumerate(clauses)
            if answers.get(f"q_{i}", {}).get("type") == "predicate"
        }

    async def close(self) -> None:
        await self._http.aclose()
