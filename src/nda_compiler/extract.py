"""Pull the literals out of an enforceable clause, and prove they are real.

A fast text model fills a fixed form per clause. Every string it returns must
appear verbatim in the NDA; anything that does not is recorded as ungrounded
and the clause goes to human review instead of becoming a rule. That one check
removes invented party names and folder ids entirely.

With no OPENAI_API_KEY a regex stand-in fills the same form.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Protocol

from .models import Clause, ControlKind, Obligation

_SCHEMA = {
    "type": "object",
    "properties": {
        "bound_party": {"type": "string"},
        "action": {"type": "string"},
        "subject": {"type": "string"},
        "permitted_recipients": {"type": "array", "items": {"type": "string"}},
        "prohibited_recipients": {"type": "array", "items": {"type": "string"}},
        "exceptions": {"type": "array", "items": {"type": "string"}},
        "marked_terms": {"type": "array", "items": {"type": "string"}},
        "purpose": {"type": ["string", "null"]},
        "duration": {"type": ["string", "null"]},
        "source_quote": {"type": "string"},
    },
    "required": ["bound_party", "action", "subject", "source_quote"],
    "additionalProperties": False,
}

# OpenAI strict mode needs every property listed as required; optional ones stay nullable.
_STRICT_SCHEMA = {**_SCHEMA, "required": list(_SCHEMA["properties"])}

_SYSTEM = (
    "You extract obligations from NDA clauses into a fixed form. Copy every value "
    "verbatim from the clause or the definitions; never paraphrase a name, term or "
    "quote. Return only the JSON object."
)


class Extractor(Protocol):
    model: str

    async def extract(self, clause: Clause, kind: ControlKind, definitions: str) -> Obligation: ...


def _squash(value: str) -> str:
    """Compare without whitespace or quote style: PDF extraction wraps words
    mid-line and curls quotes, neither of which makes a value a paraphrase."""

    curly = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
    return re.sub(r"\s+", "", value).translate(curly).lower()


def _grounded(obligation: Obligation, nda_text: str) -> Obligation:
    haystack = _squash(nda_text)
    literals = [
        *obligation.permitted_recipients,
        *obligation.prohibited_recipients,
        *obligation.marked_terms,
        obligation.source_quote,
    ]
    missing = [value for value in literals if value and _squash(value) not in haystack]
    return obligation.model_copy(update={"ungrounded": missing})


class FakeExtractor:
    model = "fake-extractor"

    async def extract(self, clause: Clause, kind: ControlKind, definitions: str) -> Obligation:
        text = clause.text
        quoted = re.findall(r'"([^"]+)"', definitions)
        return Obligation(
            clause_id=clause.id,
            kind=kind,
            bound_party="Receiving Party" if "Receiving Party" in text else "both parties",
            action={
                ControlKind.PERMITTED_RECIPIENTS: "disclose",
                ControlKind.THIRD_PARTY_DISCLOSURE: "transmit",
                ControlKind.USE_RESTRICTION: "use",
                ControlKind.MARKED_MATERIAL: "reproduce",
                ControlKind.PERSONAL_DATA: "reproduce",
            }.get(kind, "comply"),
            subject="Confidential Information",
            permitted_recipients=["Representatives"] if "Representatives" in text else [],
            prohibited_recipients=re.findall(r"such as ([A-Z][A-Za-z]+)", text),
            marked_terms=[q for q in quoted if q not in {"Disclosing Party", "Receiving Party"}]
            if kind == ControlKind.MARKED_MATERIAL
            else [],
            purpose=_purpose(definitions) if kind == ControlKind.USE_RESTRICTION else None,
            source_quote=text.split(". ")[0],
        )


def _purpose(definitions: str) -> str | None:
    match = re.search(r'"Purpose" means ([^.]+)\.', definitions)
    return match.group(1).strip() if match else None


class OpenAIExtractor:
    """Strict JSON-schema structured output; the model cannot add or drop a field."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def extract(self, clause: Clause, kind: ControlKind, definitions: str) -> Obligation:
        # gpt-5 models reason by default; extraction is a copy task, so turn it off.
        extra = {"reasoning_effort": "minimal"} if self.model.startswith("gpt-5") else {}
        response = await self._client.chat.completions.create(
            model=self.model,
            **extra,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "definitions": definitions,
                            "clause_id": clause.id,
                            "clause": clause.text,
                            "control_kind": kind.value,
                        }
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "obligation", "strict": True, "schema": _STRICT_SCHEMA},
            },
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return Obligation(
            clause_id=clause.id, kind=kind, **{k: v for k, v in payload.items() if v is not None}
        )


def extractor_from_env() -> Extractor:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return FakeExtractor()
    model = os.environ.get("OPENAI_MODEL", "").strip() or "gpt-5-mini"
    return OpenAIExtractor(key, model)


async def extract_all(
    extractor: Extractor,
    items: list[tuple[Clause, ControlKind]],
    definitions: str,
    nda_text: str,
) -> list[Obligation]:
    results = await asyncio.gather(*(extractor.extract(c, k, definitions) for c, k in items))
    return [_grounded(o, nda_text) for o in results]
