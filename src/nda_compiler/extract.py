"""Pull the literals out of an enforceable clause, and prove they are real.

A fast text model fills a fixed form per clause. Every string it returns must
appear verbatim in the NDA; anything that does not is recorded as ungrounded
and the clause goes to human review instead of becoming a rule. That one check
removes invented party names and folder ids entirely.

Claude (ANTHROPIC_API_KEY) fills the form by default; OpenAI is the fallback,
and with neither key a regex stand-in fills the same form.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Protocol

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

# v2: three judgements the pipeline otherwise guesses from keywords, and
# sharper instructions on the fields that decide rules.
_SCHEMA_V2 = {
    **_SCHEMA,
    "properties": {
        **_SCHEMA["properties"],
        "applies": {"type": "boolean"},
        "consent_unlocks": {"type": "boolean"},
        "breach_is_material": {"type": "boolean"},
    },
}
_STRICT_SCHEMA_V2 = {**_SCHEMA_V2, "required": list(_SCHEMA_V2["properties"])}

_SYSTEM_V2 = """You read one clause of a confidentiality agreement and fill a fixed form about the duty it puts on the party receiving the confidential information. The form decides runtime rules for an AI agent that handles the disclosing party's documents, so precision matters more than coverage.

Copying values
- Copy every name, term and quote character for character from the clause, including its typographic quotes and apostrophes (’ “ ”). Never paraphrase, summarise or fix spelling.
- source_quote: the shortest exact sentence or phrase from the clause that states the duty.

Recipients
- permitted_recipients: only people or groups the clause allows to receive or see the information (for example "Representatives who need to know").
- prohibited_recipients: everyone the clause forbids, including named organisations ("Northwind Storage Partners"), competitors, bidders, and catch-alls such as "any other person". A named outside organisation is never a permitted recipient.
- Leave a list empty rather than invent an entry.

marked_terms: only markings or codenames the clause forbids reproducing, using or disclosing. Do not list defined terms such as "Confidential Information", and do not list a codename the clause merely mentions.

applies: true only if this clause itself imposes on the receiving party the kind of duty named in control_kind:
- permitted_recipients: limits who may see or receive the confidential information.
- third_party_disclosure: forbids sending, transmitting, emailing or uploading the information outside the receiving party. A general duty not to disclose is not this.
- marked_material: forbids reproducing a marking or codename.
- secure_processing: requires an isolated, segregated or secure environment for processing.
- use_restriction / personal_data: limits purpose / personal data handling.
False for clauses about standstills, non-solicitation, contacting employees or customers, process, remedies, compelled disclosure, definitions, or exceptions.

consent_unlocks: true if the restricted action may happen with the disclosing party's consent, approval, authorisation, permission or written agreement, however worded ("unless otherwise agreed in writing", "except with the prior written consent of the Company", "without our written agreement").

breach_is_material: true only if the clause itself says breaching it is a material breach, or entitles the disclosing party to terminate or to injunctive relief specifically for that clause. A general remedies clause covering the whole agreement does not count.

Return only the JSON object."""

# Models sometimes return typographic quotes as the C0 control character
# with the same low byte (U+2019 -> U+0019). Map them back before checking.
_CONTROL_QUOTES = str.maketrans(
    {"\x18": "\u2018", "\x19": "\u2019", "\x1c": "\u201c", "\x1d": "\u201d", "\x13": "\u2013", "\x14": "\u2014"}
)
_OTHER_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean_text(value: str) -> str:
    return _OTHER_CONTROLS.sub(" ", value.translate(_CONTROL_QUOTES))


def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            out[key] = clean_text(value)
        elif isinstance(value, list):
            out[key] = [clean_text(v) if isinstance(v, str) else v for v in value]
        else:
            out[key] = value
    return out


def prompt_version() -> str:
    """v2 (default) asks the extra judgements; v1 is the original form."""

    return os.environ.get("EXTRACT_PROMPT", "v2").strip() or "v2"


def _system() -> str:
    return _SYSTEM if prompt_version() == "v1" else _SYSTEM_V2


def _strict_schema() -> dict[str, Any]:
    return _STRICT_SCHEMA if prompt_version() == "v1" else _STRICT_SCHEMA_V2


def _user_message(clause: Clause, kind: ControlKind, definitions: str) -> str:
    return json.dumps(
        {
            "definitions": definitions,
            "clause_id": clause.id,
            "clause": clause.text,
            "control_kind": kind.value,
        }
    )


def _obligation(clause: Clause, kind: ControlKind, payload: dict[str, Any]) -> Obligation:
    if prompt_version() != "v1":
        payload = clean_payload(payload)
    # A marking is matched with `contains` on the tool's text, so the
    # quotation marks the clause puts around it must not travel with it.
    if isinstance(payload.get("marked_terms"), list):
        payload["marked_terms"] = [t.strip().strip('"“”\'‘’') for t in payload["marked_terms"] if t]
    return Obligation(
        clause_id=clause.id, kind=kind, **{k: v for k, v in payload.items() if v is not None}
    )


class Extractor(Protocol):
    model: str

    async def extract(self, clause: Clause, kind: ControlKind, definitions: str) -> Obligation: ...


def _squash(value: str) -> str:
    """Compare without whitespace or quote style: PDF extraction wraps words
    mid-line and curls quotes, neither of which makes a value a paraphrase."""

    curly = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})
    return re.sub(r"\s+", "", value).translate(curly).lower()


def in_clause(values: list[str], clause: Clause) -> list[str]:
    """Only the values that appear in this clause's own text."""

    haystack = _squash(f"{clause.heading}. {clause.text}")
    return [v for v in values if _squash(v) in haystack]


def _grounded(obligation: Obligation, nda_text: str) -> Obligation:
    haystack = _squash(nda_text)
    # Only values a rule is built from must be verbatim. Recipients never reach
    # a rule's conditions (they are filtered to the clause's own text later), so
    # a paraphrased recipient must not cost the clause its rules.
    literals = [*obligation.marked_terms, obligation.source_quote]
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
                ControlKind.SECURE_PROCESSING: "process",
            }.get(kind, "comply"),
            subject="Confidential Information",
            permitted_recipients=["Representatives"] if "Representatives" in text else [],
            prohibited_recipients=re.findall(r"(?:such as|to) ([A-Z][a-z]+(?: [A-Z][a-z]+)?)(?= or |,| is| constitutes)", text) or re.findall(r"such as ([A-Z][A-Za-z]+)", text),
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
        # Same clause, same extraction: a fixed seed keeps sampling as
        # deterministic as the API allows. gpt-5 models reason by default;
        # extraction is a copy task, so reasoning is turned down.
        extra: dict[str, Any] = {"seed": 7}
        if self.model.startswith("gpt-6"):
            # GPT-6 models take "none" .. "xhigh"; extraction is a copy task.
            extra["reasoning_effort"] = os.environ.get("OPENAI_REASONING", "").strip() or "none"
        elif self.model.startswith("gpt-5"):
            # gpt-5 / gpt-5-mini / gpt-5-nano take "minimal"; gpt-5.1 and later
            # take "none" .. "xhigh". OPENAI_REASONING overrides either.
            first_gen = self.model.split("-")[1] in {"5", "5"} and not self.model.startswith("gpt-5.")
            default = "minimal" if first_gen else "none"
            extra["reasoning_effort"] = os.environ.get("OPENAI_REASONING", "").strip() or default
        else:
            extra["temperature"] = 0
        response = await self._client.chat.completions.create(
            model=self.model,
            **extra,
            messages=[
                {"role": "system", "content": _system()},
                {"role": "user", "content": _user_message(clause, kind, definitions)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "obligation", "strict": True, "schema": _strict_schema()},
            },
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        return _obligation(clause, kind, payload)


class AnthropicExtractor:
    """Claude with structured output: the reply is the form's JSON, schema-checked."""

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic(api_key=api_key, max_retries=4)
        self.model = model

    async def extract(self, clause: Clause, kind: ControlKind, definitions: str) -> Obligation:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=_system(),
            messages=[{"role": "user", "content": _user_message(clause, kind, definitions)}],
            extra_body={"output_config": {"format": {"type": "json_schema", "schema": _strict_schema()}}},
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return _obligation(clause, kind, json.loads(text or "{}"))


def extractor_from_env() -> Extractor:
    """Claude when ANTHROPIC_API_KEY is set (the measured best), else OpenAI,
    else the offline stand-in. EXTRACT_PROVIDER=anthropic|openai|fake overrides."""

    provider = os.environ.get("EXTRACT_PROVIDER", "").strip().lower()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not provider:
        provider = "anthropic" if anthropic_key else "openai" if openai_key else "fake"
    if provider == "anthropic":
        return AnthropicExtractor(
            anthropic_key, os.environ.get("ANTHROPIC_MODEL", "").strip() or "claude-sonnet-5-5"
        )
    if provider == "openai" and openai_key:
        return OpenAIExtractor(openai_key, os.environ.get("OPENAI_MODEL", "").strip() or "gpt-5-mini")
    return FakeExtractor()


async def extract_all(
    extractor: Extractor,
    items: list[tuple[Clause, ControlKind]],
    definitions: str,
    nda_text: str,
) -> list[Obligation]:
    results = await asyncio.gather(*(extractor.extract(c, k, definitions) for c, k in items))
    return [_grounded(o, nda_text) for o in results]


_SMALL_WORDS = {"of", "and", "&", "for", "the", "de", "von"}


def is_defined_term(name: str, nda_text: str) -> bool:
    """'Representatives', 'Clean Team Members': terms the NDA defines in quotes."""

    return any(f"{q}{name}{r}" in nda_text for q, r in (('"', '"'), ("\u201c", "\u201d"), ("'", "'")))


def outsider_names(names: list[str], nda_text: str) -> list[str]:
    """Organisation names among a clause's recipients: every word capitalised,
    and not a term the NDA itself defines. 'Northwind Storage Partners' is one;
    'Representatives', 'Supplier Personnel' and "Brightwater's officers" are not."""

    out = []
    for name in names:
        words = [w for w in name.replace(",", " ").split() if w.lower() not in _SMALL_WORDS]
        if not words or not all(w[:1].isupper() for w in words) or "'" in name or "\u2019" in name:
            continue
        if is_defined_term(name, nda_text):
            continue
        out.append(name)
    return out
