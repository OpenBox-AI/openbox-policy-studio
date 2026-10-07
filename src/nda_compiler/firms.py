"""Create a firm from its contract.

The bindings file used to be written by hand. Almost everything in it is in
the contract itself: the parties, the codenames and markings, the folder the
material lives in when the contract cites one, the Purpose and any named
competitors. This module reads those out of the document into a FirmProfile,
and writes the bindings once the onboarding person has confirmed them and
picked the agent. The only facts that are not in the contract are the agent
and whether it counts as a Representative.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_FOLDER = re.compile(r"\b(\d{4}/\d{5})\b")
_QUOTED = re.compile(r"[\"“]([^\"”]{3,60})[\"”]")
_GENERIC_MARKINGS = {
    "confidential",
    "confidential information",
    "restricted",
    "internal",
    "secret",
    "disclosing party",
    "receiving party",
    "representatives",
    "purpose",
    "agreement",
    "portfolio",
    "approved environment",
    "authorised persons",
    "permitted purpose",
    "bank",
    "adviser",
    "you",
    "we",
}

_SCHEMA = {
    "type": "object",
    "properties": {
        "disclosing_party": {"type": "string"},
        "disclosing_party_aliases": {"type": "array", "items": {"type": "string"}},
        "receiving_party": {"type": "string"},
        "codenames": {"type": "array", "items": {"type": "string"}},
        "purpose": {"type": "string"},
        "competitors": {"type": "array", "items": {"type": "string"}},
        "folder_references": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "disclosing_party",
        "disclosing_party_aliases",
        "receiving_party",
        "codenames",
        "purpose",
        "competitors",
        "folder_references",
    ],
    "additionalProperties": False,
}

_SYSTEM = (
    "You read a confidentiality agreement and return who the parties are and how the "
    "agreement refers to things. Copy names and terms verbatim from the text. "
    "disclosing_party: the party whose information is protected (if mutual, the party "
    "whose data room or material the agreement describes). disclosing_party_aliases: "
    "shorter names the text uses for that party (e.g. 'the Bank', 'Atlas'). codenames: "
    "project codenames and confidentiality markings the text puts in quotation marks, "
    "without the quotation marks. competitors: companies the agreement names as parties "
    "who must not receive the information. folder_references: any matter or folder "
    "numbers the text cites, exactly as written. purpose: the defined Purpose, verbatim."
)


class FirmProfile(BaseModel):
    disclosing_party: str = ""
    disclosing_party_aliases: list[str] = Field(default_factory=list)
    receiving_party: str = ""
    codenames: list[str] = Field(default_factory=list)
    purpose: str = ""
    competitors: list[str] = Field(default_factory=list)
    folder_references: list[str] = Field(default_factory=list)
    slug: str = ""

    def finish(self, text: str) -> FirmProfile:
        """Normalise what came back and fill gaps from the text itself."""

        squashed = " ".join(text.split())
        folders = [*self.folder_references, *_FOLDER.findall(squashed)]
        self.folder_references = sorted({_FOLDER.search(f).group(1) + "/" for f in folders if _FOLDER.search(f)})
        names = []
        for n in self.codenames:
            n = n.strip().strip("\"“”'")
            if n and n.lower() not in _GENERIC_MARKINGS and n not in names:
                names.append(n)
        self.codenames = names
        self.competitors = [c.strip() for c in self.competitors if c.strip() and c.lower() not in _GENERIC_MARKINGS]
        self.disclosing_party_aliases = [
            a.strip()
            for a in self.disclosing_party_aliases
            if a.strip() and a.strip() != self.disclosing_party and a.strip().lower() not in _GENERIC_MARKINGS
        ]
        if not self.slug:
            base = re.sub(r"\b(plc|ltd|limited|llp|ag|inc|gmbh|sa|co)\b\.?", "", self.disclosing_party, flags=re.I)
            self.slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:40] or "firm"
        return self


def profile_from_text_fake(text: str) -> FirmProfile:
    """Regex stand-in for offline runs and tests."""

    squashed = " ".join(text.split())
    m = re.search(r"between\s+(.+?)\s*\((?:the\s+)?\"?(?:Disclosing Party|the Bank|Bank)\"?\)", squashed, re.I)
    party = m.group(1).split(",")[0].strip() if m else ""
    if not party:
        m = re.search(r"From:\s*(.+?)\s*\(", squashed)
        party = m.group(1).strip() if m else ""
    recv = re.search(r"and\s+(Sellist[^,(]+)", squashed)
    purpose = re.search(r"\"Purpose\" means ([^.]+)\.", squashed) or re.search(r"\"Permitted Purpose\" means ([^.]+)\.", squashed)
    competitors = re.findall(r"such as ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*)", squashed)
    competitors += re.findall(r"disclose Confidential Information to ([A-Z][A-Za-z ]+?), ([A-Z][A-Za-z ]+?) or", squashed) and [x for pair in re.findall(r"disclose Confidential Information to ([A-Z][A-Za-z ]+?), ([A-Z][A-Za-z ]+?) or", squashed) for x in pair] or []
    return FirmProfile(
        disclosing_party=party,
        receiving_party=recv.group(1).strip() if recv else "",
        codenames=_QUOTED.findall(squashed),
        purpose=purpose.group(1).strip() if purpose else "",
        competitors=competitors,
    ).finish(text)


async def profile_from_text(text: str) -> FirmProfile:
    """One structured call over the contract's opening pages; regex when no key."""

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return profile_from_text_fake(text)
    from openai import AsyncOpenAI

    model = os.environ.get("OPENAI_MODEL", "gpt-5.5").strip() or "gpt-5.5"
    extra: dict[str, Any] = {"seed": 7}
    if model.startswith("gpt-5"):
        first_gen = not model.startswith("gpt-5.")
        extra["reasoning_effort"] = os.environ.get("OPENAI_REASONING", "").strip() or ("minimal" if first_gen else "none")
    else:
        extra["temperature"] = 0
    client = AsyncOpenAI(api_key=key)
    response = await client.chat.completions.create(
        model=model,
        **extra,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "firm_profile", "strict": True, "schema": _SCHEMA},
        },
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": text[:12000]},
        ],
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    return FirmProfile.model_validate(payload).finish(text)


def write_bindings(
    directory: Path,
    profile: FirmProfile,
    *,
    agent_id: str,
    agent_name: str,
    representative: bool,
    covered_folders: list[str],
    competitor_folders: dict[str, str],
) -> Path:
    slug = profile.slug
    agent = {"id": agent_id, "name": agent_name}
    body: dict[str, Any] = {
        "matter": slug,
        "disclosing_party": profile.disclosing_party,
        "disclosing_party_aliases": profile.disclosing_party_aliases,
        "receiving_party": profile.receiving_party or "Sellist Advisory LLP",
        "codenames": profile.codenames,
        "purpose": profile.purpose,
        "covered_folders": covered_folders,
        "competitor_folders": competitor_folders,
        "representatives": [agent] if representative else [],
        "other_agents": [] if representative else [agent],
    }
    path = directory / f"{slug}.yaml"
    path.write_text(
        f"# Created from the firm's contract in the Policy Studio.\n"
        + yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
