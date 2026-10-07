"""The bindings table: NDA vocabulary → platform identifiers.

The NDA says "Representatives" and "the Disclosing Party"; OpenBox rules match
on agent ids and folder prefixes. No model can bridge that gap because the
answer is not in the document. The compliance team fills this in once per
matter, and every compile of that matter's NDA reuses it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AgentBinding(BaseModel):
    id: str
    name: str
    role: str = "representative"


class Bindings(BaseModel):
    matter: str
    disclosing_party: str
    disclosing_party_aliases: list[str] = Field(default_factory=list)
    receiving_party: str
    covered_folders: list[str]
    representatives: list[AgentBinding]
    other_agents: list[AgentBinding] = Field(default_factory=list)
    read_tools: list[str] = Field(default_factory=lambda: ["read_document"])
    file_tools: list[str] = Field(default_factory=lambda: ["upload_document"])
    outbound_tools: list[str] = Field(default_factory=lambda: ["send_email", "http_post"])
    purpose: str = ""
    codenames: list[str] = Field(default_factory=list)

    @property
    def all_agents(self) -> list[AgentBinding]:
        return [*self.representatives, *self.other_agents]

    @property
    def party_terms(self) -> list[str]:
        return [self.disclosing_party, *self.disclosing_party_aliases, *self.codenames]


def load_bindings(path: Path) -> Bindings:
    return Bindings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
