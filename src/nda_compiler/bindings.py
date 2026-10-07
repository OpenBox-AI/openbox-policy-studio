"""The bindings table: NDA vocabulary → platform identifiers.

The NDA says "Representatives" and "the Disclosing Party"; OpenBox rules match
on agent ids and folder prefixes. No model can bridge that gap because the
answer is not in the document. The compliance team fills this in once per
matter, and every compile of that matter's NDA reuses it.

Which tools an agent has is not in the bindings either: it comes from the
agent's graph (see agent_graph.py). The yaml tool lists remain only as the
fallback for an agent whose graph has never been exported.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .agent_graph import AgentGraph, ToolSpec, load_graph


class AgentBinding(BaseModel):
    id: str
    name: str
    role: str = "representative"
    # Exported by scripts/export_graph.py; defaults to graphs/<id>.json.
    graph: str = ""


class ToolMap(BaseModel):
    """The tools a template may bind to, resolved for one agent."""

    read: list[ToolSpec] = Field(default_factory=list)
    file: list[ToolSpec] = Field(default_factory=list)
    outbound: list[ToolSpec] = Field(default_factory=list)
    graph: AgentGraph | None = None

    @property
    def access(self) -> list[ToolSpec]:
        return [*self.read, *self.file]


class Bindings(BaseModel):
    matter: str
    disclosing_party: str
    disclosing_party_aliases: list[str] = Field(default_factory=list)
    receiving_party: str
    covered_folders: list[str]
    # Named outsiders the NDA forbids disclosure to -> their folder prefix on the platform.
    competitor_folders: dict[str, str] = Field(default_factory=dict)
    representatives: list[AgentBinding]
    other_agents: list[AgentBinding] = Field(default_factory=list)
    # Fallbacks for agents without an exported graph.
    read_tools: list[str] = Field(default_factory=lambda: ["read_document"])
    file_tools: list[str] = Field(default_factory=lambda: ["upload_document"])
    outbound_tools: list[str] = Field(default_factory=lambda: ["send_email", "http_post"])
    purpose: str = ""
    codenames: list[str] = Field(default_factory=list)
    graphs: dict[str, AgentGraph] = Field(default_factory=dict, exclude=True)

    @property
    def all_agents(self) -> list[AgentBinding]:
        return [*self.representatives, *self.other_agents]

    @property
    def party_terms(self) -> list[str]:
        return [self.disclosing_party, *self.disclosing_party_aliases, *self.codenames]

    def competitor(self, name: str) -> tuple[str, str] | None:
        """(bindings name, folder) for a recipient the clause names, if mapped."""

        wanted = name.lower()
        for key, folder in self.competitor_folders.items():
            if key.lower() in wanted or wanted in key.lower():
                return key, folder
        return None

    def competitor_folder(self, name: str) -> str | None:
        hit = self.competitor(name)
        return hit[1] if hit else None

    def tools_for(self, agent: AgentBinding) -> ToolMap:
        graph = self.graphs.get(agent.id)
        if graph is None:
            return ToolMap(
                read=[
                    ToolSpec(name=n, args={"document_id": "string"}, role="reads_material")
                    for n in self.read_tools
                ],
                file=[
                    ToolSpec(
                        name=n, args={"destination_document_id": "string"}, role="files_to_store"
                    )
                    for n in self.file_tools
                ],
                outbound=[ToolSpec(name=n, role="sends_outbound") for n in self.outbound_tools],
            )
        return ToolMap(
            read=graph.with_role("reads_material"),
            file=graph.with_role("files_to_store"),
            outbound=graph.with_role("sends_outbound"),
            graph=graph,
        )


def load_bindings(path: Path, graphs_dir: Path | None = None) -> Bindings:
    bindings = Bindings.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    graphs_dir = graphs_dir if graphs_dir is not None else path.parent.parent / "graphs"
    for agent in bindings.all_agents:
        candidate = graphs_dir / (agent.graph or f"{agent.id}.json")
        if candidate.exists():
            bindings.graphs[agent.id] = load_graph(candidate)
    return bindings
