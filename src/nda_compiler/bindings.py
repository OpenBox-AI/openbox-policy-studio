"""The bindings: the few facts about a firm that are not in its contract.

The NDA says "the Disclosing Party" and "Confidential Information"; OpenBox
rules match on an agent id and a folder prefix. So a firm's bindings hold the
party's names (to recognise its documents), the folder its material lives in,
and the agent its policies apply to. Nothing else: which tools the agent has
comes from its compiled graph, and the yaml tool lists remain only as the
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
    # Exported by scripts/export_graph.py; defaults to graphs/<id>.json.
    graph: str = ""


class ToolMap(BaseModel):
    """The tools a template may bind to, resolved for the agent."""

    read: list[ToolSpec] = Field(default_factory=list)
    file: list[ToolSpec] = Field(default_factory=list)
    outbound: list[ToolSpec] = Field(default_factory=list)
    graph: AgentGraph | None = None

    @property
    def access(self) -> list[ToolSpec]:
        return [*self.read, *self.file]


class Bindings(BaseModel):
    firm: str
    disclosing_party: str
    disclosing_party_aliases: list[str] = Field(default_factory=list)
    receiving_party: str = ""
    codenames: list[str] = Field(default_factory=list)
    purpose: str = ""
    covered_folders: list[str]
    agent: AgentBinding
    # Fallbacks for an agent without an exported graph.
    read_tools: list[str] = Field(default_factory=lambda: ["read_document"])
    file_tools: list[str] = Field(default_factory=lambda: ["upload_document"])
    outbound_tools: list[str] = Field(default_factory=lambda: ["send_email", "http_post"])
    graphs: dict[str, AgentGraph] = Field(default_factory=dict, exclude=True)

    @property
    def all_agents(self) -> list[AgentBinding]:
        return [self.agent]

    @property
    def party_terms(self) -> list[str]:
        return [self.disclosing_party, *self.disclosing_party_aliases, *self.codenames]

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
    agent = bindings.agent
    candidate = graphs_dir / (agent.graph or f"{agent.id}.json")
    if candidate.exists():
        bindings.graphs[agent.id] = load_graph(candidate)
    return bindings
