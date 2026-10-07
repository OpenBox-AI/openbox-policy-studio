"""The agent's own graph: which tools it has, what they take, and what they do.

The NDA names duties; the platform enforces them on tool calls. Which tool
calls exist is not a guess either side can make - it is in the agent's
compiled LangGraph, and in the activity events the OpenBox SDK has already
reported for it. Two sources, merged:

  introspected  the compiled graph: nodes, edges, every ToolNode's tools with
                their argument schemas (the same walk the SDK's
                tool_activity_binding does when it wraps a graph)
  observed      the agent's ActivityStarted events on OpenBox: the tool names
                actually called, how often, and the exact shape of
                activity_input the rules will be evaluated against

Roles (reads material, files it, sends it outside ...) are a judgement over
the tool's name, description and arguments, so a template asks for "the tools
that read stored material" and gets this agent's answer, not a hard-coded
read_document.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

ToolRole = Literal[
    "reads_material",
    "searches_index",
    "writes_internal",
    "files_to_store",
    "sends_outbound",
    "other",
]

ROLE_CRITERIA: dict[str, dict[str, str]] = {
    "reads_material": {
        "what": "Returns the content of a stored document, file or record to the agent, "
        "selected by an identifier or path.",
        "not_for": "Tools that only list or search and return titles or metadata.",
        "example": "read_document(document_id) -> the document text",
    },
    "searches_index": {
        "what": "Searches or lists documents and returns only titles, ids or metadata, "
        "never the content itself.",
        "example": "search_documents(query, limit) -> matching document ids",
    },
    "writes_internal": {
        "what": "Creates or edits a draft, report or note inside the agent's own working "
        "area, without placing it anywhere another party can reach.",
        "example": "write_briefing(content) -> path of the staged report",
    },
    "files_to_store": {
        "what": "Saves, copies, files or uploads content into a shared repository, client "
        "folder, drive or document system identified by a destination path.",
        "not_for": "Sending content to an external party over email, HTTP or chat.",
        "example": "upload_document(destination_document_id)",
    },
    "sends_outbound": {
        "what": "Transmits content to a party or service outside the firm: email, HTTP "
        "POST, webhook, chat message, SMS.",
        "example": "send_email(to, body), http_post(url, payload)",
    },
    "other": {
        "what": "Anything else: calculations, scheduling, formatting, control flow.",
        "example": "schedule_next_lead()",
    },
}

_FAKE_ROLE_RULES: list[tuple[ToolRole, re.Pattern[str]]] = [
    ("sends_outbound", re.compile(r"send|email|post|webhook|notify|publish|slack", re.I)),
    ("files_to_store", re.compile(r"upload|file_|destination|save_to|store|archive", re.I)),
    ("searches_index", re.compile(r"search|list|find|lookup|query", re.I)),
    ("reads_material", re.compile(r"read|fetch|get_|open|load", re.I)),
    ("writes_internal", re.compile(r"write|draft|note|report", re.I)),
]

_DOC_ARG_HINT = re.compile(r"document|path|file|destination|target|source|_id$|^id$", re.I)
_NOT_DOC_ARG = {"query", "content", "agent_slug", "limit", "text", "body", "message"}


class ToolSpec(BaseModel):
    name: str
    description: str = ""
    args: dict[str, str] = Field(default_factory=dict)  # arg name -> json type
    node: str = ""
    role: ToolRole = "other"
    role_confidence: float | None = None
    observed: int = 0
    observed_args: list[str] = Field(default_factory=list)
    input_shape: Literal["list", "dict", ""] = ""

    @property
    def document_arg(self) -> str | None:
        """The argument carrying the document path, observed first, then by schema."""

        for arg in self.observed_args:
            if arg not in _NOT_DOC_ARG and _DOC_ARG_HINT.search(arg):
                return arg
        for arg, kind in self.args.items():
            if kind == "string" and arg not in _NOT_DOC_ARG and _DOC_ARG_HINT.search(arg):
                return arg
        return None

    def input_path(self, arg: str) -> str:
        """The OPA input field for an argument, matching the shape the SDK sends.

        The LangGraph SDK reports tool arguments as [args, {"__openbox": ...}];
        until an event has been seen, that convention is assumed.
        """

        if self.input_shape == "dict":
            return f"activity_input.{arg}"
        return f"activity_input[0].{arg}"

    def example_input(self, arg: str, value: str) -> Any:
        if self.input_shape == "dict":
            return {arg: value}
        return [{arg: value}, {"__openbox": {"tool_type": "builtin"}}]


class GraphEdge(BaseModel):
    source: str
    target: str
    conditional: bool = False


class AgentGraph(BaseModel):
    agent_id: str
    name: str = ""
    nodes: list[str] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    def tool(self, name: str) -> ToolSpec | None:
        return next((t for t in self.tools if t.name == name), None)

    def with_role(self, *roles: str) -> list[ToolSpec]:
        return [t for t in self.tools if t.role in roles]

    def reaches(self, source: str, target: str) -> bool:
        """Whether a run can call `target` after `source` (path in the graph)."""

        out: dict[str, set[str]] = {}
        for edge in self.edges:
            out.setdefault(edge.source, set()).add(edge.target)
        seen, stack = set(), [source]
        while stack:
            node = stack.pop()
            for nxt in out.get(node, ()):
                if nxt == target:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def ordered_nodes(self) -> list[str]:
        """Nodes in first-visit order from __start__, for display."""

        out: dict[str, list[str]] = {}
        for edge in self.edges:
            out.setdefault(edge.source, []).append(edge.target)
        order, seen, queue = [], set(), ["__start__"]
        while queue:
            node = queue.pop(0)
            if node in seen:
                continue
            seen.add(node)
            if node not in ("__start__", "__end__"):
                order.append(node)
            queue.extend(sorted(out.get(node, [])))
        return order + [n for n in self.nodes if n not in seen and not n.startswith("__")]

    def summary(self) -> str:
        """Plain-English account of the graph, for the verifier and the review screen."""

        chain = " -> ".join(self.ordered_nodes())
        tools = "; ".join(
            f"{t.name}({', '.join(t.args)}) {t.role.replace('_', ' ')}" for t in self.tools
        )
        return f"Graph: {chain}. Tools: {tools}."


def introspect(compiled: Any, agent_id: str, name: str = "") -> AgentGraph:
    """Read a compiled LangGraph's nodes, edges and ToolNode tools.

    Walks the same wrapper attributes the OpenBox SDK does, so whatever the SDK
    would govern is what the compiler sees.
    """

    from langgraph.prebuilt import ToolNode

    drawn = compiled.get_graph()
    edges = [
        GraphEdge(source=e.source, target=e.target, conditional=bool(e.conditional))
        for e in drawn.edges
    ]
    tools: list[ToolSpec] = []
    for node_name, node in getattr(compiled, "nodes", {}).items():
        tool_node = _find_tool_node(node, ToolNode, set(), 0)
        if tool_node is None:
            continue
        for tool in tool_node.tools_by_name.values():
            schema = getattr(tool, "args", {}) or {}
            tools.append(
                ToolSpec(
                    name=tool.name,
                    description=(getattr(tool, "description", "") or "").strip(),
                    args={k: str(v.get("type", "string")) for k, v in schema.items()},
                    node=node_name,
                )
            )
    return AgentGraph(
        agent_id=agent_id,
        name=name,
        nodes=list(drawn.nodes),
        edges=edges,
        tools=tools,
        sources=["introspected"],
    )


def _find_tool_node(obj: Any, cls: type, seen: set[int], depth: int) -> Any:
    if obj is None or depth > 4 or id(obj) in seen:
        return None
    seen.add(id(obj))
    if isinstance(obj, cls):
        return obj
    for attr in ("bound", "node", "runnable", "steps", "func", "_func"):
        child = getattr(obj, attr, None)
        for candidate in child if isinstance(child, (list, tuple)) else (child,):
            found = _find_tool_node(candidate, cls, seen, depth + 1)
            if found is not None:
                return found
    return None


def observe(events: list[dict[str, Any]], graph: AgentGraph) -> AgentGraph:
    """Fold the agent's ActivityStarted events from OpenBox into the graph.

    Adds tools the graph did not list (an older deployment, a dynamic tool),
    and for every tool records how often it was called, which argument names
    appeared, and whether the input arrived as a list or a dict.
    """

    graph = graph.model_copy(deep=True)
    for event in events:
        if event.get("event_type") != "ActivityStarted":
            continue
        name = event.get("activity_type")
        if not name or name == "LangGraph":
            continue
        spec = graph.tool(name)
        if spec is None:
            spec = ToolSpec(name=name)
            graph.tools.append(spec)
        spec.observed += 1
        payload = event.get("input")
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            spec.input_shape = "list"
            args = payload[0]
        elif isinstance(payload, dict):
            spec.input_shape = "dict"
            args = payload
        else:
            continue
        for key in args:
            if key not in spec.observed_args:
                spec.observed_args.append(key)
            spec.args.setdefault(key, type(args[key]).__name__.replace("str", "string"))
    if any(t.observed for t in graph.tools) and "observed" not in graph.sources:
        graph.sources.append("observed")
    return graph


def fake_roles(tools: list[ToolSpec]) -> dict[str, tuple[ToolRole, float]]:
    """Keyword stand-in for the judge, for offline runs and tests."""

    out: dict[str, tuple[ToolRole, float]] = {}
    for tool in tools:
        # The name is the stronger signal ("read_document ... from search results").
        for text in (tool.name, tool.description):
            hit = next((role for role, p in _FAKE_ROLE_RULES if p.search(text)), None)
            if hit:
                out[tool.name] = (hit, 0.9)
                break
        else:
            out[tool.name] = ("other", 0.9)
    return out


def with_roles(graph: AgentGraph, roles: dict[str, tuple[str, float]]) -> AgentGraph:
    graph = graph.model_copy(deep=True)
    for tool in graph.tools:
        if tool.name in roles:
            tool.role, tool.role_confidence = roles[tool.name]  # type: ignore[assignment]
    return graph


def load_graph(path: Path) -> AgentGraph:
    return AgentGraph.model_validate(json.loads(path.read_text(encoding="utf-8")))


def save_graph(graph: AgentGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.model_dump_json(indent=2) + "\n", encoding="utf-8")
