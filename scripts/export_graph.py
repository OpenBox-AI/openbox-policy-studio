"""Export an agent's compiled LangGraph for the compiler.

Runs inside the agent's own environment, because compiling the graph needs
the agent's code. No model is called: a fake chat model stands in so the
graph can be built and read.

    cd ~/openbox-barrier-demo && uv run --with pydantic \
      env PYTHONPATH=~/openbox-nda-compiler/src python ~/openbox-nda-compiler/scripts/export_graph.py \
      --profile amy --agent-id 16e4c8e5-0921-4946-ae5b-96fe1966e57d --name "compliance agent" \
      --out ~/openbox-nda-compiler/graphs/16e4c8e5-0921-4946-ae5b-96fe1966e57d.json
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from nda_compiler.agent_graph import introspect, save_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="amy", help="barrier-demo profile whose graph to compile")
    parser.add_argument("--agent-id", required=True, help="OpenBox agent id the graph runs as")
    parser.add_argument("--name", default="", help="display name")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from openbox_langgraph_client_intelligence.profiles import get_profile
    from openbox_langgraph_client_intelligence.repository import DocumentRepository
    from openbox_langgraph_client_intelligence.tools import build_document_tools
    from openbox_langgraph_client_intelligence.workflow import build_research_graph

    root = Path(tempfile.mkdtemp())
    for sub in ("library", "output", "filed"):
        (root / sub).mkdir()
    repository = DocumentRepository(root / "library", root / "output", root / "filed")
    tools = build_document_tools(repository, agent_slug=args.profile)
    compiled = build_research_graph(get_profile(args.profile), tools, FakeListChatModel(responses=["-"]))

    graph = introspect(compiled, agent_id=args.agent_id, name=args.name or args.profile)
    save_graph(graph, args.out)
    print(f"{args.out}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, {len(graph.tools)} tools")
    for tool in graph.tools:
        print(f"  {tool.node}: {tool.name}({', '.join(tool.args)})")


if __name__ == "__main__":
    main()
