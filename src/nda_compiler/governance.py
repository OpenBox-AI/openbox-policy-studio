"""Govern the compiler itself.

The agent that writes other agents' policies is the one you least want
running unsupervised. When COMPILER_OPENBOX_API_KEY is set, the compiled graph
is wrapped with the OpenBox LangGraph handler, so every stage is an activity
OpenBox sees and a REQUIRE_APPROVAL rule on the `apply` activity pauses the
run until a person approves the new NDA rules. Without the key the graph runs
bare, which is what tests and offline dry-runs want.
"""

from __future__ import annotations

import os
from typing import Any


def maybe_govern(graph: Any) -> Any:
    api_key = os.environ.get("COMPILER_OPENBOX_API_KEY", "").strip()
    if not api_key:
        return graph
    from openbox_langgraph import create_openbox_graph_handler

    workload_key = os.environ.get("COMPILER_OPENBOX_WORKLOAD_PRIVATE_KEY", "").strip()
    options: dict[str, Any] = {}
    if workload_key:
        options["workload_private_key"] = workload_key
    return create_openbox_graph_handler(
        graph=graph,
        api_url=os.environ.get("OPENBOX_API_URL", "https://core.openbox.ai").rstrip("/"),
        api_key=api_key,
        agent_name=os.environ.get("COMPILER_OPENBOX_AGENT_NAME", "NdaPolicyCompiler"),
        task_queue="nda-compiler",
        on_api_error="fail_closed",
        send_chain_start_event=True,
        send_chain_end_event=True,
        **options,
    )
