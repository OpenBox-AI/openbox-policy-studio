"""Obligation x bindings x agent graph -> OpenBox control payloads.

No model writes a rule. Each control kind has one template that fills slots
from the obligation (what the NDA says), the bindings (who the parties are on
the platform) and the agent's graph (which tools exist, what they take, what
they do). Payloads match the backend DTOs exactly; the API rejects unknown
fields with 422, so nothing extra goes in.

Every template also emits the test cases that /evaluate runs before the rule
is activated: one input the rule must catch, one it must let through. Test
inputs mirror the activity_input shape the SDK has actually been seen sending
for that tool.
"""

from __future__ import annotations

from typing import Any

from .agent_graph import ToolSpec
from .bindings import AgentBinding, Bindings
from .models import Control, ControlKind, Obligation, TestCase

# Guardrail enums from openbox-backend (modules/agent/enums).
GUARDRAIL_PII = "1"
GUARDRAIL_BANLIST = "4"
STAGE_OUTPUT = "1"
ON_FAIL_BLOCK = 1
# Behavior-rule verdicts (common/enums/verdict.enum.ts).
VERDICT_BLOCK = 3


def _field(path: str) -> dict[str, Any]:
    return {"kind": "field", "field": path, "transform": "value", "valueType": "string"}


def _literal(value: str) -> dict[str, Any]:
    return {"kind": "literal", "value": value, "valueType": "string"}


def _condition(cid: str, field: str, operator: str, value: str | None = None) -> dict[str, Any]:
    condition: dict[str, Any] = {"id": cid, "left": _field(field), "operator": operator}
    if value is not None:
        condition["right"] = _literal(value)
    return condition


def _rule_name(prefix: str, obligation: Obligation, bindings: Bindings) -> str:
    return f"NDA {bindings.matter} §{obligation.clause_id} {prefix}"[:255]


def _reason(obligation: Obligation, bindings: Bindings) -> str:
    return f"NDA {bindings.disclosing_party} §{obligation.clause_id}: {obligation.source_quote}"


def _binding(tool: ToolSpec, arg: str | None = None) -> dict[str, Any]:
    return {
        "tool": tool.name,
        "arg": arg,
        "node": tool.node,
        "role": tool.role,
        "observed": tool.observed,
        "input_shape": tool.input_shape or "list (SDK convention)",
    }


def _tool_event(agent_id: str, tool: ToolSpec, arg: str, document_id: str) -> dict[str, Any]:
    return {
        "event_type": "ActivityStarted",
        "agent_id": agent_id,
        "activity_type": tool.name,
        "activity_input": tool.example_input(arg, document_id),
    }


def permitted_recipients(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """Non-representatives may not read or file the Disclosing Party's material.

    One rule per (agent, tool that reads or files material, covered folder).
    The tool list and the argument that carries the path come from the
    agent's graph.
    """

    controls = []
    for agent in bindings.other_agents:
        for tool in bindings.tools_for(agent).access:
            arg = tool.document_arg
            if arg is None:
                continue  # a reader with no path argument cannot be scoped to a folder
            for folder in bindings.covered_folders:
                payload = {
                    "rule_name": _rule_name(f"{tool.name} {folder}", obligation, bindings),
                    "description": obligation.source_quote,
                    "priority": 90,
                    "match_mode": "all",
                    "conditions": [
                        _condition("tool", "activity_type", "equals", tool.name),
                        _condition("folder", tool.input_path(arg), "starts_with", folder),
                    ],
                    "decision": "BLOCK",
                    "reason": _reason(obligation, bindings),
                    "constraints": [],
                    "trust_impact": "medium",
                    "is_active": False,
                }
                other_folder = "9999/99999/"
                controls.append(
                    Control(
                        clause_id=obligation.clause_id,
                        kind=obligation.kind,
                        type="policy_rule",
                        agent_id=agent.id,
                        payload=payload,
                        binding=_binding(tool, arg),
                        tests=[
                            TestCase(
                                label=f"{agent.name} {tool.name} covered folder",
                                input=_tool_event(agent.id, tool, arg, f"{folder}deck.docx"),
                                expect="BLOCK",
                            ),
                            TestCase(
                                label=f"{agent.name} {tool.name} other folder",
                                input=_tool_event(agent.id, tool, arg, f"{other_folder}deck.docx"),
                                expect="ALLOW",
                            ),
                        ],
                    )
                )
    return controls


def third_party_disclosure(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """After reading covered material, no outbound send within the window.

    Behavior rules run over instrumented spans: the trigger is the outbound
    tool call, the prior state is a read. Both come from the graph, and the
    rule is only proposed for an agent whose graph can actually reach an
    outbound tool after a read. An agent with no outbound tool gets nothing
    here; the clause is reported as not applicable instead.
    """

    controls = []
    for agent in bindings.all_agents:
        tools = bindings.tools_for(agent)
        if not tools.outbound or not tools.read:
            continue
        reads = [
            r
            for r in tools.read
            if tools.graph is None or any(tools.graph.reaches(r.node, o.node) for o in tools.outbound)
        ] or tools.read
        for outbound in tools.outbound:
            # Known HTTP senders are matched on their semantic span type; any
            # other outbound tool on its tool-call span name.
            if outbound.name in ("http_post", "http_put", "http_patch"):
                trigger, trigger_match = outbound.name, []
            else:
                trigger = "llm_tool_call"
                trigger_match = [{"field": "name", "op": "eq", "value": outbound.name}]
            payload = {
                "rule_name": _rule_name(f"{outbound.name} after read", obligation, bindings),
                "description": obligation.source_quote,
                "priority": 90,
                "trigger": trigger,
                "trigger_match": trigger_match,
                "states": [
                    {
                        "semantic_type": "llm_tool_call",
                        "match": [{"field": "name", "op": "contains", "value": r.name}],
                    }
                    for r in reads
                ],
                "time_window": 3600,
                "verdict": VERDICT_BLOCK,
                "reject_message": _reason(obligation, bindings),
                "trust_impact": "high",
            }
            controls.append(
                Control(
                    clause_id=obligation.clause_id,
                    kind=obligation.kind,
                    type="behavior_rule",
                    agent_id=agent.id,
                    payload=payload,
                    binding={**_binding(outbound), "after": [r.name for r in reads]},
                )
            )
    return controls


def not_applicable(obligation: Obligation, bindings: Bindings) -> list[str]:
    """Why an enforceable clause yields nothing for these agents' graphs."""

    notes = []
    if obligation.kind == ControlKind.THIRD_PARTY_DISCLOSURE:
        for agent in bindings.all_agents:
            tools = bindings.tools_for(agent)
            if tools.outbound:
                continue
            path = " → ".join(tools.graph.ordered_nodes()) if tools.graph else "graph not exported"
            filing = ", ".join(t.name for t in tools.file) or "none"
            notes.append(
                f"§{obligation.clause_id} third-party disclosure: {agent.name} has no tool that "
                f"sends outside the firm (graph: {path}). Its only way to move material is "
                f"{filing}, which the access rules already scope to the covered folder."
            )
    if obligation.kind == ControlKind.USE_RESTRICTION:
        notes.append(
            f"§{obligation.clause_id} use restriction: whether a task serves the Purpose is not "
            f"a condition OpenBox can check on a tool call, a span or an output, so no policy "
            f"is proposed. The access rules keep the material to the covered folder."
        )
    if obligation.kind == ControlKind.PERMITTED_RECIPIENTS:
        for agent in bindings.other_agents:
            readers = [t for t in bindings.tools_for(agent).access if t.document_arg is None]
            for tool in readers:
                notes.append(
                    f"§{obligation.clause_id}: {tool.name}({', '.join(tool.args)}) on {agent.name} "
                    f"reads material but has no path argument, so it cannot be scoped to a folder."
                )
    return notes


def marked_material(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """Codenames and party names may not appear in any agent's output."""

    words = sorted({*obligation.marked_terms, *bindings.codenames})
    if not words:
        return []
    return [
        Control(
            clause_id=obligation.clause_id,
            kind=obligation.kind,
            type="guardrail",
            agent_id=agent.id,
            payload={
                "name": _rule_name("marked terms", obligation, bindings),
                "guardrail_type": GUARDRAIL_BANLIST,
                "processing_stage": STAGE_OUTPUT,
                "params": {"banned_words": words, "max_l_dist": 1},
                "settings": {
                    "on_fail": ON_FAIL_BLOCK,
                    "timeout": 5000,
                    "retry_attempts": 1,
                    "log_violation": True,
                },
                "trust_impact": "medium",
            },
            binding={"tool": "every output", "stage": "output"},
            tests=[
                TestCase(label="mentions codename", input={"text": words[0]}, expect="BLOCK"),
                TestCase(label="plain text", input={"text": "quarterly summary"}, expect="ALLOW"),
            ],
        )
        for agent in bindings.other_agents
    ]


def personal_data(obligation: Obligation, bindings: Bindings) -> list[Control]:
    return [
        Control(
            clause_id=obligation.clause_id,
            kind=obligation.kind,
            type="guardrail",
            agent_id=agent.id,
            payload={
                "name": _rule_name("personal data", obligation, bindings),
                "guardrail_type": GUARDRAIL_PII,
                "processing_stage": STAGE_OUTPUT,
                "params": {
                    "entities": ["EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "CREDIT_CARD"],
                    "replace_values": ["[email]", "[phone]", "[name]", "[card]"],
                },
                "settings": {
                    "on_fail": ON_FAIL_BLOCK,
                    "timeout": 5000,
                    "retry_attempts": 1,
                    "log_violation": True,
                },
                "trust_impact": "medium",
            },
            binding={"tool": "every output", "stage": "output"},
            tests=[
                TestCase(
                    label="contains email",
                    input={"text": "contact jane.doe@coca-cola.com"},
                    expect="BLOCK",
                ),
                TestCase(label="plain text", input={"text": "integration plan"}, expect="ALLOW"),
            ],
        )
        for agent in bindings.all_agents
    ]


TEMPLATES = {
    ControlKind.PERMITTED_RECIPIENTS: permitted_recipients,
    ControlKind.THIRD_PARTY_DISCLOSURE: third_party_disclosure,
    ControlKind.MARKED_MATERIAL: marked_material,
    ControlKind.PERSONAL_DATA: personal_data,
    # USE_RESTRICTION ("solely for the Purpose") has no platform policy: whether
    # a task serves the Purpose is not a condition on a tool call, a span or an
    # output. It is reported under not_applicable, never invented.
}


def build_controls(obligation: Obligation, bindings: Bindings) -> list[Control]:
    template = TEMPLATES.get(obligation.kind)
    return template(obligation, bindings) if template else []


def describe(control: Control, bindings: Bindings) -> str:
    """The control in plain English, for the verifier and the review screen.

    JEV reads literally, so it is shown what the rule does to whom, not the
    payload's operators, enum numbers and placeholder ids.
    """

    names = {a.id: a.name for a in bindings.all_agents}
    agent = names.get(control.agent_id, control.agent_id)
    is_rep = control.agent_id in {a.id for a in bindings.representatives}
    role = "a Representative" if is_rep else "not a Representative"
    reps = ", ".join(a.name for a in bindings.representatives)
    rep_note = f"; the Representatives are {reps}" if reps else "; no agent is a Representative"
    folders = ", ".join(bindings.covered_folders)
    party = bindings.disclosing_party
    p = control.payload
    b = control.binding
    if control.type == "policy_rule":
        tool = p["conditions"][0]["right"]["value"]
        what = "file" if b.get("role") == "files_to_store" else "read"
        return (
            f"Agent {agent} ({role} under this NDA{rep_note}) is blocked from calling its "
            f"'{tool}' tool to {what} any document whose '{b.get('arg')}' lies under the "
            f"{party} folder {folders}. Documents in other folders are allowed."
        )
    if control.type == "behavior_rule":
        reads = ", ".join(b.get("after", []))
        return (
            f"If agent {agent} has called {reads} within the last {p['time_window'] // 60} "
            f"minutes, its '{b.get('tool')}' tool (which sends content outside the firm's "
            f"systems) is blocked."
        )
    if p["guardrail_type"] == GUARDRAIL_PII:
        return (
            f"Every output agent {agent} produces is scanned for personal data (names, "
            f"email addresses, phone numbers, card numbers); any output containing such "
            f"personal data is blocked before delivery."
        )
    return (
        f"Every output agent {agent} produces is scanned for the confidential markings "
        f"{p['params']['banned_words']}; any output containing one is blocked before delivery."
    )


def summarize(control: Control, agents: list[AgentBinding]) -> str:
    names = {a.id: a.name for a in agents}
    p = control.payload
    return (
        f"{control.type} on agent {names.get(control.agent_id, control.agent_id)}: "
        f"{p.get('rule_name') or p.get('name')}"
    )
