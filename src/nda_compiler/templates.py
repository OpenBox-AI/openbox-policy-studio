"""Obligation x bindings -> OpenBox control payloads.

No model writes a rule. Each control kind has one template that fills slots
from the obligation (what the NDA says) and the bindings (what that means on
the platform). Payloads match the backend DTOs exactly; the API rejects
unknown fields with 422, so nothing extra goes in.

Every template also emits the test cases that /evaluate runs before the rule
is activated: one input the rule must catch, one it must let through.
"""

from __future__ import annotations

from typing import Any

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


def _document_arg(tool: str, bindings: Bindings) -> str:
    """Which tool argument carries the document path (confirmed against live events)."""

    return "destination_document_id" if tool in bindings.file_tools else "document_id"


def _tool_event(agent_id: str, tool: str, arg: str, document_id: str) -> dict[str, Any]:
    # The LangGraph SDK sends tool arguments as a list: [args, {"__openbox": ...}].
    # Rules therefore address activity_input[0]; the test input mirrors that shape.
    return {
        "event_type": "ActivityStarted",
        "agent_id": agent_id,
        "activity_type": tool,
        "activity_input": [{arg: document_id}, {"__openbox": {"tool_type": "builtin"}}],
    }


def permitted_recipients(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """Non-representatives may not read or file the Disclosing Party's material."""

    controls = []
    tools = [*bindings.read_tools, *bindings.file_tools]
    for agent in bindings.other_agents:
        for tool in tools:
            for folder in bindings.covered_folders:
                arg = _document_arg(tool, bindings)
                payload = {
                    "rule_name": _rule_name(f"{tool} {folder}", obligation, bindings),
                    "description": obligation.source_quote,
                    "priority": 90,
                    "match_mode": "all",
                    "conditions": [
                        _condition("tool", "activity_type", "equals", tool),
                        _condition("folder", f"activity_input[0].{arg}", "starts_with", folder),
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
                        tests=[
                            TestCase(
                                label=f"{agent.name} {tool} covered folder",
                                input=_tool_event(agent.id, tool, arg, f"{folder}deck.docx"),
                                expect="BLOCK",
                            ),
                            TestCase(
                                label=f"{agent.name} {tool} other folder",
                                input=_tool_event(agent.id, tool, arg, f"{other_folder}deck.docx"),
                                expect="ALLOW",
                            ),
                        ],
                    )
                )
    return controls


def third_party_disclosure(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """After reading covered material, no outbound send within the window."""

    controls = []
    for agent in bindings.all_agents:
        for tool in bindings.outbound_tools:
            payload = {
                "rule_name": _rule_name(f"outbound after read via {tool}", obligation, bindings),
                "description": obligation.source_quote,
                "priority": 90,
                "trigger": "llm_tool_call",
                "trigger_match": [{"field": "tool_name", "op": "equals", "value": tool}],
                "states": [
                    {
                        "semantic_type": "llm_tool_call",
                        "match": [
                            {"field": "tool_name", "op": "in", "value": bindings.read_tools},
                            {
                                "field": "document_id",
                                "op": "starts_with",
                                "value": folder,
                            },
                        ],
                    }
                    for folder in bindings.covered_folders
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
                )
            )
    return controls


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


def use_restriction(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """A judgement question the harness asks JEV at runtime, carrying the Purpose."""

    purpose = obligation.purpose or bindings.purpose
    if not purpose:
        return []
    return [
        Control(
            clause_id=obligation.clause_id,
            kind=obligation.kind,
            type="judgement",
            agent_id=agent.id,
            payload={
                "rule_name": _rule_name("use within Purpose", obligation, bindings),
                "triggers": ["llm_tool_call"],
                "trigger_match": [{"field": "tool_name", "op": "in", "value": bindings.read_tools}],
                "question": {
                    "type": "noul",
                    "instructions": {
                        "what": "Is this use of the Disclosing Party's Confidential Information "
                        "outside the Purpose defined in the NDA?",
                        "purpose": purpose,
                        "true": "The task the agent is performing serves another client, "
                        "benchmarking, marketing or any aim other than the Purpose.",
                        "false": "The task is a step in " + purpose + ".",
                    },
                },
                "model": "jev-latest",
                "block_above": 0.7,
                "verdict": VERDICT_BLOCK,
                "reject_message": _reason(obligation, bindings),
                "on_unavailable": "block",
            },
        )
        for agent in bindings.all_agents
    ]


TEMPLATES = {
    ControlKind.PERMITTED_RECIPIENTS: permitted_recipients,
    ControlKind.THIRD_PARTY_DISCLOSURE: third_party_disclosure,
    ControlKind.MARKED_MATERIAL: marked_material,
    ControlKind.PERSONAL_DATA: personal_data,
    ControlKind.USE_RESTRICTION: use_restriction,
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
    reps = ", ".join(a.name for a in bindings.representatives) or "none"
    role = (
        "a Representative"
        if control.agent_id in {a.id for a in bindings.representatives}
        else "not a Representative"
    )
    folders = ", ".join(bindings.covered_folders)
    party = bindings.disclosing_party
    p = control.payload
    if control.type == "policy_rule":
        tool = p["conditions"][0]["right"]["value"]
        return (
            f"Agent {agent} ({role} under this NDA; the Representatives are {reps}) is blocked "
            f"from calling the tool '{tool}' on any document stored under the {party} folder "
            f"{folders}. Attempts on documents in other folders are allowed."
        )
    if control.type == "behavior_rule":
        tool = p["trigger_match"][0]["value"]
        return (
            f"If agent {agent} has read any document from the {party} folder {folders} within "
            f"the last {p['time_window'] // 60} minutes, then calling the outbound tool "
            f"'{tool}' (which sends content outside the firm's systems) is blocked."
        )
    if control.type == "guardrail":
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
    purpose = p["question"]["instructions"]["purpose"]
    return (
        f"Whenever agent {agent} reads a {party} document, an independent judge is asked "
        f"whether the task being performed falls outside the Purpose ('{purpose}'); if the "
        f"probability that it does exceeds {p['block_above']}, the read is blocked."
    )


def summarize(control: Control, agents: list[AgentBinding]) -> str:
    names = {a.id: a.name for a in agents}
    p = control.payload
    return (
        f"{control.type} on agent {names.get(control.agent_id, control.agent_id)}: "
        f"{p.get('rule_name') or p.get('name')}"
    )
