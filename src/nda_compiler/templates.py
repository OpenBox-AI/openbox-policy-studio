"""Obligation x bindings x agent graph -> OpenBox policy rules.

One output type: the policy rule an agent's Policies tab lists
(CreatePolicyRuleDto: structured conditions over the tool call, a decision).
No model writes a rule. Each clause kind has one template that fills slots
from the obligation (what the NDA says), the bindings (who the parties are on
the platform) and the agent's graph (which tools exist and what they take).
A clause that cannot be expressed as conditions on a tool call is reported
as not applicable, with the reason, and nothing is invented for it.

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


def _binding(tool: ToolSpec, arg: str | None, what: str) -> dict[str, Any]:
    return {
        "tool": tool.name,
        "arg": arg,
        "node": tool.node,
        "role": tool.role,
        "observed": tool.observed,
        "input_shape": tool.input_shape or "list (SDK convention)",
        "what": what,
    }


def _tool_event(agent_id: str, tool: ToolSpec, arg: str, value: str) -> dict[str, Any]:
    return {
        "event_type": "ActivityStarted",
        "agent_id": agent_id,
        "activity_type": tool.name,
        "activity_input": tool.example_input(arg, value),
    }


def _rule(
    obligation: Obligation,
    bindings: Bindings,
    agent: AgentBinding,
    tool: ToolSpec,
    name: str,
    conditions: list[dict[str, Any]],
    tests: list[TestCase],
    binding: dict[str, Any],
    priority: int = 90,
    decision: str | None = None,
) -> Control:
    """One policy rule. Decisions, in the platform's own terms:

    CONSTRAIN        let it run, inside the sandbox (the platform's only constraint)
    REQUIRE_APPROVAL hold for a person; chosen when the clause says "with consent"
    BLOCK            refuse the call; "shall not"
    HALT             refuse and stop the run; "material breach", "injunctive relief"
    """

    decision = decision or obligation.decision
    return Control(
        clause_id=obligation.clause_id,
        kind=obligation.kind,
        type="policy_rule",
        agent_id=agent.id,
        payload={
            "rule_name": _rule_name(name, obligation, bindings),
            "description": obligation.source_quote,
            "priority": priority,
            "match_mode": "all",
            "conditions": conditions,
            "decision": decision,
            "reason": _reason(obligation, bindings),
            "constraints": ["run_in_sandbox"] if decision == "CONSTRAIN" else [],
            "trust_impact": "none" if decision == "ALLOW" else "medium",
            "is_active": False,
        },
        tests=tests,
        binding={**binding, "decision": decision},
    )


def _folder_rule(
    obligation: Obligation,
    bindings: Bindings,
    agent: AgentBinding,
    tool: ToolSpec,
    folder: str,
    decision: str,
    label: str,
    priority: int = 90,
) -> Control | None:
    arg = tool.document_arg
    if arg is None:
        return None  # reported by not_applicable()
    what = "file" if tool.role == "files_to_store" else "read"
    other_folder = "9999/99999/"
    return _rule(
        obligation,
        bindings,
        agent,
        tool,
        f"{tool.name} {folder}{label}",
        [
            _condition("tool", "activity_type", "equals", tool.name),
            _condition("folder", tool.input_path(arg), "starts_with", folder),
        ],
        [
            TestCase(
                label=f"{agent.name} {tool.name} {folder}",
                input=_tool_event(agent.id, tool, arg, f"{folder}deck.docx"),
                expect=decision,
            ),
            TestCase(
                label=f"{agent.name} {tool.name} other folder",
                input=_tool_event(agent.id, tool, arg, f"{other_folder}deck.docx"),
                expect="ALLOW",
            ),
        ],
        {**_binding(tool, arg, what), "folder": folder},
        priority=priority,
        decision=decision,
    )


def permitted_recipients(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """Who may handle the Disclosing Party's material.

    The agent is not a party to the NDA, so the clause's restriction applies
    to it in full: one rule per tool that reads or files material, on the
    folder the material lives in, with the clause's decision (BLOCK for
    "shall not", REQUIRE_APPROVAL for "with consent", HALT for "material
    breach").
    """

    # A clause that only names outsiders ("shall not disclose to Northwind or
    # Kestrel") is about who a recipient is, which a rule on the agent's own
    # folders cannot see; it is reported, not turned into a folder rule.
    named = [
        n
        for n in obligation.prohibited_recipients
        if n[:1].isupper() and not n.lower().startswith(("any", "other", "the "))
    ]
    if named and not obligation.permitted_recipients:
        return []
    controls: list[Control] = []
    for tool in bindings.tools_for(bindings.agent).access:
        for folder in bindings.covered_folders:
            rule = _folder_rule(obligation, bindings, bindings.agent, tool, folder, obligation.decision, "")
            if rule:
                controls.append(rule)
    return controls


def secure_processing(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """Material may be handled only inside an isolated environment: CONSTRAIN (sandbox).

    The platform's one constraint is run_in_sandbox, so every read of the
    covered folder by any agent runs sandboxed. Priority 80 keeps it below the
    access decisions above it.
    """

    controls = []
    agent = bindings.agent
    for tool in bindings.tools_for(agent).read:
        for folder in bindings.covered_folders:
            rule = _folder_rule(obligation, bindings, agent, tool, folder, "CONSTRAIN", " sandboxed", 80)
            if rule:
                controls.append(rule)
    return controls


def third_party_disclosure(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """Nothing may be sent outside the firm: block every outbound tool the agent has.

    A policy rule sees one tool call, not what was read before it, so the rule
    is on the sending itself. An agent whose graph has no outbound tool gets
    nothing here and the clause is reported as not applicable.
    """

    controls = []
    for agent in bindings.all_agents:
        tools = bindings.tools_for(agent)
        # Tools core has seen making outbound HTTP calls count as senders too,
        # whatever the judge made of their name: the spans are the evidence.
        http_tools = tools.graph.tools_with_span("http_post", "http_put", "http_patch", "http") if tools.graph else []
        senders = [*tools.outbound, *(t for t in http_tools if t not in tools.outbound)]
        for tool in senders:
            arg = tool.text_arg or next(iter(tool.args), "payload")
            controls.append(
                _rule(
                    obligation,
                    bindings,
                    agent,
                    tool,
                    f"{tool.name} outbound",
                    [_condition("tool", "activity_type", "equals", tool.name)],
                    [
                        TestCase(
                            label=f"{agent.name} {tool.name}",
                            input=_tool_event(agent.id, tool, arg, "quarterly summary"),
                            expect=obligation.decision,
                        ),
                        TestCase(
                            label=f"{agent.name} other tool",
                            input={
                                "event_type": "ActivityStarted",
                                "agent_id": agent.id,
                                "activity_type": "search_documents",
                                "activity_input": [{"query": "x"}, {"__openbox": {}}],
                            },
                            expect="ALLOW",
                        ),
                    ],
                    {**_binding(tool, None, "send"), "spans": tool.span_types},
                )
            )
    return controls


def marked_material(obligation: Obligation, bindings: Bindings) -> list[Control]:
    """Codenames and markings may not appear in anything the agent writes, files or sends.

    One rule per (agent, tool with a free-text argument, marked term): block
    the call when that argument contains the term.
    """

    words = sorted({*obligation.marked_terms, *bindings.codenames})
    controls = []
    for agent in bindings.all_agents:
        tools = bindings.tools_for(agent)
        for tool in [*tools.file, *tools.outbound, *(tools.graph.with_role("writes_internal") if tools.graph else [])]:
            arg = tool.text_arg
            if arg is None:
                continue
            for word in words:
                controls.append(
                    _rule(
                        obligation,
                        bindings,
                        agent,
                        tool,
                        f"{tool.name} mentions {word}",
                        [
                            _condition("tool", "activity_type", "equals", tool.name),
                            _condition("term", tool.input_path(arg), "contains", word),
                        ],
                        [
                            TestCase(
                                label=f"{agent.name} {tool.name} mentions {word}",
                                input=_tool_event(agent.id, tool, arg, f"Notes on {word} pricing"),
                                expect=obligation.decision,
                            ),
                            TestCase(
                                label=f"{agent.name} {tool.name} plain text",
                                input=_tool_event(agent.id, tool, arg, "quarterly summary"),
                                expect="ALLOW",
                            ),
                        ],
                        {**_binding(tool, arg, "mention"), "term": word},
                    )
                )
    return controls


def not_applicable(obligation: Obligation, bindings: Bindings) -> list[str]:
    """Why an enforceable clause yields no policy rule for these agents."""

    notes = []
    if obligation.kind == ControlKind.THIRD_PARTY_DISCLOSURE:
        for agent in bindings.all_agents:
            tools = bindings.tools_for(agent)
            if tools.outbound:
                continue
            if tools.graph and tools.graph.tools_with_span("http_post", "http_put", "http_patch", "http"):
                continue  # covered by the span-evidenced senders above
            path = " → ".join(tools.graph.ordered_nodes()) if tools.graph else "graph not exported"
            filing = ", ".join(t.name for t in tools.file) or "none"
            spans = sorted({s for t in (tools.graph.tools if tools.graph else []) for s in t.span_types})
            seen = f" Spans recorded in its calls: {', '.join(spans)}; no outbound HTTP among them." if spans else ""
            notes.append(
                f"§{obligation.clause_id} third-party disclosure: {agent.name} has no tool that "
                f"sends outside the firm (graph: {path}).{seen} Its only way to move material is "
                f"{filing}, which the access rules already scope to the covered folder."
            )
    if obligation.kind == ControlKind.USE_RESTRICTION:
        notes.append(
            f"§{obligation.clause_id} use restriction: whether a task serves the Purpose is not "
            f"a condition a policy rule can check on a tool call, so no rule is proposed. The "
            f"access rules keep the material to the covered folder."
        )
    if obligation.kind == ControlKind.PERSONAL_DATA:
        notes.append(
            f"§{obligation.clause_id} personal data: a policy rule matches literal values "
            f"(equals, contains, starts_with) and cannot recognise a name, email address or "
            f"card number in free text, so no rule is proposed."
        )
    if obligation.kind == ControlKind.MARKED_MATERIAL:
        words = {*obligation.marked_terms, *bindings.codenames}
        if not words:
            notes.append(
                f"§{obligation.clause_id} marked material: the clause names no marking or "
                f"codename to match on, so no rule is proposed."
            )
        for agent in bindings.all_agents:
            tools = bindings.tools_for(agent)
            writers = [*tools.file, *tools.outbound, *(tools.graph.with_role("writes_internal") if tools.graph else [])]
            if not any(t.text_arg for t in writers):
                notes.append(
                    f"§{obligation.clause_id} marked material: {agent.name} has no filing or "
                    f"sending tool with a free-text argument to check for the markings."
                )
    if obligation.kind == ControlKind.PERMITTED_RECIPIENTS:
        named = [
            n
            for n in obligation.prohibited_recipients
            if n[:1].isupper() and not n.lower().startswith(("any", "other", "the "))
        ]
        if named and not obligation.permitted_recipients:
            notes.append(
                f"§{obligation.clause_id}: the clause forbids disclosure to {', '.join(named)}. "
                f"A policy rule sees the agent's own folders, not who a recipient is, so this "
                f"is covered only in so far as the access rules keep the material in "
                f"{', '.join(bindings.covered_folders)}."
            )
        for tool in bindings.tools_for(bindings.agent).access:
            if tool.document_arg is None:
                notes.append(
                    f"§{obligation.clause_id}: {tool.name}({', '.join(tool.args)}) on "
                    f"{bindings.agent.name} handles material but has no path argument, so it "
                    f"cannot be scoped to a folder."
                )
    return notes


TEMPLATES = {
    ControlKind.PERMITTED_RECIPIENTS: permitted_recipients,
    ControlKind.THIRD_PARTY_DISCLOSURE: third_party_disclosure,
    ControlKind.MARKED_MATERIAL: marked_material,
    ControlKind.SECURE_PROCESSING: secure_processing,
    # USE_RESTRICTION and PERSONAL_DATA have no policy-rule expression; see not_applicable().
}


def build_controls(obligation: Obligation, bindings: Bindings) -> list[Control]:
    template = TEMPLATES.get(obligation.kind)
    return template(obligation, bindings) if template else []


def describe(control: Control, bindings: Bindings) -> str:
    """The rule in plain English, for the verifier and the review screen.

    JEV reads literally, so it is shown what the rule does to whom, not the
    payload's operators and placeholder ids.
    """

    agent = bindings.agent.name
    folders = ", ".join(bindings.covered_folders)
    party = bindings.disclosing_party
    b = control.binding
    tool = b.get("tool")
    what = b.get("what")
    decision = control.payload.get("decision", "BLOCK")
    effect = {
        "BLOCK": "blocked from calling",
        "HALT": "stopped entirely (the whole run halts) if it calls",
        "REQUIRE_APPROVAL": "held for a person's approval in OpenBox before it may call",
        "CONSTRAIN": "made to run inside the sandbox when it calls",
        "ALLOW": "explicitly allowed to call",
    }.get(decision, "blocked from calling")
    if what in ("read", "file"):
        return (
            f"Agent {agent}, which is not a party to this NDA, is {effect} its "
            f"'{tool}' tool to {what} any document whose '{b.get('arg')}' lies under the "
            f"{party} folder {folders}. Documents in other folders are allowed."
        )
    if what == "send":
        return (
            f"Agent {agent} is {effect} its '{tool}' tool at all, because that tool "
            f"sends content outside the firm's systems and the NDA forbids disclosure to third "
            f"parties. Its other tools are unaffected."
        )
    return (
        f"Agent {agent} is {effect} its '{tool}' tool whenever the '{b.get('arg')}' "
        f"it passes contains the confidential marking '{b.get('term')}'. Calls without that "
        f"marking are allowed."
    )


def summarize(control: Control, agents: list[AgentBinding]) -> str:
    names = {a.id: a.name for a in agents}
    return f"policy rule on agent {names.get(control.agent_id, control.agent_id)}: {control.payload['rule_name']}"
