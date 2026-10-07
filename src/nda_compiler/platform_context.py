"""What the platform can express, held locally.

The judge and the templates should know what a policy rule is on OpenBox:
the input fields OPA actually sees (event fields, span fields per semantic
type), the operators and transforms a condition may use, the five decisions
and what each does, and which rules already exist on the agent. All of it is
read from the platform's own sources by scripts/sync_platform_context.py and
kept in platform/ as JSON, so a compile never depends on the frontend repo
being present and the models get the same context every time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DECISION_MEANINGS: dict[str, str] = {
    "ALLOW": "Let the call proceed and record that this rule matched.",
    "CONSTRAIN": "Let the call proceed with the rule's constraints applied to it.",
    "REQUIRE_APPROVAL": "Hold the call until a person approves it in OpenBox; expired "
    "approvals fail closed.",
    "BLOCK": "Refuse this call; the agent receives the rule's reason and carries on.",
    "HALT": "Refuse this call and stop the agent's whole run.",
}


class FieldDef(BaseModel):
    key: str
    label: str = ""
    type: str = "string"
    group: str = ""
    enum_values: list[str] = Field(default_factory=list)


class ExistingRule(BaseModel):
    id: str
    base_rule_id: str = ""
    rule_name: str
    decision: str
    priority: int = 0
    match_mode: str = "all"
    conditions: list[dict[str, Any]] = Field(default_factory=list)
    is_active: bool = True
    created_at: str = ""


class PlatformContext(BaseModel):
    synced_at: str = ""
    sources: dict[str, str] = Field(default_factory=dict)
    decisions: list[str] = Field(default_factory=lambda: list(DECISION_MEANINGS))
    operators: list[str] = Field(default_factory=list)
    transforms: list[str] = Field(default_factory=list)
    value_types: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    span_semantic_types: list[str] = Field(default_factory=list)
    fields: list[FieldDef] = Field(default_factory=list)
    # Behavior-rule match-field catalog per semantic type: the span fields core
    # forwards for each kind of span. Useful context even though only policy
    # rules are produced: it says what a span of each type carries.
    span_fields: dict[str, list[str]] = Field(default_factory=dict)
    existing_rules: dict[str, list[ExistingRule]] = Field(default_factory=dict)  # agent id ->

    def field(self, key: str) -> FieldDef | None:
        # activity_input[0].x and spans[_].attributes.x are prefix matches.
        for f in self.fields:
            if f.key == key:
                return f
        root = key.split("[")[0].split(".")[0]
        return next((f for f in self.fields if f.key == root), None)

    def rules_for(self, agent_id: str) -> list[ExistingRule]:
        return self.existing_rules.get(agent_id, [])

    def summary(self) -> dict[str, Any]:
        """The context as the judge sees it: compact, literal, no ids."""

        return {
            "policy_rule": "A set of conditions over one governed event (a tool call the "
            "agent makes, with its arguments, plus the spans recorded inside it); when they "
            "match, the decision applies to that call.",
            "decisions": {d: DECISION_MEANINGS.get(d, "") for d in self.decisions},
            "operators": self.operators,
            "input_fields": [f"{f.key} ({f.type})" for f in self.fields if not f.key.startswith("spans")],
            "span_fields_by_type": self.span_fields,
            "cannot_do": [
                "regular expressions or pattern recognition over free text",
                "remembering earlier calls (each event is judged on its own)",
                "reading the content a tool returns before deciding (ActivityStarted carries "
                "only the inputs)",
            ],
        }

    def existing_summary(self, agent_id: str) -> list[dict[str, Any]]:
        out = []
        for rule in self.rules_for(agent_id):
            conds = [
                f"{c.get('left', {}).get('field')} {c.get('operator')} "
                f"{c.get('right', {}).get('value', '')}".strip()
                for c in rule.conditions
            ]
            out.append(
                {
                    "rule": rule.rule_name,
                    "decision": rule.decision,
                    "when": f" {rule.match_mode} of ".join(["", ""]).join(conds) if conds else "",
                    "conditions": conds,
                    "active": rule.is_active,
                }
            )
        return out


def load_context(root: Path) -> PlatformContext:
    """platform/catalog.json plus every platform/agents/<id>.json; empty when absent."""

    catalog = root / "catalog.json"
    ctx = (
        PlatformContext.model_validate(json.loads(catalog.read_text(encoding="utf-8")))
        if catalog.exists()
        else PlatformContext()
    )
    for path in sorted((root / "agents").glob("*.json")) if (root / "agents").exists() else []:
        rows = json.loads(path.read_text(encoding="utf-8"))
        ctx.existing_rules[path.stem] = [ExistingRule.model_validate(r) for r in rows]
    return ctx
