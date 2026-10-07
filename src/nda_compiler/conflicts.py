"""Cross-firm conflicts, found before anything is applied.

Several firms' policies live on one agent, told apart by the folder prefix
in each rule. That holds as long as nobody's folders overlap and nobody's
rule contradicts another firm's on the same conditions. Both are checkable
against the rules already on the agent, which the platform snapshot holds.

A conflict never blocks a proposal; it puts the row in review with a note
naming the other rule, and the officer decides.
"""

from __future__ import annotations

from typing import Any

from .models import Control
from .platform_context import ExistingRule

# Markings so common that matching them in free text would catch every
# firm's output, not this firm's.
GENERIC_TERMS = {"confidential", "private", "restricted", "internal", "secret", "strictly private", "confidential information"}


def _folder_condition(conditions: list[dict[str, Any]]) -> str | None:
    for c in conditions:
        if c.get("operator") == "starts_with":
            value = (c.get("right") or {}).get("value")
            if isinstance(value, str):
                return value
    return None


def _tool_condition(conditions: list[dict[str, Any]]) -> str | None:
    for c in conditions:
        if c.get("operator") == "equals" and (c.get("left") or {}).get("field") == "activity_type":
            return (c.get("right") or {}).get("value")
    return None


def _same_firm(rule_name: str, firm: str) -> bool:
    return rule_name.startswith(f"NDA {firm} ")


def check(controls: list[Control], existing: list[ExistingRule], firm: str) -> list[Control]:
    """Annotate controls that collide with another firm's rules on this agent."""

    others = [r for r in existing if r.is_active and not _same_firm(r.rule_name, firm)]
    out = []
    for control in controls:
        notes: list[str] = []
        tool = _tool_condition(control.payload["conditions"])
        folder = _folder_condition(control.payload["conditions"])
        decision = control.payload["decision"]
        if folder:
            for rule in others:
                other_folder = _folder_condition(rule.conditions)
                other_tool = _tool_condition(rule.conditions)
                if not other_folder or other_tool != tool:
                    continue
                if other_folder == folder and rule.decision != decision:
                    notes.append(
                        f"another firm's rule '{rule.rule_name}' decides {rule.decision} on the "
                        f"same tool and folder; the higher priority wins, so one firm's clause "
                        f"is defeated"
                    )
                elif other_folder == folder:
                    notes.append(
                        f"another firm's rule '{rule.rule_name}' already governs this folder; "
                        f"two firms are bound to one folder, which is usually a wrong binding"
                    )
                elif other_folder != folder and (
                    other_folder.startswith(folder) or folder.startswith(other_folder)
                ):
                    notes.append(
                        f"folder {folder} overlaps {other_folder} used by another firm's rule "
                        f"'{rule.rule_name}'; calls under the shorter prefix match both"
                    )
        term = control.binding.get("term")
        if term and term.strip().lower() in GENERIC_TERMS:
            notes.append(
                f"the marking '{term}' is generic; matching it in the agent's text would "
                f"catch every firm's output, not this firm's"
            )
        if notes:
            control = control.model_copy(
                update={"status": "review", "note": (control.note + "; " if control.note else "") + "conflict: " + " · ".join(notes)}
            )
        out.append(control)
    return out
