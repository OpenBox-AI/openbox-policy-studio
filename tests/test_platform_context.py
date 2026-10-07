from pathlib import Path

from nda_compiler.jev import fake_decisions
from nda_compiler.models import ControlKind, Obligation
from nda_compiler.openbox_api import _same_rule
from nda_compiler.platform_context import load_context

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_is_synced_from_the_platform_sources():
    ctx = load_context(ROOT / "platform")
    assert ctx.decisions == ["ALLOW", "CONSTRAIN", "REQUIRE_APPROVAL", "BLOCK", "HALT"]
    assert {"equals", "contains", "starts_with", "exists"} <= set(ctx.operators)
    assert ctx.field("activity_type") is not None
    assert ctx.field("activity_input[0].document_id").key == "activity_input"
    assert ctx.field("spans[_].http_url") is not None
    assert "file_path" in ctx.span_fields["file_read"]
    assert "no_such_field" not in {f.key for f in ctx.fields}
    summary = ctx.summary()
    assert "BLOCK" in summary["decisions"] and summary["cannot_do"]


def _obligation(quote: str) -> Obligation:
    return Obligation(
        clause_id="2",
        kind=ControlKind.PERMITTED_RECIPIENTS,
        bound_party="Receiving Party",
        action="disclose",
        subject="Confidential Information",
        source_quote=quote,
    )


def test_decision_follows_the_clause_wording():
    assert fake_decisions([_obligation("shall not disclose to any third party")])["2"] == "BLOCK"
    assert (
        fake_decisions([_obligation("may disclose only with the prior written consent of X")])["2"]
        == "REQUIRE_APPROVAL"
    )
    assert fake_decisions([_obligation("any such disclosure is a material breach")])["2"] == "HALT"


def test_same_rule_ignores_condition_ids_and_order():
    existing = {
        "decision": "BLOCK",
        "match_mode": "all",
        "priority": 90,
        "conditions": [
            {"id": "b", "left": {"field": "x"}, "operator": "equals", "right": {"value": "1"}},
            {"id": "a", "left": {"field": "y"}, "operator": "starts_with", "right": {"value": "p/"}},
        ],
    }
    payload = {
        "decision": "BLOCK",
        "match_mode": "all",
        "priority": 90,
        "conditions": [
            {"id": "folder", "left": {"field": "y"}, "operator": "starts_with", "right": {"value": "p/"}},
            {"id": "tool", "left": {"field": "x"}, "operator": "equals", "right": {"value": "1"}},
        ],
    }
    assert _same_rule(existing, payload)
    assert not _same_rule({**existing, "decision": "HALT"}, payload)
