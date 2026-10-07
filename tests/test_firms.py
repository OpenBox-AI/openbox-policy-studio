from pathlib import Path

from nda_compiler import conflicts
from nda_compiler.firms import profile_from_text_fake, write_bindings
from nda_compiler.models import Control, ControlKind
from nda_compiler.platform_context import ExistingRule

ROOT = Path(__file__).resolve().parents[1]


def test_profile_reads_the_contract():
    text = (ROOT.parent / "Downloads" / "sample-ndas" / "nda-atlas-grid-energy.md").read_text()
    p = profile_from_text_fake(text)
    assert p.disclosing_party == "Atlas Grid Energy plc"
    assert p.folder_references == ["0005/70001/"]
    assert "Project Halcyon" in p.codenames and "Strictly Private" in p.codenames
    assert "Confidential" not in p.codenames  # generic markings are dropped
    assert p.slug == "atlas-grid-energy"


def test_write_bindings_round_trips(tmp_path):
    p = profile_from_text_fake((ROOT.parent / "Downloads" / "sample-ndas" / "nda-atlas-grid-energy.md").read_text())
    path = write_bindings(
        tmp_path, p, agent_id="a1", agent_name="compliance agent", representative=False,
        covered_folders=["0005/70001/"], competitor_folders={"Northwind Storage Partners": "0005/20001/"},
    )
    from nda_compiler.bindings import load_bindings

    b = load_bindings(path, graphs_dir=tmp_path)
    assert b.matter == "atlas-grid-energy"
    assert b.other_agents[0].id == "a1" and not b.representatives
    assert b.competitor_folder("Northwind Storage Partners") == "0005/20001/"


def _control(tool, folder, decision, term=None):
    conds = [
        {"id": "tool", "left": {"kind": "field", "field": "activity_type"}, "operator": "equals", "right": {"kind": "literal", "value": tool}},
    ]
    if folder:
        conds.append({"id": "folder", "left": {"kind": "field", "field": "activity_input[0].x"}, "operator": "starts_with", "right": {"kind": "literal", "value": folder}})
    return Control(
        clause_id="3", kind=ControlKind.PERMITTED_RECIPIENTS, type="policy_rule", agent_id="a1",
        payload={"rule_name": "NDA atlas §3 x", "conditions": conds, "decision": decision, "priority": 90, "match_mode": "all"},
        binding={"term": term} if term else {},
    )


def _existing(name, tool, folder, decision):
    return ExistingRule(id="e", rule_name=name, decision=decision, is_active=True, conditions=_control(tool, folder, decision).payload["conditions"])


def test_conflicts_between_firms_are_flagged():
    existing = [
        _existing("NDA harrow §2 upload_document 0005/20001/ permitted", "upload_document", "0005/20001/", "ALLOW"),
        _existing("NDA harrow §2 read_document 0005/", "read_document", "0005/", "BLOCK"),
    ]
    halt = _control("upload_document", "0005/20001/", "HALT")
    overlap = _control("read_document", "0005/70001/", "BLOCK")
    generic = _control("write_briefing", None, "BLOCK", term="Confidential")
    own = _control("upload_document", "0005/70001/", "BLOCK")
    out = conflicts.check([halt, overlap, generic, own], existing, "atlas")
    assert out[0].status == "review" and "decides ALLOW" in out[0].note
    assert out[1].status == "review" and "overlaps 0005/" in out[1].note
    assert out[2].status == "review" and "generic" in out[2].note
    assert out[3].status == "draft"
