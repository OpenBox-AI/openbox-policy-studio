from pathlib import Path

import pytest

from nda_compiler import pdf
from nda_compiler.bindings import load_bindings
from nda_compiler.extract import FakeExtractor
from nda_compiler.graph import Services, compile_nda
from nda_compiler.jev import FakeJudge
from nda_compiler.models import ControlKind
from nda_compiler.openbox_api import RecordingBackend

ROOT = Path(__file__).resolve().parents[1]
NDA = ROOT / "fixtures" / "coca_cola_nda.md"
BINDINGS = ROOT / "bindings" / "coca-cola.yaml"


def test_split_keeps_untitled_subclauses():
    text = (
        "Section 2. Access.\n"
        "2.1 The Adviser shall ensure access is limited.\n"
        "2.2 The Adviser shall not permit contractors."
    )
    clauses = pdf.split_clauses(text)
    assert [c.id for c in clauses] == ["2.1", "2.2"]
    assert clauses[0].heading == ""
    assert clauses[0].text.startswith("The Adviser shall ensure")


def test_split_clauses_finds_numbered_sections():
    _, clauses = pdf.load(NDA)
    assert [c.id for c in clauses] == [str(i) for i in range(1, 10)]
    assert clauses[1].heading == "Permitted Disclosure"
    assert "6" in clauses[3].references
    assert '"Purpose" means' in pdf.definitions_text(clauses)


@pytest.fixture
def services():
    return Services(FakeJudge(), FakeExtractor(), RecordingBackend(), load_bindings(BINDINGS), 0.8)


async def test_compile_emits_grounded_controls(services):
    report = await compile_nda(NDA, services)
    kinds = {c.clause_id: c.kind for c in report.classifications}
    assert kinds["2"] == ControlKind.PERMITTED_RECIPIENTS
    assert kinds["4"] == ControlKind.THIRD_PARTY_DISCLOSURE
    assert kinds["7"] == ControlKind.NOT_ENFORCEABLE
    assert report.review == []
    assert report.coverage["not_enforceable"] == 3

    access = [c for c in report.controls if c.kind == ControlKind.PERMITTED_RECIPIENTS]
    # One agent x (read + file) x one covered folder, blocked.
    policy_rules = [c for c in access if c.payload["decision"] == "BLOCK"]
    assert len(policy_rules) == 2
    # §4 covers every outbound tool; this firm has no exported graph, so the
    # bindings' two outbound tools apply, with the clause's consent wording.
    sends = [c for c in report.controls if c.kind == ControlKind.THIRD_PARTY_DISCLOSURE]
    assert len(sends) == 2 and all(c.payload["decision"] == "REQUIRE_APPROVAL" for c in sends)
    assert all(c.binding["what"] == "send" for c in sends)
    rule = policy_rules[0].payload
    assert rule["decision"] == "BLOCK"
    assert rule["is_active"] is False
    assert rule["conditions"][1]["right"]["value"] == "0001/10001/"
    fields = {c.payload["conditions"][1]["left"]["field"] for c in policy_rules}
    assert fields == {
        "activity_input[0].document_id",
        "activity_input[0].destination_document_id",
    }
    assert "§2" in rule["reason"]
    assert all(c.status == "active" for c in report.controls)

    # Only platform policy types are ever proposed; the Purpose clause is
    # reported as not applicable rather than turned into something OpenBox
    # cannot store.
    assert {c.type for c in report.controls} == {"policy_rule"}
    assert any("use restriction" in n for n in report.not_applicable)

    calls = services.backend.calls
    assert any(path.endswith("/evaluate") for _, path, _ in calls)


async def test_ungrounded_extraction_goes_to_review(services):
    class LyingExtractor(FakeExtractor):
        async def extract(self, clause, kind, definitions):
            obligation = await super().extract(clause, kind, definitions)
            return obligation.model_copy(update={"prohibited_recipients": ["Dr Pepper"]})

    services.extractor = LyingExtractor()
    report = await compile_nda(NDA, services)
    assert any("Dr Pepper" in item for item in report.review)
    assert not [c for c in report.controls if c.type == "policy_rule"]
