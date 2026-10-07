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

    policy_rules = [c for c in report.controls if c.type == "policy_rule"]
    # Two non-representatives x (read + file) x one covered folder.
    assert len(policy_rules) == 4
    rule = policy_rules[0].payload
    assert rule["decision"] == "BLOCK"
    assert rule["is_active"] is False
    assert rule["conditions"][1]["right"]["value"] == "0001/10001/"
    assert "§2" in rule["reason"]
    assert all(c.status == "active" for c in report.controls)

    judgement = [c for c in report.controls if c.type == "judgement"]
    assert judgement
    assert "integration risk" in judgement[0].payload["question"]["instructions"]["purpose"]

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
