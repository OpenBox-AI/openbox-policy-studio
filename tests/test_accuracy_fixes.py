"""Cases from the accuracy evaluation: each was a wrong or missing rule before."""

from pathlib import Path

import httpx
import pytest

from nda_compiler.bindings import load_bindings
from nda_compiler.extract import FakeExtractor, outsider_names
from nda_compiler.graph import Services, compile_nda
from nda_compiler.jev import FakeJudge, decision_from_wording
from nda_compiler.models import Classification, Control, ControlKind
from nda_compiler.openbox_api import HttpBackend, RecordingBackend

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "bindings" / "coca-cola.yaml"


@pytest.mark.parametrize(
    "text, decision",
    [
        ("only with our prior written approval", "REQUIRE_APPROVAL"),
        ("without our written agreement", "REQUIRE_APPROVAL"),
        ("except with the Trust's prior written approval", "REQUIRE_APPROVAL"),
        ("without the prior written consent of the Disclosing Party", "REQUIRE_APPROVAL"),
        ("shall constitute a material breach", "HALT"),
        ("shall not disclose Confidential Information", None),
        ("approved in writing by the Bank", None),
    ],
)
def test_consent_wording(text, decision):
    assert decision_from_wording(text) == decision


def test_outsider_names_are_organisations_the_nda_does_not_define():
    nda = '"Representatives" means ... "Clean Team Members" means ...'
    names = [
        "Northwind Storage Partners",
        "Kestrel Infrastructure Fund",
        "Representatives",
        "Clean Team Members",
        "Brightwater's officers, employees and outside counsel",
        "any other person",
    ]
    assert outsider_names(names, nda) == ["Northwind Storage Partners", "Kestrel Infrastructure Fund"]


NDA_TEXT = """MUTUAL NON-DISCLOSURE AGREEMENT

1. Definitions. "Confidential Information" means the documents filed under matter 0001/10001. "Representatives" means the Receiving Party's employees who need to know.

2. Competitors. The Receiving Party shall not disclose Confidential Information to Northwind Storage Partners. Any such disclosure constitutes a material breach.

3. Transmission.
3.1 The Receiving Party shall not email or upload Confidential Information outside its systems.
3.2 Any transmission in breach of clause 3.1 shall constitute a material breach of this Agreement.
"""


class Judge(FakeJudge):
    KINDS = {
        "1": ControlKind.DEFINITION,
        "2": ControlKind.PERMITTED_RECIPIENTS,
        "3.1": ControlKind.THIRD_PARTY_DISCLOSURE,
        "3.2": ControlKind.NOT_ENFORCEABLE,
    }

    async def classify(self, nda_text, clauses):
        return [Classification(clause_id=c.id, kind=self.KINDS[c.id], confidence=1.0) for c in clauses]


class BiddersAsPermitted(FakeExtractor):
    async def extract(self, clause, kind, definitions):
        obligation = await super().extract(clause, kind, definitions)
        if clause.id == "2":
            # What gpt-5-mini returned for Atlas §3 in 4 of 5 runs.
            return obligation.model_copy(
                update={"permitted_recipients": ["Northwind Storage Partners"], "prohibited_recipients": []}
            )
        return obligation


async def test_named_bidders_never_become_folder_rules_and_breach_subclause_raises_halt(tmp_path):
    nda = tmp_path / "nda.md"
    nda.write_text(NDA_TEXT)
    services = Services(Judge(), BiddersAsPermitted(), RecordingBackend(), load_bindings(BINDINGS), 0.8)
    report = await compile_nda(nda, services)
    # §2 only names a bidder: no folder rule, so no HALT on the counterparty's folder.
    assert not [c for c in report.controls if c.clause_id == "2"]
    bidder = next(o for o in report.obligations if o.clause_id == "2")
    assert bidder.permitted_recipients == []
    assert "Northwind Storage Partners" in bidder.prohibited_recipients
    # §3.2 says breaching §3.1 is a material breach: §3.1's outbound rules HALT.
    sends = [c for c in report.controls if c.clause_id == "3.1"]
    assert sends and all(c.payload["decision"] == "HALT" for c in sends)


def _control(name: str) -> Control:
    return Control(
        clause_id="2",
        kind=ControlKind.PERMITTED_RECIPIENTS,
        type="policy_rule",
        agent_id="agent-1",
        payload={"rule_name": name, "decision": "BLOCK", "conditions": [], "reason": "r"},
    )


async def test_existing_rule_found_beyond_the_first_page():
    """With more rules than one page, the rule must still be found (not re-created)."""

    rules = [{"id": f"r{i}", "rule_name": f"rule {i}", "is_current_version": True,
              "is_active": True, "decision": "BLOCK", "conditions": []} for i in range(150)]
    created: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            page = int(request.url.params.get("page", 0))
            per = int(request.url.params.get("perPage", 10))
            return httpx.Response(200, json={"status": 200, "data": {"data": rules[page * per:(page + 1) * per], "total": len(rules)}})
        created.append(request.url.path)
        return httpx.Response(201, json={"status": 201, "data": {"id": "new"}})

    backend = HttpBackend("http://openbox.test", "key")
    backend._client = httpx.AsyncClient(base_url="http://openbox.test", transport=httpx.MockTransport(handler))
    found = await backend._current_by_name("/agent/agent-1/policy-rule", "rule 142")
    await backend.close()
    assert found is not None and found["id"] == "r142"


async def test_a_timeout_fails_one_rule_not_the_apply():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    backend = HttpBackend("http://openbox.test", "key")
    backend._client = httpx.AsyncClient(base_url="http://openbox.test", transport=httpx.MockTransport(handler))
    result = await backend.apply(_control("rule"))
    await backend.close()
    assert result.status == "failed" and "no response" in result.note


def test_repeated_clause_numbers_get_unique_ids():
    from nda_compiler import pdf

    text = "1. Definitions. X means y.\n2. Confidentiality. Keep it secret.\nAMENDMENT\n1. Scope. Amends the agreement.\n2. Extra. Do not tell Acme."
    clauses = pdf.split_clauses(text)
    assert [c.id for c in clauses] == ["1", "2", "1~2", "2~2"]
    assert clauses[1].text.startswith("Keep it secret")


async def test_decisions_judge_sends_documented_shape_and_handles_refusals():
    import json as _json

    from nda_compiler.decisions import DecisionsJudge
    from nda_compiler.models import Clause

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content)
        seen.update(body)
        answers = [
            {"type": "choice", "name": body["questions"][0]["name"], "choice": "permitted_recipients",
             "confidence": 0.91, "probabilities": [{"value": "permitted_recipients", "probability": 0.91}]},
            {"type": "refusal", "name": body["questions"][1]["name"]},
        ]
        return httpx.Response(200, json={"answers": answers})

    judge = DecisionsJudge("key")
    judge._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    clauses = [Clause(id="2", text="Disclose only to Representatives."), Clause(id="3", text="Governed by Delaware law.")]
    out = await judge.classify("the NDA", clauses)
    await judge.close()
    assert seen["model"] == "gpt-6-luna" and seen["input"] == "the NDA"
    assert [q["type"] for q in seen["questions"]] == ["choice", "choice"]
    assert {"value", "description"} <= set(seen["questions"][0]["choices"][0])
    assert out[0].kind == ControlKind.PERMITTED_RECIPIENTS and out[0].confidence == 0.91
    assert out[1].kind == ControlKind.BOILERPLATE and out[1].confidence == 0.0


@pytest.mark.parametrize(
    "env, extractor, judge",
    [
        ({}, "FakeExtractor", "FakeJudge"),
        ({"OPENAI_API_KEY": "k"}, "OpenAIExtractor", "DecisionsJudge"),
        ({"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k"}, "AnthropicExtractor", "DecisionsJudge"),
        ({"ANTHROPIC_API_KEY": "k", "TYPESAFE_API_KEY": "k"}, "AnthropicExtractor", "TypeSafeJudge"),
        ({"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k", "TYPESAFE_API_KEY": "k", "JUDGE_PROVIDER": "typesafe"},
         "AnthropicExtractor", "TypeSafeJudge"),
        ({"ANTHROPIC_API_KEY": "k", "OPENAI_API_KEY": "k", "EXTRACT_PROVIDER": "openai"}, "OpenAIExtractor", "DecisionsJudge"),
    ],
)
def test_providers_follow_the_keys(monkeypatch, env, extractor, judge):
    from nda_compiler.extract import extractor_from_env
    from nda_compiler.jev import judge_from_env

    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TYPESAFE_API_KEY", "EXTRACT_PROVIDER", "JUDGE_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert type(extractor_from_env()).__name__ == extractor
    assert type(judge_from_env()).__name__ == judge
