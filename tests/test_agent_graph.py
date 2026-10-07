from pathlib import Path

from nda_compiler.agent_graph import AgentGraph, GraphEdge, ToolSpec, fake_roles, observe, with_roles
from nda_compiler.bindings import load_bindings
from nda_compiler.extract import FakeExtractor
from nda_compiler.graph import Services, compile_nda
from nda_compiler.jev import FakeJudge
from nda_compiler.openbox_api import RecordingBackend

ROOT = Path(__file__).resolve().parents[1]
NDA = ROOT / "fixtures" / "coca_cola_nda.md"
AGENT = "16e4c8e5-0921-4946-ae5b-96fe1966e57d"


def _graph() -> AgentGraph:
    return AgentGraph(
        agent_id=AGENT,
        nodes=["__start__", "search", "read_document", "upload_document", "__end__"],
        edges=[
            GraphEdge(source="__start__", target="search"),
            GraphEdge(source="search", target="read_document", conditional=True),
            GraphEdge(source="read_document", target="upload_document"),
            GraphEdge(source="upload_document", target="__end__"),
        ],
        tools=[
            ToolSpec(name="search_documents", description="Search metadata", args={"query": "string"}, node="search"),
            ToolSpec(name="read_document", description="Read one document", args={"document_id": "string"}, node="read_document"),
            ToolSpec(name="upload_document", description="File the report", args={"destination_document_id": "string"}, node="upload_document"),
        ],
    )


def test_exported_graph_matches_agent_code():
    graph = load_bindings(ROOT / "bindings" / "trial.yaml").graphs[AGENT]
    assert {t.name for t in graph.tools} == {
        "search_documents",
        "read_document",
        "write_briefing",
        "upload_document",
    }
    assert graph.tool("read_document").args == {"document_id": "string"}
    assert graph.reaches("read_document", "upload_document")
    assert not graph.reaches("upload_document", "read_document")


def test_observed_events_fix_input_shape_and_args():
    events = [
        {
            "event_type": "ActivityStarted",
            "activity_type": "upload_document",
            "input": [{"destination_document_id": "0001/30002/x.md"}, {"__openbox": {}}],
        },
        {"event_type": "ActivityCompleted", "activity_type": "upload_document", "input": None},
        {"event_type": "ActivityStarted", "activity_type": "send_email", "input": {"to": "a@b"}},
    ]
    graph = observe(events, _graph())
    upload = graph.tool("upload_document")
    assert upload.observed == 1
    assert upload.input_shape == "list"
    assert upload.input_path("destination_document_id") == "activity_input[0].destination_document_id"
    mail = graph.tool("send_email")
    assert mail is not None and mail.input_shape == "dict"
    assert mail.input_path("to") == "activity_input.to"
    assert "observed" in graph.sources


def test_roles_pick_document_argument():
    graph = with_roles(_graph(), fake_roles(_graph().tools))
    assert graph.tool("read_document").role == "reads_material"
    assert graph.tool("upload_document").role == "files_to_store"
    assert graph.tool("search_documents").role == "searches_index"
    assert graph.tool("search_documents").document_arg is None
    assert graph.tool("upload_document").document_arg == "destination_document_id"


async def test_templates_bind_to_graph_tools_and_report_gaps():
    bindings = load_bindings(ROOT / "bindings" / "trial.yaml")
    services = Services(FakeJudge(), FakeExtractor(), RecordingBackend(), bindings, 0.8)
    report = await compile_nda(NDA, services)

    rules = [c for c in report.controls if c.type == "policy_rule"]
    assert {c.binding["tool"] for c in rules} == {"read_document", "upload_document"}
    assert {c.binding["arg"] for c in rules} == {"document_id", "destination_document_id"}
    assert all(c.binding["node"] for c in rules)
    # The agent has no outbound tool, so no sequence rule is invented for it.
    assert not [c for c in report.controls if c.type == "behavior_rule"]
    assert any("no tool that sends outside" in n for n in report.not_applicable)
    assert any("upload_document" in n for n in report.not_applicable)
