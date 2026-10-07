"""`nda-compile <nda> --bindings <yaml> [--out report.json]`"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .bindings import load_bindings
from .extract import extractor_from_env
from .governance import maybe_govern
from .graph import Services, build_graph, compile_nda
from .jev import judge_from_env
from .models import CompileReport
from .openbox_api import backend_from_env


def render(report: CompileReport, console: Console) -> None:
    names = {}
    table = Table(title=f"NDA → OpenBox controls · {report.matter}")
    table.add_column("§")
    table.add_column("Kind")
    table.add_column("Control")
    table.add_column("Agent")
    table.add_column("Judge")
    table.add_column("Status")
    for control in report.controls:
        table.add_row(
            control.clause_id,
            control.kind.value,
            control.payload.get("rule_name") or control.payload.get("name", ""),
            names.get(control.agent_id, control.agent_id),
            f"{control.verify_probability:.2f}" if control.verify_probability is not None else "-",
            control.status + (f" · {control.note}" if control.note else ""),
        )
    console.print(table)

    coverage = report.coverage
    console.print(
        f"[bold]Coverage[/bold]: {coverage['clauses']} clauses · "
        f"{coverage['enforced']} enforced at runtime · {coverage['judgement']} under judgement · "
        f"{coverage['not_enforceable']} outside runtime scope · {coverage['review']} for review"
    )
    for item in report.review:
        console.print(f"  [yellow]review[/yellow] {item}")
    timings = " · ".join(f"{t.stage} {t.ms:.0f}ms" for t in report.timings)
    console.print(f"[bold]Timing[/bold]: {timings} · total {report.total_ms:.0f}ms")
    console.print(f"[dim]models: {report.models}[/dim]")


async def main_async(args: argparse.Namespace) -> CompileReport:
    bindings = load_bindings(Path(args.bindings))
    services = Services(judge_from_env(), extractor_from_env(), backend_from_env(), bindings)
    graph = maybe_govern(build_graph(services))
    try:
        return await compile_nda(Path(args.nda), services, graph)
    finally:
        await services.backend.close()


def run() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Compile an NDA into OpenBox controls")
    parser.add_argument("nda")
    parser.add_argument("--bindings", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    render(report, Console())
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2))
        print(f"report written to {args.out}")
    else:
        print(json.dumps([c.payload for c in report.controls][:1], indent=2))
