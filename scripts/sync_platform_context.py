"""Refresh platform/ from the platform's own sources.

  catalog.json      policy-rule field catalog, operators, transforms, value
                    types, decisions, OPA event types, span semantic types
                    (openbox-fe policy-field-catalog.ts / policy-rule-conditions.ts)
                    and the per-semantic-type span field catalog
                    (openbox-backend behavior-rule-match-field-catalog.ts)
  agents/<id>.json  the agent's current policy rules (GET /agent/:id/policy-rule)

    uv run python scripts/sync_platform_context.py \
        --fe ~/openbox-core/openbox-fe --backend ~/openbox-core/openbox-backend \
        --agent 16e4c8e5-0921-4946-ae5b-96fe1966e57d
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parents[1]


def _const_list(src: str, name: str) -> list[str]:
    m = re.search(rf"{name}[^=]*=\s*\[(.*?)\]\s*as const", src, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def _fields(src: str) -> list[dict]:
    out = []
    for m in re.finditer(r"\{\s*key:\s*\"([^\"]+)\",\s*label:\s*\"([^\"]+)\",\s*type:\s*\"(\w+)\",\s*group:\s*\"([^\"]+)\"([^}]*)\}", src):
        key, label, typ, group, rest = m.groups()
        enum_name = re.search(r"enumValues:\s*(\w+)", rest)
        out.append(
            {
                "key": key,
                "label": label,
                "type": typ,
                "group": group,
                "enum_values": _const_list(src, enum_name.group(1)) if enum_name else [],
            }
        )
    return out


def _span_fields(src: str) -> dict[str, list[str]]:
    lists: dict[str, list[str]] = {}
    for m in re.finditer(r"const (\w+_FIELDS): MatchFieldDef\[\] = \[(.*?)\];", src, re.S):
        lists[m.group(1)] = re.findall(r"f\('([^']+)'", m.group(2))
    common = lists.get("COMMON_FIELDS", [])
    out: dict[str, list[str]] = {}
    for m in re.finditer(r"\[BehaviorRuleTrigger\.(\w+)\]:\s*(\w+_FIELDS)", src):
        out[m.group(1).lower()] = [*common, *lists.get(m.group(2), [])]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fe", type=Path, default=Path.home() / "openbox-core/openbox-fe")
    parser.add_argument("--backend", type=Path, default=Path.home() / "openbox-core/openbox-backend")
    parser.add_argument("--agent", action="append", default=[], help="agent id to snapshot (repeatable)")
    parser.add_argument("--out", type=Path, default=HERE / "platform")
    args = parser.parse_args()

    policies = args.fe / "src/components/pages/agent/components/authorize/policies/utils"
    catalog_src = (policies / "policy-field-catalog.ts").read_text(encoding="utf-8")
    conditions_src = (policies / "policy-rule-conditions.ts").read_text(encoding="utf-8")
    behavior_src = (
        args.backend / "src/modules/agent/utils/behavior-rule-match-field-catalog.ts"
    ).read_text(encoding="utf-8")

    catalog = {
        "synced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {
            "fields": "openbox-fe policy-field-catalog.ts",
            "conditions": "openbox-fe policy-rule-conditions.ts",
            "span_fields": "openbox-backend behavior-rule-match-field-catalog.ts",
        },
        "decisions": _const_list(conditions_src, "DECISIONS"),
        "operators": _const_list(conditions_src, "OPERATORS"),
        "transforms": _const_list(conditions_src, "TRANSFORMS"),
        "value_types": _const_list(conditions_src, "VALUE_TYPES"),
        "event_types": _const_list(catalog_src, "OPA_EVENT_TYPE_VALUES"),
        "span_semantic_types": _const_list(catalog_src, "SPAN_SEMANTIC_TYPE_VALUES"),
        "fields": _fields(catalog_src),
        "span_fields": _span_fields(behavior_src),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(
        f"catalog: {len(catalog['fields'])} fields, {len(catalog['operators'])} operators, "
        f"{len(catalog['decisions'])} decisions, {len(catalog['span_fields'])} span types"
    )

    if args.agent:
        load_dotenv(HERE / ".env")
        base = os.environ.get("OPENBOX_BACKEND_URL", "http://localhost:3000").rstrip("/")
        key = os.environ["OPENBOX_ORG_API_KEY"]
        (args.out / "agents").mkdir(exist_ok=True)
        for agent in args.agent:
            r = httpx.get(
                f"{base}/agent/{agent}/policy-rule",
                params={"limit": 200},
                headers={"X-API-Key": key},
                timeout=15,
            )
            r.raise_for_status()
            page = r.json()["data"]
            rows = page.get("data", page) if isinstance(page, dict) else page
            current = [
                {
                    "id": x["id"],
                    "base_rule_id": x.get("base_rule_id", ""),
                    "rule_name": x["rule_name"],
                    "decision": x["decision"],
                    "priority": x.get("priority", 0),
                    "match_mode": x.get("match_mode", "all"),
                    "conditions": x.get("conditions", []),
                    "is_active": x.get("is_active", False),
                    "created_at": x.get("created_at", ""),
                }
                for x in rows
                if x.get("is_current_version", True)
            ]
            (args.out / "agents" / f"{agent}.json").write_text(
                json.dumps(current, indent=2) + "\n", encoding="utf-8"
            )
            print(f"agent {agent[:8]}: {len(current)} current rules")


if __name__ == "__main__":
    main()
