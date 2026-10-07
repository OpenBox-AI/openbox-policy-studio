"""Create the barrier-demo agents in an OpenBox org and bind them to a matter.

    uv run python scripts/bootstrap_agents.py bindings/coca-cola.yaml

Uses OPENBOX_BACKEND_URL + OPENBOX_ORG_API_KEY from .env. Agents that already
exist in the bindings (a real UUID rather than a placeholder) are left alone,
so the script is safe to re-run. Runtime keys returned at create time are
written to .env.agents (gitignored) for the barrier demo to use.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def main(bindings_path: Path) -> None:
    load_dotenv()
    base = os.environ.get("OPENBOX_BACKEND_URL", "http://localhost:3000").rstrip("/")
    key = os.environ["OPENBOX_ORG_API_KEY"]
    client = httpx.Client(base_url=base, headers={"X-API-Key": key}, timeout=30)

    data = yaml.safe_load(bindings_path.read_text())
    keys_out: list[str] = []
    for group in ("representatives", "other_agents"):
        for agent in data.get(group, []):
            if _UUID.match(agent["id"]):
                print(f"keep   {agent['name']} {agent['id']}")
                continue
            response = client.post(
                "/agent/create",
                json={
                    "agent_name": agent["name"],
                    "agent_type": "langgraph",
                    "description": f"openbox-barrier-demo agent ({agent.get('role', 'agent')})",
                    "attestation_mode": "kms",
                    "tags": ["barrier-demo", data["matter"]],
                },
            )
            if response.status_code >= 300:
                detail = response.text[:300]
                sys.exit(f"create {agent['name']} failed: {response.status_code} {detail}")
            body = response.json()
            created = body.get("agent", body)
            agent["id"] = created.get("id") or created.get("agent_id")
            runtime_key = body.get("api_key") or created.get("api_key") or ""
            slug = agent["name"].replace("ClientIntelligenceAgent", "").upper()
            if runtime_key:
                keys_out.append(f"{slug}_OPENBOX_API_KEY={runtime_key}")
            print(f"create {agent['name']} {agent['id']}")

    bindings_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    if keys_out:
        Path(".env.agents").write_text("\n".join(keys_out) + "\n")
        print(f"runtime keys written to .env.agents ({len(keys_out)})")
    print(f"bindings updated: {bindings_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "bindings/coca-cola.yaml"))
