"""Check, or register, the agent a firm's binding points at.

    uv run python scripts/bootstrap_agents.py bindings/northwind.yaml

Uses OPENBOX_BACKEND_URL and OPENBOX_ORG_API_KEY from .env.

- If the binding's agent.id is a real agent ID, checks that the agent exists
  in the organisation the key belongs to, and says so either way.
- If agent.id is missing or still the template's placeholder
  (00000000-0000-0000-0000-000000000000), registers a new agent in OpenBox
  under agent.name, writes its ID into the binding (the rest of the file,
  comments included, is kept), and appends the agent's own runtime key and
  workload private key to .env.agents (gitignored), for whoever runs the
  agent to use.

Safe to re-run: an agent that already exists is never created again.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import sys
from pathlib import Path

import httpx
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from dotenv import load_dotenv

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_PLACEHOLDER = "00000000-0000-0000-0000-000000000000"


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def workload_keypair() -> tuple[str, dict[str, str]]:
    """One RSA workload key per agent: PKCS8 PEM for the agent, public JWK for OpenBox."""

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    numbers = key.public_key().public_numbers()
    jwk = {
        "kid": hashlib.sha256(numbers.n.to_bytes(256, "big")).hexdigest()[:32],
        "kty": "RSA",
        "alg": "RS256",
        "use": "sig",
        "n": _b64url(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")),
        "e": _b64url(numbers.e.to_bytes((numbers.e.bit_length() + 7) // 8, "big")),
    }
    return pem, jwk


def _unwrap(body: dict) -> dict:
    """The backend answers {status, data: ...}; older builds used {result: ...}."""

    return body.get("data") or body.get("result") or body


def _exists(client: httpx.Client, agent_id: str) -> bool:
    return client.get(f"/agent/{agent_id}").status_code == 200


def _register(client: httpx.Client, name: str, firm: str) -> tuple[str, str, str]:
    pem, jwk = workload_keypair()
    response = client.post(
        "/agent/create",
        json={
            "agent_name": name,
            "agent_type": "langgraph",
            "description": f"Agent governed by the Policy Studio rules for {firm}",
            "attestation_mode": "kms",
            # Organisations on managed Keycloak identity only admit
            # workload-authenticated agents.
            "identity_verification": {
                "method": "keycloak_workload",
                "mode": "generate",
                "source_type": "openbox",
                "public_jwk": jwk,
            },
            "icon": "https://api.iconify.design/carbon/document.svg",
            # "medium" risk profile, the same values the OpenBox CLI uses.
            "aivss_config": {
                "base_security": {
                    "attack_vector": 1,
                    "attack_complexity": 1,
                    "privileges_required": 2,
                    "user_interaction": 1,
                    "scope": 1,
                },
                "ai_specific": {
                    "model_robustness": 2,
                    "data_sensitivity": 2,
                    "ethical_impact": 2,
                    "decision_criticality": 2,
                    "adaptability": 2,
                },
                "impact": {
                    "confidentiality_impact": 2,
                    "integrity_impact": 2,
                    "availability_impact": 2,
                    "safety_impact": 1,
                },
            },
            "tags": ["policy-studio", firm],
        },
    )
    if response.status_code >= 300:
        sys.exit(f"registering {name} failed: {response.status_code} {response.text[:300]}")
    result = _unwrap(response.json())
    created = result.get("agent", result)
    agent_id = created.get("id") or created.get("agent_id")
    if not agent_id:
        sys.exit(f"registering {name}: no agent ID in the response")
    return agent_id, result.get("token") or result.get("api_key") or "", pem


def _write_id(path: Path, agent_id: str) -> None:
    """Set agent.id in place, keeping the file's comments and layout."""

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    in_agent = False
    for i, line in enumerate(lines):
        if re.match(r"^agent:\s*(#.*)?$", line):
            in_agent = True
            continue
        if in_agent and re.match(r"^\S", line):
            break
        if in_agent and re.match(r"^\s+id:", line):
            indent = re.match(r"^(\s+)", line).group(1)
            comment = re.search(r"\s+#.*$", line.rstrip("\n"))
            lines[i] = f"{indent}id: {agent_id}{comment.group(0) if comment else ''}\n"
            path.write_text("".join(lines), encoding="utf-8")
            return
    # No id line under agent: (or no agent block): add one.
    data = yaml.safe_load(text) or {}
    data.setdefault("agent", {})["id"] = agent_id
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main(bindings_path: Path) -> None:
    load_dotenv()
    base = os.environ.get("OPENBOX_BACKEND_URL", "http://localhost:3000").rstrip("/")
    key = os.environ.get("OPENBOX_ORG_API_KEY", "").strip()
    if not key:
        sys.exit("OPENBOX_ORG_API_KEY is not set in .env")
    client = httpx.Client(base_url=base, headers={"X-API-Key": key}, timeout=30)

    data = yaml.safe_load(bindings_path.read_text(encoding="utf-8")) or {}
    firm = data.get("firm") or bindings_path.stem
    agent = data.get("agent") or {}
    agent_id = str(agent.get("id") or "").strip()
    name = str(agent.get("name") or f"{firm} agent").strip()

    if _UUID.match(agent_id) and agent_id != _PLACEHOLDER:
        if _exists(client, agent_id):
            print(f"ok: {name} ({agent_id}) exists in this organisation on {base}")
            return
        sys.exit(
            f"not found: agent {agent_id} is not in the organisation of OPENBOX_ORG_API_KEY on {base}.\n"
            "Check the ID in the OpenBox dashboard (Agents, click the agent, the code after /agents/),\n"
            "or set it to the placeholder to register a new agent."
        )

    agent_id, runtime_key, pem = _register(client, name, firm)
    _write_id(bindings_path, agent_id)
    prefix = re.sub(r"[^A-Z0-9]+", "_", firm.upper()).strip("_")
    escaped = pem.replace("\n", "\\n")
    entries = [f"{prefix}_OPENBOX_AGENT_" + f"NAME={name}", f"{prefix}_OPENBOX_AGENT_" + f"ID={agent_id}"]
    if runtime_key:
        entries.append(f"{prefix}_OPENBOX_" + "API_" + "KEY=" + runtime_key)
    entries.append(f"{prefix}_OPENBOX_WORKLOAD_PRIVATE_" + f'KEY="{escaped}"')
    with Path(".env.agents").open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entries) + "\n")
    print(f"registered: {name} ({agent_id}) on {base}")
    print(f"binding updated: {bindings_path}")
    print("the agent's own keys were appended to .env.agents; give them to whoever runs the agent")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: uv run python scripts/bootstrap_agents.py bindings/<firm>.yaml")
    main(Path(sys.argv[1]))
