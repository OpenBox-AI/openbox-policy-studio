"""Create the barrier-demo agents in an OpenBox org and bind them to a matter.

    uv run python scripts/bootstrap_agents.py bindings/coca-cola.yaml

Uses OPENBOX_BACKEND_URL + OPENBOX_ORG_API_KEY from .env. Agents that already
exist in the bindings (a real UUID rather than a placeholder) are left alone,
so the script is safe to re-run. Runtime keys returned at create time are
written to .env.agents (gitignored) for the barrier demo to use.
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
            pem, jwk = workload_keypair()
            response = client.post(
                "/agent/create",
                json={
                    "agent_name": agent["name"],
                    "agent_type": "langgraph",
                    "description": f"openbox-barrier-demo agent ({agent.get('role', 'agent')})",
                    "attestation_mode": "kms",
                    # The org runs a managed Keycloak identity generation, which only
                    # admits workload-authenticated agents.
                    "identity_verification": {
                        "method": "keycloak_workload",
                        "mode": "generate",
                        "source_type": "openbox",
                        "public_jwk": jwk,
                    },
                    "icon": "https://api.iconify.design/carbon/document.svg",
                    # "medium" risk profile, same values the OpenBox CLI uses.
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
                    "tags": ["barrier-demo", data["matter"]],
                },
            )
            if response.status_code >= 300:
                detail = response.text[:300]
                sys.exit(f"create {agent['name']} failed: {response.status_code} {detail}")
            body = response.json()
            # The backend answers { result: { agent, token } }.
            result = body.get("result", body)
            created = result.get("agent", result)
            agent["id"] = created.get("id") or created.get("agent_id")
            runtime_key = result.get("token") or result.get("api_key") or ""
            slug = agent["name"].replace("ClientIntelligenceAgent", "").upper()
            if runtime_key:
                keys_out.append(f"{slug}_OPENBOX_API_KEY={runtime_key}")
            escaped = pem.replace("\n", "\\n")
            keys_out.append(f'{slug}_OPENBOX_WORKLOAD_PRIVATE_KEY="{escaped}"')
            keys_out.append(f"{slug}_OPENBOX_AGENT_NAME={agent['name']}")
            print(f"create {agent['name']} {agent['id']}")

    bindings_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    if keys_out:
        Path(".env.agents").write_text("\n".join(keys_out) + "\n")
        print(f"runtime keys written to .env.agents ({len(keys_out)})")
    print(f"bindings updated: {bindings_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "bindings/coca-cola.yaml"))
