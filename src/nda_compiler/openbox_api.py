"""OpenBox backend client: create a policy rule inactive, prove it, then activate.

Routes (openbox-backend, NestJS):
  GET  /agent/:agentId/policy-rule                     list (to retire same-name rules)
  POST /agent/:agentId/policy-rule                     create (is_active:false)
  POST /agent/:agentId/policy-rule/:ver/evaluate       dry-run against an OPA input
  PUT  /agent/:agentId/policy-rule/:ver/status         {is_active}
  GET  /agent/:agentId/logs                            the agent's governance events

Auth is the org key in X-API-Key. With no key a recording fake is used, so the
pipeline can run and the exact payloads can be inspected before a key exists.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Protocol

import httpx

from .models import Control


class OpenBoxBackend(Protocol):
    async def apply(self, control: Control) -> Control: ...

    async def close(self) -> None: ...


class RecordingBackend:
    """No network. Marks every rule as if it passed, keeps the calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def apply(self, control: Control) -> Control:
        path = f"/agent/{control.agent_id}/policy-rule"
        self.calls.append(("POST", path, control.payload))
        for test in control.tests:
            self.calls.append(("POST", f"{path}/<id>/evaluate", test.input))
        return control.model_copy(update={"status": "active", "remote_id": "dry-run"})

    async def close(self) -> None:
        return None


class HttpBackend:
    def __init__(self, base_url: str, api_key: str, timeout: float = 15.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key, "Content-Type": "application/json"},
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def apply(self, control: Control) -> Control:
        try:
            return await self._policy_rule(control)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300]
            return control.model_copy(
                update={"status": "failed", "note": f"{exc.response.status_code}: {body}"}
            )

    async def _policy_rule(self, control: Control) -> Control:
        base = f"/agent/{control.agent_id}/policy-rule"
        await self._retire_same_name(base, control.payload["rule_name"])
        created = _unwrap((await self._post(base, control.payload)).json())
        version_id = created.get("id") or created.get("rule_version_id")
        for test in control.tests:
            result = _unwrap(
                (await self._post(f"{base}/{version_id}/evaluate", {"input": test.input})).json()
            )
            decision = _decision(result)
            if decision != test.expect:
                return control.model_copy(
                    update={
                        "status": "failed",
                        "remote_id": version_id,
                        "note": f"test '{test.label}' expected {test.expect}, got {decision}",
                    }
                )
        await self._client.put(f"{base}/{version_id}/status", json={"is_active": True})
        return control.model_copy(update={"status": "active", "remote_id": version_id})

    async def _retire_same_name(self, base: str, rule_name: str) -> None:
        """Re-compiling an NDA replaces its rules instead of stacking duplicates."""

        response = await self._client.get(base, params={"limit": 200})
        if response.status_code >= 300:
            return
        page = _unwrap(response.json())
        rules = page.get("data", page) if isinstance(page, dict) else page
        for rule in rules or []:
            if rule.get("rule_name") == rule_name and rule.get("is_current_version", True):
                await self._client.delete(f"{base}/{rule['id']}")

    async def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        response = await self._client.post(path, json=body)
        response.raise_for_status()
        return response


async def fetch_activity_events(
    base_url: str, api_key: str, agent_id: str, pages: int = 40
) -> list[dict[str, Any]]:
    """The agent's recent governance events, newest first (GET /agent/:id/logs).

    The endpoint pages ten at a time whatever limit is asked for, and one run
    of the agent is well over a hundred events (chain, llm and tool events
    for every step), so enough pages are read to cover a few runs.
    """

    events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"), headers={"X-API-Key": api_key}, timeout=10
    ) as client:
        start = 0
        for _ in range(pages):
            try:
                response = await client.get(
                    f"/agent/{agent_id}/logs", params={"limit": 50, "start": start}
                )
            except httpx.HTTPError:
                break
            if response.status_code >= 300:
                break
            page = _unwrap(response.json())
            rows = page.get("data", []) if isinstance(page, dict) else page
            if not rows:
                break
            events.extend(rows)
            start += len(rows)
            if isinstance(page, dict) and start >= int(page.get("total", 0)):
                break
    return events


def _unwrap(body: Any) -> Any:
    """The backend wraps every response as {status, data}."""

    if isinstance(body, dict) and "data" in body and "status" in body:
        return body["data"]
    return body


def _decision(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("decision", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                return _decision(value)
    return "ALLOW"


def backend_from_env() -> OpenBoxBackend:
    key = os.environ.get("OPENBOX_ORG_API_KEY", "").strip()
    if not key:
        return RecordingBackend()
    url = os.environ.get("OPENBOX_BACKEND_URL", "").strip() or "https://api.openbox.ai"
    return HttpBackend(url, key)


async def apply_all(backend: OpenBoxBackend, controls: list[Control]) -> list[Control]:
    return list(await asyncio.gather(*(backend.apply(c) for c in controls)))
