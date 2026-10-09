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
        except httpx.TransportError as exc:
            # A timeout fails this rule only; the rest of the Apply carries on.
            return control.model_copy(
                update={"status": "failed", "note": f"no response from OpenBox ({type(exc).__name__})"}
            )

    async def _policy_rule(self, control: Control) -> Control:
        """Create the rule, or version it if a rule with this name already exists.

        OpenBox keeps a rule's history under one base_rule_id: PUT on the
        current version writes a new version and retires the old one, and the
        Policies tab shows the lineage. An existing rule whose conditions and
        decision already match is left alone and reported as unchanged.
        """

        base = f"/agent/{control.agent_id}/policy-rule"
        existing = await self._current_by_name(base, control.payload["rule_name"])
        if existing is not None and _same_rule(existing, control.payload):
            if not existing.get("is_active"):
                await self._client.put(f"{base}/{existing['id']}/status", json={"is_active": True})
            return control.model_copy(
                update={
                    "status": "active",
                    "remote_id": existing["id"],
                    "note": "already on OpenBox with these conditions; left unchanged",
                }
            )
        if existing is not None:
            body = {**control.payload, "change_log": f"Recompiled from the NDA: {control.payload['reason'][:160]}"}
            response = await self._client.put(f"{base}/{existing['id']}", json=body)
            response.raise_for_status()
            created = _unwrap(response.json())
            note = f"new version of an existing rule (was {existing['id'][:8]})"
        else:
            created = _unwrap((await self._post(base, control.payload)).json())
            note = ""
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
        return control.model_copy(update={"status": "active", "remote_id": version_id, "note": note})

    async def _current_by_name(self, base: str, rule_name: str) -> dict[str, Any] | None:
        # The list is paged (page from 0, perPage); an agent with more rules than
        # one page holds must be read to the end, or an existing rule is missed
        # and creating it again fails with POLICY_RULE_NAME_TAKEN.
        per_page = 100
        for page_no in range(50):
            response = await self._client.get(base, params={"page": page_no, "perPage": per_page})
            if response.status_code >= 300:
                return None
            page = _unwrap(response.json())
            rules = page.get("data", page) if isinstance(page, dict) else page
            for rule in rules or []:
                if rule.get("rule_name") == rule_name and rule.get("is_current_version", True):
                    return rule
            if not rules or len(rules) < per_page:
                return None
        return None

    async def _post(self, path: str, body: dict[str, Any]) -> httpx.Response:
        response = await self._client.post(path, json=body)
        response.raise_for_status()
        return response


async def fetch_activity_events(
    base_url: str, api_key: str, agent_id: str, pages: int = 10
) -> list[dict[str, Any]]:
    """The agent's recent governance events, newest first (GET /agent/:id/logs).

    The endpoint pages ten at a time whatever limit is asked for, and one run
    of the agent is well over a hundred events (chain, llm and tool events
    for every step), so enough pages are read to cover a few runs.
    """

    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"), headers={"X-API-Key": api_key}, timeout=10
    ) as client:
        for page_no in range(pages):
            try:
                # PaginationDto: zero-based `page`, `perPage` (the `limit`/`start`
                # names were being ignored, which returned the same ten rows
                # on every page and inflated every count).
                response = await client.get(
                    f"/agent/{agent_id}/logs", params={"page": page_no, "perPage": 50}
                )
            except httpx.HTTPError:
                break
            if response.status_code >= 300:
                break
            page = _unwrap(response.json())
            rows = page.get("data", []) if isinstance(page, dict) else page
            # Rows are deduplicated by id: a page that brings nothing new means
            # the server ignored the offset, and the walk stops there.
            fresh = [r for r in rows or [] if str(r.get("id")) not in seen]
            if not fresh:
                break
            seen.update(str(r.get("id")) for r in fresh)
            events.extend(fresh)
            if isinstance(page, dict) and len(events) >= int(page.get("total", 0)):
                break
    return events


def _same_rule(existing: dict[str, Any], payload: dict[str, Any]) -> bool:
    """Same decision, match mode, priority and conditions (ignoring condition ids)."""

    def strip(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            ({k: v for k, v in c.items() if k != "id"} for c in conditions),
            key=lambda c: str(c),
        )

    return (
        existing.get("decision") == payload["decision"]
        and existing.get("match_mode") == payload["match_mode"]
        and int(existing.get("priority", -1)) == payload["priority"]
        and strip(existing.get("conditions", [])) == strip(payload["conditions"])
    )


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


# Each save rebuilds the agent's policy bundle on the backend; a handful at a
# time keeps a long NDA's Apply inside the request timeout.
APPLY_CONCURRENCY = int(os.environ.get("OPENBOX_APPLY_CONCURRENCY", "3"))


async def apply_all(backend: OpenBoxBackend, controls: list[Control]) -> list[Control]:
    gate = asyncio.Semaphore(max(1, APPLY_CONCURRENCY))

    async def one(control: Control) -> Control:
        async with gate:
            return await backend.apply(control)

    return list(await asyncio.gather(*(one(c) for c in controls)))
