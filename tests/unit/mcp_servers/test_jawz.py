"""Contract tests for the optional Jawz stdio adapter.

Jawz's published contract is a keyless MCP streamable-HTTP endpoint. The
adapter is deliberately narrow: four advertised macro tools, one JSON-RPC call
shape, strict response parsing, and the repository's standard success/error
envelope. Remote text never becomes agent-visible error detail.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import httpx
import pytest

from .conftest import assert_error, assert_ok_envelope

_MOD = "plugins.jawz.jawz_mcp_server"


def _sse(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text=f"event: message\ndata: {json.dumps(payload)}\n\n",
    )


def _result(structured: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"structuredContent": structured, "content": []},
    }


class _FakeJawzClient:
    def __init__(
        self, result: dict[str, Any] | None = None, error: Exception | None = None
    ):
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, arguments))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@pytest.fixture
def fake_jawz(monkeypatch):
    def _install(
        *, result: dict[str, Any] | None = None, error: Exception | None = None
    ) -> _FakeJawzClient:
        fake = _FakeJawzClient(result=result, error=error)
        module = importlib.import_module(_MOD)
        monkeypatch.setattr(module, "JawzClient", lambda: fake)
        return fake

    return _install


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "function_name",
        "kwargs",
        "expected_tool",
        "expected_arguments",
        "structured",
        "count",
    ),
    [
        (
            "get_macro_regime",
            {},
            "get_macro_regime",
            {},
            {"status": "ok", "summary": {"regime": "YELLOW"}},
            1,
        ),
        (
            "get_financial_conditions",
            {"summary_only": True},
            "get_financial_conditions",
            {"summary_only": True},
            {"composite": "neutral", "direction": "tightening"},
            1,
        ),
        (
            "get_liquidity_history",
            {"lookback_weeks": 4, "interval": "monthly"},
            "get_liquidity_history",
            {"lookback_weeks": 4, "interval": "monthly"},
            {"data": {"rows": [{"date": "2026-08-01"}, {"date": "2026-09-01"}]}},
            2,
        ),
        (
            "get_event_calendar",
            {"lookahead_days": 7, "event_types": ["cpi", "fomc"]},
            "get_event_calendar",
            {"lookahead_days": 7, "event_types": ["cpi", "fomc"]},
            {"events": [{"type": "cpi", "date": "2026-10-02"}]},
            1,
        ),
    ],
)
async def test_tool_requests_map_to_jawz_and_results_use_the_standard_envelope(
    fake_jawz,
    function_name,
    kwargs,
    expected_tool,
    expected_arguments,
    structured,
    count,
):
    module = importlib.import_module(_MOD)
    function = getattr(module, function_name)
    fake = fake_jawz(result=structured)

    result = await function(**kwargs)

    assert fake.calls == [(expected_tool, expected_arguments)]
    assert_ok_envelope(result, source="jawz", count=count, extra_keys=("tool",))
    assert result["tool"] == expected_tool
    assert result["data"] == structured


@pytest.mark.asyncio
async def test_http_call_uses_the_published_streamable_mcp_contract():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _sse(_result({"status": "ok", "summary": {"regime": "GREEN"}}))

    from plugins.jawz.jawz_mcp_server import JawzClient

    async with JawzClient(transport=httpx.MockTransport(handler)) as client:
        structured = await client.call("get_macro_regime", {})

    assert structured == {"status": "ok", "summary": {"regime": "GREEN"}}
    request = seen[0]
    assert str(request.url) == "https://jawz.ai/api/mcp"
    assert request.method == "POST"
    assert request.headers["accept"] == "application/json, text/event-stream"
    assert "authorization" not in request.headers
    body = json.loads(request.content)
    assert body["jsonrpc"] == "2.0"
    assert body["method"] == "tools/call"
    assert body["params"] == {"name": "get_macro_regime", "arguments": {}}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "error_code"),
    [
        (
            httpx.Response(500, text="upstream body has SECRET"),
            "upstream_error",
        ),
        (
            httpx.Response(429, text="rate limit SECRET"),
            "rate_limited",
        ),
        (
            httpx.Response(400, text="bad arguments SECRET"),
            "invalid_argument",
        ),
        (
            httpx.Response(401, text="auth SECRET"),
            "auth_failed",
        ),
        (
            httpx.Response(404, text="missing SECRET"),
            "not_found",
        ),
    ],
)
async def test_http_failures_fail_closed_without_echoing_the_body(response, error_code):
    from plugins.jawz.jawz_mcp_server import JawzClient, JawzError

    async with JawzClient(
        transport=httpx.MockTransport(lambda _request: response)
    ) as client:
        with pytest.raises(JawzError) as exc:
            await client.call("get_macro_regime", {})

    assert exc.value.code == error_code
    assert "SECRET" not in str(exc.value)


@pytest.mark.asyncio
async def test_timeout_fails_closed_without_echoing_the_transport_error():
    from plugins.jawz.jawz_mcp_server import JawzClient, JawzError

    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out at ?token=SECRET")

    async with JawzClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JawzError) as exc:
            await client.call("get_macro_regime", {})

    assert exc.value.code == "upstream_error"
    assert "SECRET" not in str(exc.value)
    assert "timed out" in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json SECRET"),
        httpx.Response(200, text="event: message\ndata: not-json SECRET\n\n"),
        _sse({"jsonrpc": "2.0", "id": 1}),
        _sse({"jsonrpc": "2.0", "id": 1, "error": {"message": "SECRET"}}),
        _sse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"isError": True, "content": [{"text": "SECRET"}]},
            }
        ),
        _sse({"jsonrpc": "2.0", "id": 1, "result": {"content": []}}),
        _sse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"structuredContent": "SECRET"},
            }
        ),
    ],
)
async def test_invalid_or_error_responses_fail_closed_without_echoing_remote_text(
    response,
):
    from plugins.jawz.jawz_mcp_server import JawzClient, JawzError

    async with JawzClient(
        transport=httpx.MockTransport(lambda _request: response)
    ) as client:
        with pytest.raises(JawzError) as exc:
            await client.call("get_macro_regime", {})

    assert exc.value.code == "upstream_error"
    assert "SECRET" not in str(exc.value)


@pytest.mark.asyncio
async def test_adapter_converts_unexpected_exceptions_to_a_sanitized_envelope(
    fake_jawz,
):
    from plugins.jawz.jawz_mcp_server import get_macro_regime

    fake_jawz(error=RuntimeError("SECRET from transport"))

    result = await get_macro_regime()

    assert_error(result, "upstream_error", detail_excludes=("SECRET",))
    assert result["tool"] == "get_macro_regime"
