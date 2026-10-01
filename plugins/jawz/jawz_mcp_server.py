#!/usr/bin/env python3
"""Optional Jawz macro MCP adapter.

Jawz publishes a keyless, read-only MCP streamable-HTTP endpoint. This adapter
exposes the four capabilities named in the integration issue while keeping the
remote transport strict: a single JSON-RPC request shape, explicit timeout,
no credential, no third-party SDK, and sanitized errors.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Self

import httpx

try:
    from _bootstrap import MCPServer  # script launch: bundle is sys.path[0]
except ModuleNotFoundError:  # imported as a package module (tests)
    from plugins.jawz._bootstrap import MCPServer

from mcp_servers._envelope import make_error, make_response
from mcp_servers._schemas import OBJECT, STR, envelope_schema, output_model

JAWZ_MCP_URL = "https://jawz.ai/api/mcp"
JAWZ_TIMEOUT_SECONDS = 20.0
SOURCE = "jawz"

mcp = MCPServer("JawzMCP")

EventType = Literal["fomc", "cpi", "nfp", "pce", "ecb", "boj", "boe"]


class JawzError(Exception):
    """A sanitized failure that is safe to turn into an agent envelope."""

    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


class JawzClient:
    """One-request MCP client for the published Jawz endpoint."""

    def __init__(
        self,
        *,
        url: str = JAWZ_MCP_URL,
        timeout: float = JAWZ_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self.url = url
        self.timeout = timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            transport=self._transport,
        )
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one Jawz tool and return only its structuredContent object."""
        if self._client is None:
            raise JawzError("client_unavailable", "Jawz client is unavailable")

        request_id = 1
        # Jawz's published read surface accepts one-shot tools/call requests
        # without an initialize handshake; keep that assumption at the boundary.
        body = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
        try:
            response = await self._client.post(
                self.url,
                headers={"Accept": "application/json, text/event-stream"},
                json=body,
            )
        except httpx.TimeoutException:
            raise JawzError("upstream_error", "Jawz request timed out") from None
        except httpx.RequestError:
            raise JawzError("upstream_error", "Jawz request failed") from None

        self._raise_for_status(response.status_code)
        payload = self._decode_payload(response, request_id)
        if "error" in payload:
            raise JawzError("upstream_error", "Jawz reported an upstream error")

        result = payload.get("result")
        if not isinstance(result, dict) or result.get("isError"):
            raise JawzError("upstream_error", "Jawz reported an upstream error")

        structured = result.get("structuredContent")
        if not isinstance(structured, dict):
            raise JawzError("upstream_error", "Jawz returned an invalid response")
        return structured

    @staticmethod
    def _raise_for_status(status: int) -> None:
        if status < 400:
            return
        if status in (401, 403):
            raise JawzError("auth_failed", "Jawz authentication failed")
        if status == 404:
            raise JawzError("not_found", "Jawz tool was not found")
        if status == 429:
            raise JawzError("rate_limited", "Jawz rate limit reached")
        if 400 <= status < 500:
            raise JawzError("invalid_argument", "Jawz rejected the request")
        raise JawzError("upstream_error", "Jawz request failed")

    @staticmethod
    def _decode_payload(response: httpx.Response, request_id: int) -> dict[str, Any]:
        text = response.text
        data_lines = [
            line[5:].lstrip() for line in text.splitlines() if line.startswith("data:")
        ]
        if data_lines:
            text = "\n".join(data_lines)
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError, ValueError):
            raise JawzError(
                "upstream_error", "Jawz returned an invalid response"
            ) from None
        if (
            not isinstance(payload, dict)
            or payload.get("jsonrpc") != "2.0"
            or payload.get("id") != request_id
        ):
            raise JawzError("upstream_error", "Jawz returned an invalid response")
        return payload


_OUT_MACRO_REGIME = output_model(
    "GetJawzMacroRegimeOut",
    envelope_schema(
        OBJECT,
        echo={"tool": STR},
        data_description="Jawz structured macro-regime payload.",
    ),
)
_OUT_FINANCIAL_CONDITIONS = output_model(
    "GetJawzFinancialConditionsOut",
    envelope_schema(
        OBJECT,
        echo={"tool": STR},
        data_description="Jawz structured financial-conditions payload.",
    ),
)
_OUT_LIQUIDITY_HISTORY = output_model(
    "GetJawzLiquidityHistoryOut",
    envelope_schema(
        OBJECT,
        echo={"tool": STR},
        data_description="Jawz structured global-liquidity history.",
    ),
)
_OUT_EVENT_CALENDAR = output_model(
    "GetJawzEventCalendarOut",
    envelope_schema(
        OBJECT, echo={"tool": STR}, data_description="Jawz structured event calendar."
    ),
)


def _invalid_response(tool: str) -> dict[str, Any]:
    return make_error(
        "upstream_error",
        f"Jawz returned an invalid {tool} response",
        tool=tool,
    )


@mcp.tool()
async def get_macro_regime() -> _OUT_MACRO_REGIME:
    """Fetch Jawz's structured macro-regime read with source and as-of provenance.

    Returns:
        dict: {count, data, source, tool}. data is Jawz's structured macro
        regime payload: status, summary (regime and business cycle), pillars,
        presentation, tables, and provenance. count is 1. On error: {error:
        <code>, detail, tool}.
    """
    try:
        async with JawzClient() as client:
            data = await client.call("get_macro_regime", {})
        if not isinstance(data.get("summary"), dict):
            return _invalid_response("macro-regime")
    except JawzError as exc:
        return make_error(exc.code, exc.detail, tool="get_macro_regime")
    except Exception:  # noqa: BLE001
        return make_error(
            "upstream_error", "Jawz request failed", tool="get_macro_regime"
        )
    return make_response(data, source=SOURCE, tool="get_macro_regime", count=1)


@mcp.tool()
async def get_financial_conditions(
    summary_only: bool = False,
) -> _OUT_FINANCIAL_CONDITIONS:
    """Fetch Jawz's structured financial-conditions and liquidity read.

    Args:
        summary_only: Return the composite, direction, and one-line driver only.

    Returns:
        dict: {count, data, source, tool}. data is Jawz's structured financial
        conditions payload with composite, direction, liquidity basis/change,
        as-of, and provenance. count is 1. On error: {error: <code>, detail,
        tool}.
    """
    try:
        async with JawzClient() as client:
            data = await client.call(
                "get_financial_conditions", {"summary_only": summary_only}
            )
        if not isinstance(data.get("composite"), str) or not isinstance(
            data.get("direction"), str
        ):
            return _invalid_response("financial-conditions")
    except JawzError as exc:
        return make_error(exc.code, exc.detail, tool="get_financial_conditions")
    except Exception:  # noqa: BLE001
        return make_error(
            "upstream_error",
            "Jawz request failed",
            tool="get_financial_conditions",
        )
    return make_response(
        data,
        source=SOURCE,
        tool="get_financial_conditions",
        count=1,
    )


@mcp.tool()
async def get_liquidity_history(
    lookback_weeks: int = 12,
    interval: Literal["weekly", "monthly"] = "weekly",
) -> _OUT_LIQUIDITY_HISTORY:
    """Fetch Jawz's structured global-liquidity history and liquidity events.

    Args:
        lookback_weeks: Weeks of history to cover (clamped by Jawz to 1-52).
        interval: Sample cadence, weekly or monthly.

    Returns:
        dict: {count, data, source, tool}. data is Jawz's structured liquidity
        history: data.rows, events, trend, and basis changes. count is the row
        total. On error: {error: <code>, detail, tool}.
    """
    try:
        async with JawzClient() as client:
            data = await client.call(
                "get_liquidity_history",
                {"lookback_weeks": lookback_weeks, "interval": interval},
            )
        nested = data.get("data")
        rows = nested.get("rows") if isinstance(nested, dict) else None
        if not isinstance(rows, list):
            return _invalid_response("liquidity-history")
    except JawzError as exc:
        return make_error(exc.code, exc.detail, tool="get_liquidity_history")
    except Exception:  # noqa: BLE001
        return make_error(
            "upstream_error", "Jawz request failed", tool="get_liquidity_history"
        )
    return make_response(
        data,
        source=SOURCE,
        tool="get_liquidity_history",
        count=len(rows),
    )


@mcp.tool()
async def get_event_calendar(
    lookahead_days: int = 14,
    event_types: list[EventType] | None = None,
) -> _OUT_EVENT_CALENDAR:
    """Fetch Jawz's sourced forward macro-event calendar.

    Args:
        lookahead_days: Days forward to look.
        event_types: Optional filters: fomc, cpi, nfp, pce, ecb, boj, boe.

    Returns:
        dict: {count, data, source, tool}. data is Jawz's structured calendar
        with events, as_of, data_sources, and staleness flags. count is the
        event total. On error: {error: <code>, detail, tool}.
    """
    arguments: dict[str, Any] = {"lookahead_days": lookahead_days}
    if event_types is not None:
        arguments["event_types"] = event_types
    try:
        async with JawzClient() as client:
            data = await client.call("get_event_calendar", arguments)
        events = data.get("events")
        if not isinstance(events, list):
            return _invalid_response("event-calendar")
    except JawzError as exc:
        return make_error(exc.code, exc.detail, tool="get_event_calendar")
    except Exception:  # noqa: BLE001
        return make_error(
            "upstream_error", "Jawz request failed", tool="get_event_calendar"
        )
    return make_response(
        data,
        source=SOURCE,
        tool="get_event_calendar",
        count=len(events),
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
