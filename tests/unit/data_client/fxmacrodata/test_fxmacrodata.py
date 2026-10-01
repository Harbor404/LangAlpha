"""Contract tests for the optional FXMacroData macro source."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from src.data_client.fxmacrodata import (
    FXMacroDataClient,
    FXMacroDataRequestError,
    FXMacroDataResponseError,
    FXMacroDataSource,
    FXMacroDataUnavailable,
    get_fxmacrodata_source,
)


def _iso_seconds(value: int) -> str:
    return datetime.fromtimestamp(value, tz=UTC).isoformat()


@pytest.mark.asyncio
async def test_source_discovery_exports_factory():
    source = await get_fxmacrodata_source()

    assert isinstance(source, FXMacroDataSource)


def test_missing_key_disables_non_usd_but_keeps_keyless_usd():
    source = FXMacroDataSource(client=FXMacroDataClient(api_key=""))

    assert source.has_api_key is False
    assert source.supports_currency("usd") is True
    assert source.supports_currency("AUD") is False


@pytest.mark.asyncio
async def test_missing_key_rejects_non_usd_without_http_request():
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={"data": []}, request=request)

    source = FXMacroDataSource(
        client=FXMacroDataClient(api_key="", transport=httpx.MockTransport(handler))
    )

    with pytest.raises(FXMacroDataUnavailable, match="non-USD"):
        await source.get_macro_announcements("AUD", "policy_rate")

    assert called is False


@pytest.mark.asyncio
async def test_announcements_use_header_offset_paging_and_normalize_null_values():
    seen: dict[str, object] = {}
    timestamp = 1767225600

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["api_key"] = request.headers.get("X-API-Key")
        return httpx.Response(
            200,
            json={
                "currency": "AUD",
                "indicator": "policy_rate",
                "pagination": {
                    "limit": 10,
                    "offset": 20,
                    "returned_count": 1,
                    "total_count": 31,
                    "has_more": True,
                    "next_offset": 30,
                },
                "data": [
                    {
                        "date": "2026-01-01",
                        "val": 4.35,
                        "previous_value": None,
                        "announcement_datetime": timestamp,
                        "release_time_assumed": None,
                        "source": "Reserve Bank of Australia",
                    }
                ],
            },
            request=request,
        )

    source = FXMacroDataSource(
        client=FXMacroDataClient(
            api_key="test-key", transport=httpx.MockTransport(handler)
        )
    )

    result = await source.get_macro_announcements(
        "aud",
        "policy_rate",
        start_date="2025-01-01",
        end_date="2025-12-31",
        limit=10,
        offset=20,
    )

    assert seen == {
        "path": "/v1/announcements/AUD/policy_rate",
        "params": {
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "limit": "10",
            "offset": "20",
        },
        "api_key": "test-key",
    }
    assert result["currency"] == "AUD"
    assert result["indicator"] == "policy_rate"
    assert result["pagination"]["next_offset"] == 30
    row = result["data"][0]
    assert row["value"] == 4.35
    assert row["val"] == 4.35
    assert row["previous_value"] is None
    assert row["announcement_datetime_utc"] == _iso_seconds(timestamp)


@pytest.mark.asyncio
async def test_keyless_usd_request_omits_api_key_header():
    seen: dict[str, str | None] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers.get("X-API-Key")
        return httpx.Response(
            200,
            json={"currency": "USD", "indicator": "inflation", "data": []},
            request=request,
        )

    source = FXMacroDataSource(
        client=FXMacroDataClient(api_key="", transport=httpx.MockTransport(handler))
    )

    result = await source.get_macro_announcements("USD", "inflation")

    assert result["data"] == []
    assert seen["api_key"] is None


@pytest.mark.asyncio
async def test_release_calendar_uses_calendar_path_and_normalizes_rows():
    seen: dict[str, object] = {}
    timestamp = 1767225600

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "currency": "EUR",
                "timezone": "Europe/Berlin",
                "data": [
                    {
                        "announcement_datetime": timestamp,
                        "release": "inflation",
                        "date": "2026-01-01",
                        "event_importance": "high",
                    }
                ],
            },
            request=request,
        )

    source = FXMacroDataSource(
        client=FXMacroDataClient(
            api_key="test-key", transport=httpx.MockTransport(handler)
        )
    )

    result = await source.get_macro_release_calendar(
        "eur",
        indicator="inflation",
        start_date="2026-01-01",
        end_date="2026-03-31",
        timezone="Europe/Berlin",
    )

    assert seen == {
        "path": "/v1/calendar/EUR",
        "params": {
            "indicator": "inflation",
            "start_date": "2026-01-01",
            "end_date": "2026-03-31",
            "timezone": "Europe/Berlin",
        },
    }
    assert result["currency"] == "EUR"
    row = result["data"][0]
    assert row["release"] == "inflation"
    assert row["announcement_datetime"] == timestamp
    assert row["announcement_datetime_utc"] == _iso_seconds(timestamp)


@pytest.mark.asyncio
async def test_comm_release_calendar_accepts_shared_currency_code():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(
            200,
            json={"currency": "COMM", "data": []},
            request=request,
        )

    source = FXMacroDataSource(
        client=FXMacroDataClient(
            api_key="test-key", transport=httpx.MockTransport(handler)
        )
    )

    result = await source.get_macro_release_calendar("comm")

    assert seen["path"] == "/v1/calendar/COMM"
    assert result["currency"] == "COMM"


@pytest.mark.asyncio
async def test_http_error_is_sanitized_and_maps_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"detail": "bad key SECRET /v1/announcements?api_key=SECRET"},
            request=request,
        )

    client = FXMacroDataClient(api_key="SECRET", transport=httpx.MockTransport(handler))

    with pytest.raises(FXMacroDataRequestError) as caught:
        await client.get_announcements("USD", "inflation")

    assert caught.value.status_code == 401
    assert "SECRET" not in str(caught.value)
    assert "api.fxmacrodata.com" not in str(caught.value)


@pytest.mark.asyncio
async def test_timeout_is_sanitized():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout at ?api_key=SECRET", request=request)

    client = FXMacroDataClient(api_key="SECRET", transport=httpx.MockTransport(handler))

    with pytest.raises(FXMacroDataRequestError, match="timed out") as caught:
        await client.get_calendar("USD")

    assert "SECRET" not in str(caught.value)


@pytest.mark.asyncio
async def test_non_object_response_fails_closed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"], request=request)

    client = FXMacroDataClient(
        api_key="test-key", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(FXMacroDataResponseError, match="invalid response"):
        await client.get_announcements("USD", "inflation")


@pytest.mark.asyncio
async def test_non_list_data_fails_closed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": {"not": "a list"}}, request=request)

    client = FXMacroDataClient(
        api_key="test-key", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(FXMacroDataResponseError, match="invalid response"):
        await client.get_calendar("USD")


@pytest.mark.asyncio
async def test_malformed_json_fails_closed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"{",
            headers={"content-type": "application/json"},
            request=request,
        )

    client = FXMacroDataClient(
        api_key="test-key", transport=httpx.MockTransport(handler)
    )

    with pytest.raises(FXMacroDataResponseError, match="invalid response"):
        await client.get_announcements("USD", "inflation")
