"""Macro MCP tests for the optional FXMacroData-backed tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from .conftest import assert_error, assert_ok_envelope

_MOD = "plugins.langalpha_market_data.macro_mcp_server"


class _FakeFXMacroDataSource:
    def __init__(self):
        self.get_macro_announcements = AsyncMock(
            return_value={
                "currency": "AUD",
                "indicator": "policy_rate",
                "pagination": {
                    "offset": 0,
                    "returned_count": 1,
                    "total_count": 1,
                    "has_more": False,
                },
                "data": [{"date": "2026-01-01", "val": 4.35, "value": 4.35}],
            }
        )
        self.get_macro_release_calendar = AsyncMock(
            return_value={
                "currency": "EUR",
                "timezone": "Europe/Berlin",
                "data": [{"announcement_datetime": 1767225600, "release": "inflation"}],
            }
        )


class TestGetMacroAnnouncements:
    @pytest.mark.asyncio
    async def test_success(self):
        from plugins.langalpha_market_data.macro_mcp_server import (
            get_macro_announcements,
        )

        source = _FakeFXMacroDataSource()
        with patch(f"{_MOD}.get_fxmacrodata_source", return_value=source):
            result = await get_macro_announcements(
                "AUD",
                "policy_rate",
                start_date="2025-01-01",
                end_date="2025-12-31",
                limit=10,
                offset=20,
            )

        assert_ok_envelope(result, source="fxmacrodata", count=1)
        assert result["data_type"] == "macro_announcements"
        assert result["currency"] == "AUD"
        assert result["indicator"] == "policy_rate"
        assert result["limit"] == 10
        assert result["offset"] == 20
        assert result["pagination"]["offset"] == 0
        source.get_macro_announcements.assert_awaited_once_with(
            "AUD",
            "policy_rate",
            start_date="2025-01-01",
            end_date="2025-12-31",
            limit=10,
            offset=20,
        )

    @pytest.mark.asyncio
    async def test_missing_key_non_usd_is_client_unavailable(self):
        from plugins.langalpha_market_data.macro_mcp_server import (
            get_macro_announcements,
        )
        from src.data_client.fxmacrodata import FXMacroDataUnavailable

        source = _FakeFXMacroDataSource()
        source.get_macro_announcements.side_effect = FXMacroDataUnavailable(
            "FXMacroData API key is required for non-USD currencies"
        )
        with patch(f"{_MOD}.get_fxmacrodata_source", return_value=source):
            result = await get_macro_announcements("AUD", "policy_rate")

        assert_error(result, "client_unavailable", detail_excludes=("SECRET",))
        assert result["currency"] == "AUD"
        assert result["indicator"] == "policy_rate"

    @pytest.mark.asyncio
    async def test_upstream_error_is_sanitized(self):
        from plugins.langalpha_market_data.macro_mcp_server import (
            get_macro_announcements,
        )

        source = _FakeFXMacroDataSource()
        source.get_macro_announcements.side_effect = RuntimeError(
            "upstream failed at ?api_key=SECRET"
        )
        with patch(f"{_MOD}.get_fxmacrodata_source", return_value=source):
            result = await get_macro_announcements("USD", "inflation")

        assert_error(result, "upstream_error", detail_excludes=("SECRET",))
        assert result["currency"] == "USD"
        assert result["indicator"] == "inflation"


class TestGetMacroReleaseCalendar:
    @pytest.mark.asyncio
    async def test_success(self):
        from plugins.langalpha_market_data.macro_mcp_server import (
            get_macro_release_calendar,
        )

        source = _FakeFXMacroDataSource()
        with patch(f"{_MOD}.get_fxmacrodata_source", return_value=source):
            result = await get_macro_release_calendar(
                "EUR",
                indicator="inflation",
                start_date="2026-01-01",
                end_date="2026-03-31",
                timezone="Europe/Berlin",
            )

        assert_ok_envelope(result, source="fxmacrodata", count=1)
        assert result["data_type"] == "macro_release_calendar"
        assert result["currency"] == "EUR"
        assert result["indicator"] == "inflation"
        assert result["start_date"] == "2026-01-01"
        assert result["end_date"] == "2026-03-31"
        assert result["timezone"] == "Europe/Berlin"
        source.get_macro_release_calendar.assert_awaited_once_with(
            "EUR",
            indicator="inflation",
            start_date="2026-01-01",
            end_date="2026-03-31",
            timezone="Europe/Berlin",
        )
