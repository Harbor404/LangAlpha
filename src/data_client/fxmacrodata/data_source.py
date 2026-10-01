"""Normalized FXMacroData source for macro releases and release calendars."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from .client import (
    FXMacroDataClient,
    FXMacroDataError,
)

ANNOUNCEMENT_CURRENCIES = frozenset(
    {
        "AUD",
        "BRL",
        "CAD",
        "CHF",
        "CNH",
        "CNY",
        "DKK",
        "EUR",
        "GBP",
        "HUF",
        "ILS",
        "JPY",
        "KRW",
        "MYR",
        "NGN",
        "NOK",
        "NZD",
        "PEN",
        "SEK",
        "THB",
        "TWD",
        "USD",
    }
)
CALENDAR_CURRENCIES = ANNOUNCEMENT_CURRENCIES | {"COMM"}
KEYLESS_CURRENCIES = frozenset({"USD"})

_INDICATOR_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class FXMacroDataUnavailable(FXMacroDataError):
    """The source lacks the credential required for a requested capability."""


class FXMacroDataInvalidArgument(FXMacroDataError):
    """The caller supplied a value outside the provider's documented surface."""


def _currency(value: str, *, calendar: bool = False) -> str:
    normalized = value.strip().upper()
    supported = CALENDAR_CURRENCIES if calendar else ANNOUNCEMENT_CURRENCIES
    if normalized not in supported:
        raise FXMacroDataInvalidArgument(
            f"Unsupported FXMacroData currency: {normalized or value!r}"
        )
    return normalized


def _indicator(value: str) -> str:
    normalized = value.strip().lower()
    if not _INDICATOR_RE.match(normalized):
        raise FXMacroDataInvalidArgument("Invalid FXMacroData indicator")
    return normalized


def _iso_from_seconds(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def normalize_announcement(raw: dict[str, Any]) -> dict[str, Any]:
    """Preserve the upstream row and add stable value/time aliases."""
    row = dict(raw)
    if "val" in raw:
        row["value"] = raw["val"]
    if row.get("announcement_datetime_utc") is None:
        converted = _iso_from_seconds(raw.get("announcement_datetime"))
        if converted is not None:
            row["announcement_datetime_utc"] = converted
    return row


def normalize_release(raw: dict[str, Any]) -> dict[str, Any]:
    """Preserve the calendar row and add an ISO release timestamp."""
    row = dict(raw)
    if row.get("announcement_datetime_utc") is None:
        converted = _iso_from_seconds(raw.get("announcement_datetime"))
        if converted is not None:
            row["announcement_datetime_utc"] = converted
    return row


class FXMacroDataSource:
    """Key-aware adapter over :class:`FXMacroDataClient`."""

    def __init__(self, client: FXMacroDataClient | None = None) -> None:
        self.client = client or FXMacroDataClient()

    @property
    def has_api_key(self) -> bool:
        return self.client.has_api_key

    def supports_currency(self, currency: str) -> bool:
        normalized = currency.strip().upper()
        if normalized not in CALENDAR_CURRENCIES:
            return False
        return self.has_api_key or normalized in KEYLESS_CURRENCIES

    def _require_currency(self, currency: str, *, calendar: bool = False) -> str:
        normalized = _currency(currency, calendar=calendar)
        if normalized not in KEYLESS_CURRENCIES and not self.has_api_key:
            raise FXMacroDataUnavailable(
                "FXMacroData API key is required for non-USD currencies"
            )
        return normalized

    async def get_macro_announcements(
        self,
        currency: str,
        indicator: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch and normalize one currency/indicator release series."""
        normalized_currency = self._require_currency(currency)
        normalized_indicator = _indicator(indicator)
        if not 1 <= limit <= 100:
            raise FXMacroDataInvalidArgument(
                "FXMacroData limit must be between 1 and 100"
            )
        if offset < 0:
            raise FXMacroDataInvalidArgument("FXMacroData offset must be non-negative")
        payload = await self.client.get_announcements(
            normalized_currency,
            normalized_indicator,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
        return {
            **payload,
            "currency": payload.get("currency", normalized_currency),
            "indicator": payload.get("indicator", normalized_indicator),
            "data": [normalize_announcement(row) for row in payload["data"]],
        }

    async def get_macro_release_calendar(
        self,
        currency: str,
        *,
        indicator: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Fetch and normalize the forward release calendar for one currency."""
        normalized_currency = self._require_currency(currency, calendar=True)
        normalized_indicator = _indicator(indicator) if indicator else None
        payload = await self.client.get_calendar(
            normalized_currency,
            indicator=normalized_indicator,
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
        )
        return {
            **payload,
            "currency": payload.get("currency", normalized_currency),
            "data": [normalize_release(row) for row in payload["data"]],
        }

    async def close(self) -> None:
        await self.client.close()


# Export both spellings for callers that treat the acronym as a word.
FxMacroDataSource = FXMacroDataSource
