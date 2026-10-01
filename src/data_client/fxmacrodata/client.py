"""Async client for the FXMacroData macro-release API.

The client deliberately keeps upstream text out of raised exceptions: request
URLs can contain credentials when an integration falls back to query-string
auth, so callers only receive a stable, sanitized summary.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

_PLACEHOLDER_PREFIX = "${"


class FXMacroDataError(Exception):
    """Base class for sanitized FXMacroData failures."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class FXMacroDataRequestError(FXMacroDataError):
    """An HTTP transport or status failure."""


class FXMacroDataResponseError(FXMacroDataError):
    """A successful HTTP response whose body violates the API contract."""


def _clean_api_key(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.startswith(_PLACEHOLDER_PREFIX):
        return None
    return cleaned


class FXMacroDataClient:
    """Small async client for FXMacroData announcements and calendar endpoints."""

    BASE_URL = "https://api.fxmacrodata.com"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = _clean_api_key(
            api_key if api_key is not None else os.getenv("FXMACRODATA_API_KEY")
        )
        self.base_url = (
            base_url or os.getenv("FXMACRODATA_BASE_URL") or self.BASE_URL
        ).rstrip("/")
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("FXMACRODATA_TIMEOUT", "30"))
        )
        self._transport = transport

    @property
    def has_api_key(self) -> bool:
        return self.api_key is not None

    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {}
        return {"X-API-Key": self.api_key}

    async def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        query = {key: value for key, value in params.items() if value is not None}
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport

        try:
            async with httpx.AsyncClient(**kwargs) as client:
                response = await client.get(
                    f"{self.base_url}{path}",
                    params=query,
                    headers=self._headers(),
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise FXMacroDataRequestError(
                f"FXMacroData request failed ({exc.response.status_code})",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.TimeoutException as exc:
            raise FXMacroDataRequestError("FXMacroData request timed out") from exc
        except httpx.RequestError as exc:
            raise FXMacroDataRequestError("FXMacroData request failed") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise FXMacroDataResponseError(
                "FXMacroData returned an invalid response"
            ) from exc

        if not isinstance(payload, dict):
            raise FXMacroDataResponseError("FXMacroData returned an invalid response")
        data = payload.get("data")
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise FXMacroDataResponseError("FXMacroData returned an invalid response")
        return payload

    async def get_announcements(
        self,
        currency: str,
        indicator: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch release rows for one currency and indicator."""
        return await self._get(
            f"/v1/announcements/{currency}/{indicator}",
            {
                "start_date": start_date,
                "end_date": end_date,
                "limit": limit,
                "offset": offset,
            },
        )

    async def get_calendar(
        self,
        currency: str,
        *,
        indicator: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        timezone: str | None = None,
    ) -> dict[str, Any]:
        """Fetch the forward release calendar for one currency."""
        return await self._get(
            f"/v1/calendar/{currency}",
            {
                "indicator": indicator,
                "start_date": start_date,
                "end_date": end_date,
                "timezone": timezone,
            },
        )

    async def close(self) -> None:
        """Compatibility no-op; each request owns its short-lived HTTP client."""
        return


# Export both spellings for callers that treat the acronym as a word.
FxMacroDataClient = FXMacroDataClient
