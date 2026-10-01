#!/usr/bin/env python3
"""Macro MCP Server.

Raw FMP macro data plus optional FXMacroData non-US releases and calendars —
via MCP. Payloads stay vendor-native inside `data`; the envelope around them is
the standard market-data contract (AGENT_CONTRACT.md).

Tools:
- get_economic_indicator: Time series for GDP, CPI, unemployment, etc.
- get_economic_calendar: Upcoming macro events with prior/estimate/actual values
- get_treasury_rates: Full yield curve (1M to 30Y)
- get_market_risk_premium: Risk premium by country for CAPM/WACC
- get_earnings_calendar: All companies reporting in a date range
- get_macro_announcements: Non-US macro releases, optionally via FXMacroData
- get_macro_release_calendar: Forward macro release calendar via FXMacroData
"""

# NOTE: Tool docstrings in this file are hand-tuned agent prompt surface (parsed
# into agent prompts and generated sandbox wrappers) and are content-pinned by
# tests/unit/mcp_servers/test_agent_contract.py. Read mcp_servers/AGENT_CONTRACT.md
# before editing; intentional changes must regenerate agent_docstring_lock.json.

from __future__ import annotations

try:
    from _bootstrap import MCPServer  # script launch: mcp_servers/ is sys.path[0]
except ModuleNotFoundError:  # imported as a package module (tests)
    from mcp_servers._bootstrap import MCPServer


from data_client.fmp import fmp_lifespan, get_fmp_client
from data_client.fxmacrodata import (
    FXMacroDataInvalidArgument,
    FXMacroDataUnavailable,
    get_fxmacrodata_source,
)
from mcp_servers._envelope import error_from_exception, make_error, make_response
from mcp_servers._schemas import (
    NULLABLE_STR,
    RECORDS,
    STR,
    described,
    envelope_schema,
    output_model,
)

mcp = MCPServer("MacroMCP", lifespan=fmp_lifespan)

_SOURCE = "fmp"
_CLIENT_UNAVAILABLE = "FMP client is unavailable"
_UPSTREAM_FAILED = "FMP request failed"
_FX_SOURCE = "fxmacrodata"
_FX_CLIENT_UNAVAILABLE = "FXMacroData client is unavailable"
_FX_UPSTREAM_FAILED = "FXMacroData request failed"


_OUT_GET_ECONOMIC_INDICATOR = output_model(
    "GetEconomicIndicatorOut",
    envelope_schema(
        RECORDS,
        echo={
            "data_type": STR,
            "indicator": described(STR, "Echoed indicator name."),
        },
    ),
)


@mcp.tool()
async def get_economic_indicator(
    name: str,
    limit: int = 50,
) -> _OUT_GET_ECONOMIC_INDICATOR:
    """Fetch an economic indicator time series — GDP growth for the macro
    outlook, CPI/inflation for discount-rate assumptions, or unemployment and
    the Fed funds rate for economic context.

    Args:
        name: Indicator — "GDP", "CPI", "unemploymentRate", "federalFundsRate",
            "inflationRate", "retailSales", "consumerSentiment", "nonFarmPayrolls".
        limit: Number of observations to fetch (default 50).

    Returns:
        dict: {count, data, source, data_type, indicator}. data is a list of
        observations; count is the observation total. Fields: date, value (and
        for some series, name). Field names are FMP-native camelCase; date is
        "YYYY-MM-DD"; observations are newest-first as returned by FMP. On error:
        {error: <code>, detail, indicator}.
    """
    try:
        client = await get_fmp_client()
    except Exception:  # noqa: BLE001
        return make_error("client_unavailable", _CLIENT_UNAVAILABLE, indicator=name)

    try:
        data = await client.get_economic_indicators(name, limit=limit)

        return make_response(
            data or [],
            source=_SOURCE,
            data_type="economic_indicator",
            indicator=name,
        )

    except Exception as e:  # noqa: BLE001
        return error_from_exception(e, _UPSTREAM_FAILED, indicator=name)


_OUT_GET_ECONOMIC_CALENDAR = output_model(
    "GetEconomicCalendarOut",
    envelope_schema(
        RECORDS,
        echo={
            "data_type": STR,
            "from_date": NULLABLE_STR,
            "to_date": NULLABLE_STR,
        },
    ),
)


@mcp.tool()
async def get_economic_calendar(
    from_date: str | None = None,
    to_date: str | None = None,
) -> _OUT_GET_ECONOMIC_CALENDAR:
    """Fetch economic events with prior, estimate, and actual values — build a
    catalyst calendar, generate a morning note, or track Fed meetings, jobs
    reports, and CPI releases.

    Args:
        from_date: Start date "YYYY-MM-DD" (default: today).
        to_date: End date "YYYY-MM-DD" (default: 7 days from today).

    Returns:
        dict: {count, data, source, data_type, from_date, to_date}. data is a
        list of events; count is the event total. Fields: date, country, event,
        currency, previous, estimate, actual, change, changePercentage, impact,
        unit. actual/estimate/previous are null for events not yet released.
        Field names are FMP-native camelCase; date is "YYYY-MM-DD HH:MM:SS";
        order is as returned by FMP. On error: {error: <code>, detail}.
    """
    try:
        client = await get_fmp_client()
    except Exception:  # noqa: BLE001
        return make_error("client_unavailable", _CLIENT_UNAVAILABLE)

    try:
        data = await client.get_economic_calendar(from_date=from_date, to_date=to_date)

        return make_response(
            data or [],
            source=_SOURCE,
            data_type="economic_calendar",
            from_date=from_date,
            to_date=to_date,
        )

    except Exception as e:  # noqa: BLE001
        return error_from_exception(e, _UPSTREAM_FAILED)


_OUT_GET_TREASURY_RATES = output_model(
    "GetTreasuryRatesOut",
    envelope_schema(
        RECORDS,
        echo={
            "data_type": STR,
            "from_date": NULLABLE_STR,
            "to_date": NULLABLE_STR,
        },
    ),
)


@mcp.tool()
async def get_treasury_rates(
    from_date: str | None = None,
    to_date: str | None = None,
) -> _OUT_GET_TREASURY_RATES:
    """Fetch US Treasury rates across the full yield curve (1M to 30Y) — get the
    risk-free rate for DCF/WACC (typically 10Y), read the curve shape, or track
    rate trends.

    Args:
        from_date: Start date "YYYY-MM-DD" (default: recent data).
        to_date: End date "YYYY-MM-DD" (default: today).

    Returns:
        dict: {count, data, source, data_type, from_date, to_date}. data is a
        list of daily records; count is the record total. Each record has a date
        plus per-tenor rate columns: month1, month2, month3, month6, year1,
        year2, year3, year5, year7, year10, year20, year30. Rates are percent
        (4.5 = 4.5%). Field names are FMP-native camelCase; date is "YYYY-MM-DD";
        records are newest-first as returned by FMP. On error:
        {error: <code>, detail}.
    """
    try:
        client = await get_fmp_client()
    except Exception:  # noqa: BLE001
        return make_error("client_unavailable", _CLIENT_UNAVAILABLE)

    try:
        data = await client.get_treasury_rates(from_date=from_date, to_date=to_date)

        return make_response(
            data or [],
            source=_SOURCE,
            data_type="treasury_rates",
            from_date=from_date,
            to_date=to_date,
        )

    except Exception as e:  # noqa: BLE001
        return error_from_exception(e, _UPSTREAM_FAILED)


_OUT_GET_MARKET_RISK_PREMIUM = output_model(
    "GetMarketRiskPremiumOut",
    envelope_schema(RECORDS, echo={"data_type": STR}),
)


@mcp.tool()
async def get_market_risk_premium() -> _OUT_GET_MARKET_RISK_PREMIUM:
    """Fetch market risk premium by country for CAPM/WACC — get the equity risk
    premium for a DCF cost of equity, or compare premiums across markets.

    Returns:
        dict: {count, data, source, data_type}. data is a list of country
        records; count is the record total. Fields: country, continent,
        countryRiskPremium, totalEquityRiskPremium. Premiums are percent
        (5.5 = 5.5%). Field names are FMP-native camelCase; order is by country
        as returned by FMP (not time-ordered). On error: {error: <code>, detail}.
    """
    try:
        client = await get_fmp_client()
    except Exception:  # noqa: BLE001
        return make_error("client_unavailable", _CLIENT_UNAVAILABLE)

    try:
        data = await client.get_market_risk_premium()

        return make_response(
            data or [],
            source=_SOURCE,
            data_type="market_risk_premium",
        )

    except Exception as e:  # noqa: BLE001
        return error_from_exception(e, _UPSTREAM_FAILED)


_OUT_GET_EARNINGS_CALENDAR = output_model(
    "GetEarningsCalendarOut",
    envelope_schema(
        RECORDS,
        echo={"data_type": STR, "from_date": STR, "to_date": STR},
    ),
)


@mcp.tool()
async def get_earnings_calendar(
    from_date: str,
    to_date: str,
) -> _OUT_GET_EARNINGS_CALENDAR:
    """Fetch the earnings calendar for all companies reporting in a date range —
    build a catalyst calendar, generate a morning note, or track earnings-season
    volume.

    Args:
        from_date: Start date "YYYY-MM-DD".
        to_date: End date "YYYY-MM-DD".

    Returns:
        dict: {count, data, source, data_type, from_date, to_date}. data is a
        list of reporter records; count is the record total. Fields: symbol,
        date, epsActual, epsEstimated, revenueActual, revenueEstimated,
        lastUpdated. actual fields are null before a company reports. Field names
        are FMP-native camelCase; date is "YYYY-MM-DD"; order is as returned by
        FMP. On error: {error: <code>, detail}.
    """
    try:
        client = await get_fmp_client()
    except Exception:  # noqa: BLE001
        return make_error("client_unavailable", _CLIENT_UNAVAILABLE)

    try:
        data = await client.get_earnings_calendar_by_date(
            from_date=from_date, to_date=to_date
        )

        return make_response(
            data or [],
            source=_SOURCE,
            data_type="earnings_calendar",
            from_date=from_date,
            to_date=to_date,
        )

    except Exception as e:  # noqa: BLE001
        return error_from_exception(e, _UPSTREAM_FAILED)


_OUT_GET_MACRO_ANNOUNCEMENTS = output_model(
    "GetMacroAnnouncementsOut",
    envelope_schema(
        RECORDS,
        echo={
            "data_type": STR,
            "currency": described(STR, "Echoed ISO 4217 currency code."),
            "indicator": described(STR, "Echoed FXMacroData indicator slug."),
            "start_date": NULLABLE_STR,
            "end_date": NULLABLE_STR,
        },
    ),
)


@mcp.tool()
async def get_macro_announcements(
    currency: str,
    indicator: str,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> _OUT_GET_MACRO_ANNOUNCEMENTS:
    """Fetch official FXMacroData macro releases for one currency — track
    non-US rate decisions, inflation, GDP, payrolls, and yield series.

    Args:
        currency: ISO 4217 code, e.g. "AUD", "EUR", "GBP", "USD".
        indicator: FXMacroData slug, e.g. "policy_rate", "inflation", "gdp".
        start_date: Optional start date "YYYY-MM-DD".
        end_date: Optional end date "YYYY-MM-DD".
        limit: Rows per page (1-100; default 20).
        offset: Zero-based row offset for paging (default 0).

    Returns:
        dict: {count, data, source, data_type, currency, indicator,
        start_date, end_date, pagination}. data rows preserve FXMacroData
        fields and add value/announcement_datetime_utc aliases; nulls stay
        null. USD works without a key; other currencies require
        FXMACRODATA_API_KEY or return client_unavailable. On error:
        {error: <code>, detail, currency, indicator}.
    """
    try:
        source = await get_fxmacrodata_source()
    except Exception:  # noqa: BLE001
        return make_error(
            "client_unavailable",
            _FX_CLIENT_UNAVAILABLE,
            currency=currency,
            indicator=indicator,
        )

    try:
        payload = await source.get_macro_announcements(
            currency,
            indicator,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:  # noqa: BLE001
        # Tests and the server can import the data client under both ``src.``
        # and bare ``data_client`` identities; compare the stable class name so
        # the same failure maps consistently across those import paths.
        if isinstance(exc, FXMacroDataUnavailable) or type(exc).__name__ == (
            "FXMacroDataUnavailable"
        ):
            return make_error(
                "client_unavailable",
                str(exc),
                currency=currency,
                indicator=indicator,
            )
        if isinstance(exc, FXMacroDataInvalidArgument) or type(exc).__name__ == (
            "FXMacroDataInvalidArgument"
        ):
            return make_error(
                "invalid_argument",
                str(exc),
                currency=currency,
                indicator=indicator,
            )
        return error_from_exception(
            exc,
            _FX_UPSTREAM_FAILED,
            currency=currency,
            indicator=indicator,
        )

    return make_response(
        payload.get("data", []),
        source=_FX_SOURCE,
        data_type="macro_announcements",
        currency=payload.get("currency", currency),
        indicator=payload.get("indicator", indicator),
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        pagination=payload.get("pagination"),
    )


_OUT_GET_MACRO_RELEASE_CALENDAR = output_model(
    "GetMacroReleaseCalendarOut",
    envelope_schema(
        RECORDS,
        echo={
            "data_type": STR,
            "currency": described(STR, "Echoed ISO 4217 currency code."),
            "indicator": NULLABLE_STR,
            "start_date": NULLABLE_STR,
            "end_date": NULLABLE_STR,
            "timezone": NULLABLE_STR,
        },
    ),
)


@mcp.tool()
async def get_macro_release_calendar(
    currency: str,
    indicator: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    timezone: str | None = None,
) -> _OUT_GET_MACRO_RELEASE_CALENDAR:
    """Fetch the forward FXMacroData release calendar for one currency — find
    upcoming central-bank decisions, inflation, GDP, jobs, and bond releases.

    Args:
        currency: ISO 4217 code, e.g. "AUD", "EUR", "GBP", "USD".
        indicator: Optional FXMacroData slug filter, e.g. "inflation".
        start_date: Optional start date "YYYY-MM-DD".
        end_date: Optional end date "YYYY-MM-DD".
        timezone: Optional IANA timezone for returned local timestamps.

    Returns:
        dict: {count, data, source, data_type, currency, indicator, start_date,
        end_date, timezone}. data rows preserve FXMacroData calendar fields and
        add announcement_datetime_utc when a Unix timestamp is present;
        nulls stay null. USD works without a key; other currencies require
        FXMACRODATA_API_KEY or return client_unavailable. On error:
        {error: <code>, detail, currency}.
    """
    try:
        source = await get_fxmacrodata_source()
    except Exception:  # noqa: BLE001
        return make_error(
            "client_unavailable", _FX_CLIENT_UNAVAILABLE, currency=currency
        )

    try:
        payload = await source.get_macro_release_calendar(
            currency,
            indicator=indicator,
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
        )
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, FXMacroDataUnavailable) or type(exc).__name__ == (
            "FXMacroDataUnavailable"
        ):
            return make_error("client_unavailable", str(exc), currency=currency)
        if isinstance(exc, FXMacroDataInvalidArgument) or type(exc).__name__ == (
            "FXMacroDataInvalidArgument"
        ):
            return make_error("invalid_argument", str(exc), currency=currency)
        return error_from_exception(exc, _FX_UPSTREAM_FAILED, currency=currency)

    return make_response(
        payload.get("data", []),
        source=_FX_SOURCE,
        data_type="macro_release_calendar",
        currency=payload.get("currency", currency),
        indicator=payload.get("indicator", indicator),
        start_date=start_date,
        end_date=end_date,
        timezone=payload.get("timezone", timezone),
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
