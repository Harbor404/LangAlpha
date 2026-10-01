"""Optional FXMacroData source for non-US macro releases and calendars."""

from .client import (
    FXMacroDataClient,
    FXMacroDataError,
    FXMacroDataRequestError,
    FXMacroDataResponseError,
)
from .data_source import (
    ANNOUNCEMENT_CURRENCIES,
    CALENDAR_CURRENCIES,
    FXMacroDataInvalidArgument,
    FXMacroDataSource,
    FXMacroDataUnavailable,
    normalize_announcement,
    normalize_release,
)

__all__ = [
    "ANNOUNCEMENT_CURRENCIES",
    "CALENDAR_CURRENCIES",
    "FXMacroDataClient",
    "FXMacroDataError",
    "FXMacroDataInvalidArgument",
    "FXMacroDataRequestError",
    "FXMacroDataResponseError",
    "FXMacroDataSource",
    "FXMacroDataUnavailable",
    "FxMacroDataClient",
    "FxMacroDataSource",
    "get_fxmacrodata_source",
    "normalize_announcement",
    "normalize_release",
]


async def get_fxmacrodata_source() -> FXMacroDataSource:
    """Discover the optional source without requiring a key up front.

    The returned source permits keyless USD requests and fails closed for every
    other currency until ``FXMACRODATA_API_KEY`` is configured.
    """
    return FXMacroDataSource()


# Export both spellings for callers that treat the acronym as a word.
FxMacroDataClient = FXMacroDataClient
FxMacroDataSource = FXMacroDataSource
