"""AnySearch provider: optional web search and page extraction.

Contract source: https://anysearch.com/docs/api-endpoints. ``POST /v1/search``
returns a shared ``{code, message, request_id, data}`` envelope; ``POST
/v1/extract`` accepts exactly one URL and returns cleaned content. Both calls
use the existing httpx stack and optional Bearer authentication from
``ANYSEARCH_API_KEY``. Provider responses are never echoed into errors or logs:
the 402 quota flow may contain generated credentials in its message.
"""

import logging
import os
from typing import Any, Literal

import httpx
from langchain_core.tools import tool

from src.tools.web.providers._shared import (
    SNIPPET_MAX,
    lazy,
    request_json,
    result_card,
)
from src.tools.web.types import (
    FetchRequest,
    FetchResponse,
    FetchResult,
    SearchResult,
    WebError,
    WebErrorType,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.anysearch.com"
_TIMEOUT = 60.0


class AnySearchResponseError(ValueError):
    """AnySearch returned a malformed success envelope."""

    def __init__(self) -> None:
        super().__init__("AnySearch returned an invalid response")


def _require_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AnySearchResponseError()
    return value


def _require_string(value: Any, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise AnySearchResponseError()
    return value


def _normalize_search_response(
    payload: Any,
) -> tuple[list[SearchResult], dict[str, Any]]:
    envelope = _require_mapping(payload)
    if envelope.get("code") != 0:
        raise AnySearchResponseError()

    request_id = _require_string(envelope.get("request_id"), allow_empty=False)
    data = _require_mapping(envelope.get("data"))
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        raise AnySearchResponseError()

    results: list[SearchResult] = []
    for raw in raw_results:
        item = _require_mapping(raw)
        url = _require_string(item.get("url"), allow_empty=False)
        title = _require_string(item.get("title", ""))
        snippet = _require_string(item.get("snippet", ""))
        content = _require_string(item.get("content", ""))
        results.append(
            SearchResult(
                title=title,
                url=url,
                content=content or snippet,
                excerpts=(snippet,) if snippet else (),
            )
        )

    metadata = _require_mapping(data.get("metadata"))
    total_results = metadata.get("total_results", len(results))
    search_time_ms = metadata.get("search_time_ms", 0)
    if not isinstance(total_results, int) or isinstance(total_results, bool):
        raise AnySearchResponseError()
    if not isinstance(search_time_ms, (int, float)) or isinstance(search_time_ms, bool):
        raise AnySearchResponseError()
    if total_results < 0 or search_time_ms < 0:
        raise AnySearchResponseError()

    return results, {
        "request_id": request_id,
        "total_results": total_results,
        "search_time_ms": search_time_ms,
    }


def _normalize_extract_response(payload: Any) -> dict[str, str]:
    envelope = _require_mapping(payload)
    if envelope.get("code") != 0:
        raise AnySearchResponseError()
    data = _require_mapping(envelope.get("data"))
    return {
        "url": _require_string(data.get("url"), allow_empty=False),
        "title": _require_string(data.get("title", "")),
        "content": _require_string(data.get("content")),
    }


class AnySearchAPI:
    """Minimal raw-httpx client for AnySearch /v1/search and /v1/extract."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANYSEARCH_API_KEY", "")
        self.base_url = _BASE_URL
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    async def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return await request_json(
            "POST",
            f"{self.base_url}{path}",
            provider="AnySearch",
            headers=self.headers,
            json_body=body,
            params=None,
            timeout=_TIMEOUT,
        )

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        tag: str | None = None,
        params: dict[str, Any] | None = None,
        zone: Literal["cn", "intl"] | None = None,
        language: str | None = None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        if not query.strip():
            raise ValueError("query is required")
        body: dict[str, Any] = {
            "query": query,
            "max_results": max(1, min(10, max_results)),
            "format": "json",
        }
        if tag:
            body["tag"] = tag
        if params:
            body["params"] = params
        if zone:
            body["zone"] = zone
        if language:
            body["language"] = language
        return _normalize_search_response(await self._request("/v1/search", body))

    async def extract(self, url: str) -> dict[str, str]:
        if not url.strip():
            raise ValueError("url is required")
        return _normalize_extract_response(
            await self._request("/v1/extract", {"url": url})
        )


def _fetch_http_error(status: int) -> WebError:
    """Map AnySearch's provider-API status without exposing its response body."""
    if status == 400:
        return WebError(
            type=WebErrorType.UNSUPPORTED_URL,
            message="AnySearch rejected the URL",
            http_status=status,
            retryable=False,
            native_kind="invalid_extract_url",
        )
    if status in (401, 403):
        return WebError(
            type=WebErrorType.FORBIDDEN,
            message=f"AnySearch authentication failed (HTTP {status})",
            http_status=status,
            retryable=True,
        )
    if status == 402:
        return WebError(
            type=WebErrorType.BUDGET_EXCEEDED,
            message="AnySearch quota exhausted (HTTP 402)",
            http_status=status,
            retryable=True,
        )
    if status == 429:
        return WebError(
            type=WebErrorType.RATE_LIMITED,
            message="AnySearch rate limit exceeded (HTTP 429)",
            http_status=status,
            retryable=True,
        )
    if status == 422:
        return WebError(
            type=WebErrorType.PROVIDER_ERROR,
            message="AnySearch could not extract the URL (HTTP 422)",
            http_status=status,
            retryable=True,
            provider_fault=False,
            native_kind="extract_failed",
        )
    return WebError(
        type=WebErrorType.PROVIDER_ERROR,
        message=f"AnySearch request failed (HTTP {status})",
        http_status=status,
        retryable=True,
    )


class AnySearchFetchAdapter:
    """FetchAdapter over AnySearch /v1/extract (one URL per request)."""

    name = "anysearch"

    async def fetch(
        self, req: FetchRequest, native_params: dict[str, Any]
    ) -> FetchResponse:
        api = AnySearchAPI()

        by_url: dict[str, FetchResult] = {}
        for url in req.urls:
            try:
                data = await api.extract(url)
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else 0
                logger.error("AnySearch extract failed (HTTP %s)", status)
                by_url[url] = FetchResult(url=url, error=_fetch_http_error(status))
            except httpx.TimeoutException:
                logger.error("AnySearch extract timed out")
                by_url[url] = FetchResult(
                    url=url,
                    error=WebError(
                        type=WebErrorType.TIMEOUT,
                        message="AnySearch request timed out",
                        retryable=True,
                        provider_fault=False,
                    ),
                )
            except (httpx.HTTPError, AnySearchResponseError, ValueError, TypeError):
                logger.error("AnySearch extract returned an invalid or failed response")
                by_url[url] = FetchResult(
                    url=url,
                    error=WebError(
                        type=WebErrorType.PROVIDER_ERROR,
                        message="AnySearch returned an invalid response",
                        retryable=True,
                    ),
                )
            else:
                content = data["content"]
                if not content.strip():
                    by_url[url] = FetchResult(
                        url=url,
                        error=WebError(
                            type=WebErrorType.EMPTY,
                            message="AnySearch returned empty content",
                        ),
                    )
                else:
                    by_url[url] = FetchResult(
                        url=data["url"] or url,
                        title=data["title"] or None,
                        markdown=content,
                    )

        return FetchResponse(
            results=[by_url[u] for u in req.urls],
            provider=self.name,
        )


def _search_error(message: str, query: str) -> tuple[str, dict[str, Any]]:
    return (
        f"Search failed: {message}",
        {
            "type": "web_search",
            "search_engine": "anysearch",
            "query": query,
            "results": [],
            "error": message,
        },
    )


def _search_http_error(status: int) -> str:
    if status == 400:
        return "AnySearch rejected the request (HTTP 400)"
    if status in (401, 403):
        return f"AnySearch authentication failed (HTTP {status})"
    if status == 402:
        return "AnySearch quota exhausted (HTTP 402)"
    if status == 415:
        return "AnySearch rejected the request content type (HTTP 415)"
    if status == 429:
        return "AnySearch rate limit exceeded (HTTP 429)"
    if status >= 500:
        return f"AnySearch service unavailable (HTTP {status})"
    return f"AnySearch request failed (HTTP {status})"


def build_web_search_tool(
    max_results: int = 10,
    default_time_range: str | None = None,
    verbose: bool = True,
):
    """Build a per-request AnySearch web_search tool.

    ``tag`` and ``params`` are passed through so vertical capabilities remain
    provider-native without hardcoding a snapshot of AnySearch's catalog.
    AnySearch has no documented time-range or image parameters; the uniform
    builder arguments are accepted for the shared dispatcher and ignored.
    """
    _get_api_wrapper = lazy(AnySearchAPI)

    @tool(response_format="content_and_artifact")
    async def web_search(
        query: str,
        tag: str | None = None,
        params: dict[str, Any] | None = None,
        zone: Literal["cn", "intl"] | None = None,
        language: str | None = None,
    ) -> tuple[list[dict[str, Any]] | str, dict[str, Any]]:
        """Search the web with AnySearch, including optional vertical routing.

        Args:
            query: Search query to execute.
            tag: Optional AnySearch sub-domain tag such as ``code.doc``.
            params: Optional provider-native parameters for the selected tag.
            zone: Optional region, ``cn`` or ``intl``.
            language: Optional preferred language such as ``zh-CN`` or ``en``.

        Returns:
            Page dictionaries with ``title``, ``url``, and ``content``, or a
            ``Search failed: ...`` string when the request fails.
        """
        try:
            results, metadata = await _get_api_wrapper().search(
                query,
                max_results=max_results,
                tag=tag,
                params=params,
                zone=zone,
                language=language,
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            logger.error("AnySearch search failed (HTTP %s)", status)
            return _search_error(_search_http_error(status), query)
        except httpx.TimeoutException:
            logger.error("AnySearch search timed out")
            return _search_error("AnySearch request timed out", query)
        except (httpx.HTTPError, AnySearchResponseError, ValueError, TypeError):
            logger.error("AnySearch search returned an invalid or failed response")
            return _search_error("AnySearch returned an invalid response", query)

        content = [result.as_dict() for result in results]
        artifact = {
            "type": "web_search",
            "search_engine": "anysearch",
            "query": query,
            "request_id": metadata.get("request_id", ""),
            "total_results": metadata.get("total_results", len(results)),
            "response_time": metadata.get("search_time_ms", 0) / 1000,
            "results": [
                result_card(
                    title=result.title,
                    url=result.url,
                    snippet=(result.excerpts[0] if result.excerpts else result.content)[
                        :SNIPPET_MAX
                    ],
                )
                for result in results
            ],
        }
        return content, artifact

    return web_search
