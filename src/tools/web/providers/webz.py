"""Webz.io News Search provider.

API contract: https://docs.webz.io/docs/webz/news-search. The adapter uses the
existing raw-httpx stack and reads ``WEBZ_API_KEY`` lazily so a missing key is a
per-call failure rather than a build-time crash. Provider response bodies are
never echoed into errors or logs because they may contain credentials.
"""

import logging
import os
import time
from typing import Any

import httpx
from langchain_core.tools import tool

from src.tools.web.providers._shared import (
    SNIPPET_MAX,
    lazy,
    normalize_time_range,
    request_json,
    time_range_to_start_date,
)
from src.tools.web.types import SearchResult

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.webz.io"
_SEARCH_PATH = "/api/news/context"
_TIMEOUT = 30.0


class WebzConfigurationError(ValueError):
    """Webz.io cannot be called because its required credential is absent."""


class WebzResponseError(ValueError):
    """Webz.io returned a malformed success response."""

    def __init__(self) -> None:
        super().__init__("Webz.io returned an invalid response")


def _require_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WebzResponseError()
    return value


def _require_string(value: Any, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise WebzResponseError()
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WebzResponseError()
    return value.strip() or None


def _as_filter_list(value: Any, field: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise TypeError(f"{field} must be a string or a list of strings")
    if not all(isinstance(item, str) for item in values):
        raise TypeError(f"{field} must contain only strings")
    cleaned = [item.strip() for item in values if item.strip()]
    return cleaned or None


def _normalize_result(raw: Any) -> SearchResult:
    item = _require_mapping(raw)
    article = _require_mapping(item.get("article"))
    chunk = _require_mapping(item.get("chunk"))
    metadata = _require_mapping(item.get("metadata"))

    url = _require_string(article.get("url"), allow_empty=False)
    title = _require_string(article.get("title", ""))
    content = _require_string(chunk.get("text"), allow_empty=False)
    published_date = _optional_string(article.get("published_at"))
    source = _optional_string(metadata.get("domain"))

    score = item.get("score")
    if score is not None:
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise WebzResponseError()
        if score < 0 or score > 10:
            raise WebzResponseError()
        score = float(score)

    return SearchResult(
        title=title,
        url=url,
        content=content,
        excerpts=(content,),
        published_date=published_date,
        score=score,
        result_type="news",
        source=source,
        metadata=metadata,
    )


def _normalize_search_response(
    payload: Any,
) -> tuple[list[SearchResult], dict[str, Any]]:
    envelope = _require_mapping(payload)
    raw_results = envelope.get("results")
    if not isinstance(raw_results, list):
        raise WebzResponseError()

    results = [_normalize_result(item) for item in raw_results]
    total_results = envelope.get("total_results", len(results))
    if (
        not isinstance(total_results, int)
        or isinstance(total_results, bool)
        or total_results < 0
    ):
        raise WebzResponseError()

    return results, {
        "total_results": total_results,
        "requests_left": envelope.get("requests_left"),
        "credits_used": envelope.get("credits_used"),
    }


class WebzAPI:
    """Minimal raw-httpx client for Webz.io News Search."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("WEBZ_API_KEY")
        if not self.api_key:
            raise WebzConfigurationError(
                "WEBZ_API_KEY not found in environment variables"
            )
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        time_range: str | None = None,
        language: list[str] | None = None,
        country: list[str] | None = None,
        source: list[str] | None = None,
        sentiment: list[str] | None = None,
        category: list[str] | None = None,
        published_from: str | None = None,
        published_to: str | None = None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        if not query.strip():
            raise ValueError("query is required")
        if len(query) > 750:
            raise ValueError("query must be at most 750 characters")

        canonical_range = normalize_time_range(time_range, provider="Webz.io")
        effective_from = published_from or time_range_to_start_date(
            canonical_range, fmt="%Y-%m-%d", provider="Webz.io"
        )
        filters: dict[str, Any] = {}
        if effective_from:
            filters["published_from"] = effective_from
        if published_to:
            filters["published_to"] = published_to
        for key, value in (
            ("language", language),
            ("country", country),
            ("domain", source),
            ("sentiment", sentiment),
            ("category", category),
        ):
            cleaned = _as_filter_list(value, key)
            if cleaned:
                filters[key] = cleaned

        payload: dict[str, Any] = {
            "query": query,
            "k": max(1, min(100, max_results)),
        }
        if filters:
            payload["filters"] = filters

        start_time = time.time()
        data = await request_json(
            "POST",
            f"{_BASE_URL}{_SEARCH_PATH}",
            provider="Webz.io",
            headers=self.headers,
            json_body=payload,
            timeout=_TIMEOUT,
        )
        results, metadata = _normalize_search_response(data)
        metadata["response_time"] = round(time.time() - start_time, 2)
        return results, metadata


def _search_error(
    message: str,
    query: str,
    *,
    http_status: int | None = None,
) -> tuple[str, dict[str, Any]]:
    artifact: dict[str, Any] = {
        "type": "web_search",
        "search_engine": "webz",
        "query": query,
        "results": [],
        "error": message,
    }
    if http_status is not None:
        artifact["http_status"] = http_status
    return f"Search failed: {message}", artifact


def _search_http_error(status: int) -> str:
    if status in (400, 422):
        return f"Webz.io rejected the request (HTTP {status})"
    if status in (401, 403):
        return f"Webz.io authentication failed (HTTP {status})"
    if status == 402:
        return "Webz.io quota exhausted (HTTP 402)"
    if status == 429:
        return "Webz.io rate limit exceeded (HTTP 429)"
    if status >= 500:
        return f"Webz.io service unavailable (HTTP {status})"
    return f"Webz.io request failed (HTTP {status})"


def build_web_search_tool(
    max_results: int = 10,
    default_time_range: str | None = None,
    verbose: bool = True,
):
    """Build a per-request Webz.io News Search tool.

    The API wrapper is created lazily so a missing ``WEBZ_API_KEY`` is a
    fail-closed tool result, never a build-time crash. ``verbose`` is accepted
    for the uniform builder interface; Webz.io results have no image variant.
    """
    _get_api_wrapper = lazy(WebzAPI)

    @tool(response_format="content_and_artifact")
    async def web_search(
        query: str,
        time_range: str | None = None,
        language: list[str] | None = None,
        country: list[str] | None = None,
        source: list[str] | None = None,
        sentiment: list[str] | None = None,
        category: list[str] | None = None,
        published_from: str | None = None,
        published_to: str | None = None,
    ) -> tuple[list[dict[str, Any]] | str, dict[str, Any]]:
        """Search current news with Webz.io's natural-language News Search API.

        Use when you need recent, source-backed news and matching article
        excerpts. The API searches the last 30 days by default.

        Args:
            query: Topic or question to search for.
            time_range: Optional recency filter: h, d, w, m, y (or words).
            language: Optional article language names, such as ['english'].
            country: Optional source-country codes, such as ['US', 'GB'].
            source: Optional source domains, such as ['reuters.com'].
            sentiment: Optional values: positive, negative, or neutral.
            category: Optional Webz.io categories, such as ['business'].
            published_from: Optional start date (YYYY-MM-DD).
            published_to: Optional end date (YYYY-MM-DD).
        """
        effective_range = time_range or default_time_range
        try:
            results, metadata = await _get_api_wrapper().search(
                query,
                max_results=max_results,
                time_range=effective_range,
                language=language,
                country=country,
                source=source,
                sentiment=sentiment,
                category=category,
                published_from=published_from,
                published_to=published_to,
            )
        except WebzConfigurationError:
            logger.error("Webz.io search is not configured")
            return _search_error(
                "Webz.io is not configured (WEBZ_API_KEY missing)", query
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else 0
            logger.error("Webz.io search failed (HTTP %s)", status)
            return _search_error(_search_http_error(status), query, http_status=status)
        except httpx.TimeoutException:
            logger.error("Webz.io search timed out")
            return _search_error("Webz.io request timed out", query)
        except (httpx.HTTPError, WebzResponseError, ValueError, TypeError):
            logger.error("Webz.io search returned an invalid or failed response")
            return _search_error("Webz.io returned an invalid response", query)

        return [result.as_dict() for result in results], {
            "type": "web_search",
            "search_engine": "webz",
            "query": query,
            "response_time": metadata.get("response_time", 0),
            "total_results": metadata.get("total_results", len(results)),
            "requests_left": metadata.get("requests_left"),
            "credits_used": metadata.get("credits_used"),
            "results": [
                {
                    "title": result.title,
                    "url": result.url,
                    "favicon": result.favicon or "",
                    "snippet": (
                        result.excerpts[0] if result.excerpts else result.content
                    )[:SNIPPET_MAX],
                    "source": result.source or "",
                    "date": result.published_date or "",
                    "score": result.score,
                }
                for result in results
            ],
        }

    return web_search
