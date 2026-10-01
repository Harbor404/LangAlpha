"""Webz.io News Search provider contract tests.

The API contract is pinned against https://docs.webz.io/docs/webz/news-search:
``POST /api/news/context`` accepts ``query``, ``k``, and ``filters`` and
returns ``results[].{score, article, chunk, metadata}``. Provider failures are
normalized here and must never expose raw response bodies or credentials.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest

from src.tools.web.providers import webz
from src.tools.web.types import SearchResult

SEARCH_URL = "https://api.webz.io/api/news/context"
ARTICLE_URL = "https://news.example/article"


def _search_payload() -> dict:
    return {
        "query": "central bank interest rate decision",
        "total_results": 1,
        "results": [
            {
                "score": 7.1,
                "article": {
                    "article_id": "article-123",
                    "url": ARTICLE_URL,
                    "title": "Central bank holds rates",
                    "published_at": "2026-08-27T07:07:00.000+03:00",
                    "summary": "The bank left rates unchanged.",
                },
                "chunk": {
                    "chunk_id": "article-123_6",
                    "chunk_index": 6,
                    "text": "The central bank kept its policy rate unchanged.",
                },
                "metadata": {
                    "language": "english",
                    "country": "US",
                    "category": ["Economy, Business and Finance"],
                    "sentiment": "positive",
                    "domain": "news.example",
                    "source_type": "newsroom",
                    "topic": ["monetary policy"],
                    "ticker": ["USD"],
                    "domain_rank": 193,
                },
            }
        ],
        "requests_left": 99,
        "credits_used": 1,
    }


def _http_status_error(
    status: int, body: str, url: str = SEARCH_URL
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", url)
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(
        f"Webz.io API error {status}: {body}", request=request, response=response
    )


def test_webz_is_discovered_without_disturbing_existing_providers():
    from src.tools.web.manifest import (
        CAPABILITY_SEARCH,
        get_capability,
        get_web_provider_spec,
        providers_with_capability,
    )
    from src.tools.web.search import _PROVIDER_BUILDERS

    spec = get_web_provider_spec("webz")
    assert spec is not None
    assert spec.display_name == "Webz.io"
    assert spec.env_key == "WEBZ_API_KEY"
    assert get_capability("webz", CAPABILITY_SEARCH).tracking_name == "WebzSearchTool"
    assert "webz" in providers_with_capability(CAPABILITY_SEARCH)
    assert "webz" in _PROVIDER_BUILDERS
    assert {"tavily", "serper", "bocha", "exa", "parallel"} <= set(_PROVIDER_BUILDERS)


@pytest.mark.asyncio
async def test_search_maps_native_request_and_normalizes_results(monkeypatch):
    calls = []

    async def fake_request_json(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _search_payload()

    monkeypatch.setattr(webz, "request_json", fake_request_json)
    api = webz.WebzAPI(api_key="test-key")
    expected_from = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")

    results, metadata = await api.search(
        "central bank interest rate decision",
        max_results=500,
        time_range="week",
        language=["english"],
        country=["US", "GB"],
        source=["news.example", "finance.example"],
        sentiment=["positive", "neutral"],
        category=["business"],
        published_to="2026-09-01",
    )

    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == SEARCH_URL
    assert kwargs["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert kwargs["json_body"] == {
        "query": "central bank interest rate decision",
        "k": 100,
        "filters": {
            "published_from": expected_from,
            "published_to": "2026-09-01",
            "language": ["english"],
            "country": ["US", "GB"],
            "domain": ["news.example", "finance.example"],
            "sentiment": ["positive", "neutral"],
            "category": ["business"],
        },
    }
    assert results == [
        SearchResult(
            title="Central bank holds rates",
            url=ARTICLE_URL,
            content="The central bank kept its policy rate unchanged.",
            excerpts=("The central bank kept its policy rate unchanged.",),
            published_date="2026-08-27T07:07:00.000+03:00",
            score=7.1,
            result_type="news",
            source="news.example",
            metadata=_search_payload()["results"][0]["metadata"],
        )
    ]
    assert metadata["total_results"] == 1
    assert metadata["requests_left"] == 99
    assert metadata["credits_used"] == 1


@pytest.mark.asyncio
async def test_search_tool_returns_standard_content_and_artifact(monkeypatch):
    result = SearchResult(
        title="Central bank holds rates",
        url=ARTICLE_URL,
        content="The central bank kept its policy rate unchanged.",
        excerpts=("The central bank kept its policy rate unchanged.",),
        published_date="2026-08-27T07:07:00.000+03:00",
        score=7.1,
        result_type="news",
        source="news.example",
        metadata={"sentiment": "positive"},
    )
    api_search = AsyncMock(return_value=([result], {"total_results": 1}))
    monkeypatch.setenv("WEBZ_API_KEY", "test-key")
    monkeypatch.setattr(webz.WebzAPI, "search", api_search)

    tool = webz.build_web_search_tool(max_results=5)
    content, artifact = await tool.coroutine(query="central bank rates")

    assert content == [
        {
            "type": "news",
            "title": "Central bank holds rates",
            "url": ARTICLE_URL,
            "content": "The central bank kept its policy rate unchanged.",
            "date": "2026-08-27T07:07:00.000+03:00",
            "score": 7.1,
            "source": "news.example",
            "metadata": {"sentiment": "positive"},
        }
    ]
    assert artifact["type"] == "web_search"
    assert artifact["search_engine"] == "webz"
    assert artifact["query"] == "central bank rates"
    assert artifact["total_results"] == 1
    assert artifact["results"] == [
        {
            "title": "Central bank holds rates",
            "url": ARTICLE_URL,
            "favicon": "",
            "snippet": "The central bank kept its policy rate unchanged.",
            "source": "news.example",
            "date": "2026-08-27T07:07:00.000+03:00",
            "score": 7.1,
        }
    ]


@pytest.mark.asyncio
async def test_missing_api_key_is_disabled_before_network(monkeypatch):
    async def unexpected_request(*args, **kwargs):
        raise AssertionError("Webz.io must not be called without WEBZ_API_KEY")

    monkeypatch.delenv("WEBZ_API_KEY", raising=False)
    monkeypatch.setattr(webz, "request_json", unexpected_request)

    tool = webz.build_web_search_tool(max_results=5)
    content, artifact = await tool.coroutine(query="must not run")

    assert isinstance(content, str)
    assert "not configured" in artifact["error"].lower()
    assert artifact["results"] == []


@pytest.mark.asyncio
async def test_http_error_is_generic_and_redacted(monkeypatch, caplog):
    secret = "token=super-secret&password=do-not-return"

    async def fail_request(*args, **kwargs):
        raise _http_status_error(403, secret)

    monkeypatch.setenv("WEBZ_API_KEY", "test-key")
    monkeypatch.setattr(webz, "request_json", fail_request)

    with caplog.at_level("ERROR"):
        tool = webz.build_web_search_tool(max_results=5)
        content, artifact = await tool.coroutine(query="forbidden test")

    rendered = f"{content!r} {artifact!r} {caplog.text}"
    assert "super-secret" not in rendered
    assert "do-not-return" not in rendered
    assert "authentication failed" in artifact["error"].lower()
    assert artifact["results"] == []


@pytest.mark.asyncio
async def test_timeout_is_fail_closed(monkeypatch):
    async def timeout_request(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out at https://api.webz.io")

    monkeypatch.setenv("WEBZ_API_KEY", "test-key")
    monkeypatch.setattr(webz, "request_json", timeout_request)

    tool = webz.build_web_search_tool(max_results=5)
    content, artifact = await tool.coroutine(query="timeout test")

    assert isinstance(content, str)
    assert "timed out" in artifact["error"].lower()
    assert artifact["results"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"results": "not-a-list"},
        {"results": [{"article": {}, "chunk": {}, "metadata": {}}]},
        {"results": [{"article": {"url": ARTICLE_URL}, "chunk": {"text": "x"}}]},
    ],
)
async def test_invalid_response_is_fail_closed(monkeypatch, payload):
    async def fake_request(*args, **kwargs):
        return payload

    monkeypatch.setenv("WEBZ_API_KEY", "test-key")
    monkeypatch.setattr(webz, "request_json", fake_request)

    tool = webz.build_web_search_tool(max_results=5)
    content, artifact = await tool.coroutine(query="invalid response")

    assert isinstance(content, str)
    assert "invalid response" in artifact["error"].lower()
    assert artifact["results"] == []
