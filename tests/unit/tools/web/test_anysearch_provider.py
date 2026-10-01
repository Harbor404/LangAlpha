"""AnySearch provider contract tests.

The API contract is pinned against https://anysearch.com/docs/api-endpoints:
POST /v1/search accepts query/max_results/tag/params/zone/language/format and
returns {code, message, request_id, data.results[], data.metadata}; POST
/v1/extract accepts exactly {"url": ...} and returns cleaned content.
Provider failures are normalized here, not leaked as raw vendor payloads.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.tools.web.providers import anysearch
from src.tools.web.types import FetchRequest, WebErrorType

SEARCH_URL = "https://api.anysearch.com/v1/search"
EXTRACT_URL = "https://api.anysearch.com/v1/extract"
URL_OK = "https://site.example/article"
REQUEST_ID = "7d6f4e91-2a83-4c5b-9f10-6e8a3d27b541"


def _search_payload() -> dict:
    return {
        "code": 0,
        "message": "success",
        "request_id": REQUEST_ID,
        "data": {
            "results": [
                {
                    "title": "AnySearch Result",
                    "url": URL_OK,
                    "snippet": "A short snippet.",
                    "content": "Full cleaned result content.",
                }
            ],
            "metadata": {"total_results": 1, "search_time_ms": 321},
        },
    }


def _extract_payload() -> dict:
    return {
        "code": 0,
        "message": "success",
        "request_id": REQUEST_ID,
        "data": {
            "url": URL_OK,
            "title": "Extracted title",
            "content": "# Extracted markdown",
        },
    }


def _http_status_error(
    status: int, body: str, url: str = SEARCH_URL
) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", url)
    response = httpx.Response(status, text=body, request=request)
    return httpx.HTTPStatusError(
        f"AnySearch API error {status}: {body}", request=request, response=response
    )


@pytest.mark.asyncio
async def test_search_maps_native_request_and_normalizes_results(monkeypatch):
    calls = []

    async def fake_request_json(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _search_payload()

    monkeypatch.setattr(anysearch, "request_json", fake_request_json)
    api = anysearch.AnySearchAPI(api_key="test-key")

    results, metadata = await api.search(
        "Go 1.26 release notes",
        max_results=99,
        tag="code.doc",
        params={"library": "golang"},
        zone="intl",
        language="en",
    )

    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == SEARCH_URL
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json_body"] == {
        "query": "Go 1.26 release notes",
        "max_results": 10,
        "format": "json",
        "tag": "code.doc",
        "params": {"library": "golang"},
        "zone": "intl",
        "language": "en",
    }
    assert results[0].title == "AnySearch Result"
    assert results[0].url == URL_OK
    assert results[0].content == "Full cleaned result content."
    assert results[0].excerpts == ("A short snippet.",)
    assert metadata["request_id"] == REQUEST_ID
    assert metadata["total_results"] == 1
    assert metadata["search_time_ms"] == 321


@pytest.mark.asyncio
async def test_search_tool_returns_standard_content_and_artifact(monkeypatch):
    from src.tools.web.types import SearchResult

    result = SearchResult(
        title="AnySearch Result",
        url=URL_OK,
        content="Full cleaned result content.",
        excerpts=("A short snippet.",),
    )
    api_search = AsyncMock(
        return_value=([result], {"request_id": REQUEST_ID, "total_results": 1})
    )
    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")
    monkeypatch.setattr(anysearch.AnySearchAPI, "search", api_search)

    tool = anysearch.build_web_search_tool(max_results=5)
    content, artifact = await tool.coroutine(query="test query", tag="code.doc")

    assert artifact["type"] == "web_search"
    assert artifact["search_engine"] == "anysearch"
    assert artifact["results"] == [
        {
            "title": "AnySearch Result",
            "url": URL_OK,
            "favicon": "",
            "snippet": "A short snippet.",
        }
    ]
    assert content == [
        {
            "type": "page",
            "title": "AnySearch Result",
            "url": URL_OK,
            "content": "Full cleaned result content.",
        }
    ]


@pytest.mark.asyncio
async def test_extract_maps_clean_content(monkeypatch):
    calls = []

    async def fake_request_json(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _extract_payload()

    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")
    monkeypatch.setattr(anysearch, "request_json", fake_request_json)

    response = await anysearch.AnySearchFetchAdapter().fetch(
        FetchRequest(urls=[URL_OK], mode="full"), {}
    )

    method, url, kwargs = calls[0]
    assert method == "POST"
    assert url == EXTRACT_URL
    assert kwargs["json_body"] == {"url": URL_OK}
    assert response.provider == "anysearch"
    assert response.results[0].ok
    assert response.results[0].url == URL_OK
    assert response.results[0].final_url == URL_OK
    assert response.results[0].title == "Extracted title"
    assert response.results[0].markdown == "# Extracted markdown"


@pytest.mark.asyncio
async def test_extract_preserves_requested_url_when_provider_reports_redirect(
    monkeypatch,
):
    requested_url = "https://site.example/requested"
    redirected_url = "https://cdn.example/final"

    async def fake_request_json(*args, **kwargs):
        return {
            "code": 0,
            "message": "success",
            "request_id": REQUEST_ID,
            "data": {
                "url": redirected_url,
                "title": "Redirected title",
                "content": "# Redirected content",
            },
        }

    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")
    monkeypatch.setattr(anysearch, "request_json", fake_request_json)

    response = await anysearch.AnySearchFetchAdapter().fetch(
        FetchRequest(urls=[requested_url]), {}
    )

    result = response.results[0]
    assert result.ok
    assert result.url == requested_url
    assert result.final_url == redirected_url


@pytest.mark.asyncio
async def test_anonymous_search_omits_authorization(monkeypatch):
    calls = []

    async def fake_request_json(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _search_payload()

    monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
    monkeypatch.setattr(anysearch, "request_json", fake_request_json)

    await anysearch.AnySearchAPI().search("anonymous query")

    headers = calls[0][2]["headers"]
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_anonymous_fetch_omits_authorization(monkeypatch):
    calls = []

    async def fake_request_json(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _extract_payload()

    monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
    monkeypatch.setattr(anysearch, "request_json", fake_request_json)

    response = await anysearch.AnySearchFetchAdapter().fetch(
        FetchRequest(urls=[URL_OK]), {}
    )

    headers = calls[0][2]["headers"]
    assert "Authorization" not in headers
    assert headers["Content-Type"] == "application/json"
    assert response.results[0].ok


@pytest.mark.asyncio
async def test_search_http_error_is_generic_and_redacted(monkeypatch, caplog):
    secret = "api_key=super-secret&password=do-not-return"

    async def fail_request(*args, **kwargs):
        raise _http_status_error(402, secret)

    monkeypatch.setattr(anysearch, "request_json", fail_request)
    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")

    with caplog.at_level("ERROR"):
        tool = anysearch.build_web_search_tool(max_results=5)
        content, artifact = await tool.coroutine(query="quota test")

    rendered = f"{content!r} {artifact!r}"
    assert "super-secret" not in rendered
    assert "do-not-return" not in rendered
    assert "super-secret" not in caplog.text
    assert "do-not-return" not in caplog.text
    assert "quota" in artifact["error"].lower()


@pytest.mark.asyncio
async def test_extract_http_error_is_generic_and_redacted(monkeypatch, caplog):
    secret = "api_key=super-secret&password=do-not-return"

    async def fail_request(*args, **kwargs):
        raise _http_status_error(403, secret, EXTRACT_URL)

    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")
    monkeypatch.setattr(anysearch, "request_json", fail_request)

    with caplog.at_level("ERROR"):
        response = await anysearch.AnySearchFetchAdapter().fetch(
            FetchRequest(urls=[URL_OK]), {}
        )

    error = response.results[0].error
    assert error.type == WebErrorType.FORBIDDEN
    assert "super-secret" not in error.message
    assert "do-not-return" not in error.message
    assert "super-secret" not in caplog.text
    assert "do-not-return" not in caplog.text


@pytest.mark.asyncio
async def test_timeout_is_fail_closed(monkeypatch):
    async def timeout_request(*args, **kwargs):
        raise httpx.ConnectTimeout("timed out at https://api.anysearch.com")

    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")
    monkeypatch.setattr(anysearch, "request_json", timeout_request)

    tool = anysearch.build_web_search_tool(max_results=5)
    _content, artifact = await tool.coroutine(query="timeout test")
    assert "timed out" in artifact["error"].lower()

    response = await anysearch.AnySearchFetchAdapter().fetch(
        FetchRequest(urls=[URL_OK]), {}
    )
    assert response.results[0].error.type == WebErrorType.TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"code": 0, "data": {"results": "not-a-list"}},
        {"code": 0, "data": {"results": [{"title": "missing url"}]}},
        {"code": -1, "message": "api_key=super-secret", "data": {"results": []}},
    ],
)
async def test_invalid_search_response_is_generic(monkeypatch, payload):
    async def fake_request(*args, **kwargs):
        return payload

    monkeypatch.setattr(anysearch, "request_json", fake_request)
    api = anysearch.AnySearchAPI(api_key="test-key")

    with pytest.raises(anysearch.AnySearchResponseError) as exc_info:
        await api.search("invalid response test")

    assert "super-secret" not in str(exc_info.value)
    assert "invalid response" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_invalid_extract_response_is_fail_closed(monkeypatch):
    async def fake_request(*args, **kwargs):
        return {"code": 0, "data": {"url": URL_OK, "title": "missing content"}}

    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")
    monkeypatch.setattr(anysearch, "request_json", fake_request)

    response = await anysearch.AnySearchFetchAdapter().fetch(
        FetchRequest(urls=[URL_OK]), {}
    )

    error = response.results[0].error
    assert error.type == WebErrorType.PROVIDER_ERROR
    assert error.retryable is True
    assert "invalid response" in error.message.lower()


def test_anysearch_is_discovered_with_optional_auth(monkeypatch):
    from src.tools.web.manifest import (
        CAPABILITY_FETCH,
        CAPABILITY_SEARCH,
        get_capability,
        get_web_provider_spec,
        provider_is_configured,
    )
    from src.tools.web.router import _ADAPTER_BUILDERS
    from src.tools.web.search import _PROVIDER_BUILDERS

    spec = get_web_provider_spec("anysearch")
    assert spec is not None
    assert spec.env_key == "ANYSEARCH_API_KEY"
    assert spec.auth_required is False
    assert get_capability("anysearch", CAPABILITY_SEARCH) is not None
    assert get_capability("anysearch", CAPABILITY_FETCH) is not None
    assert "anysearch" in _PROVIDER_BUILDERS
    assert "anysearch" in _ADAPTER_BUILDERS

    monkeypatch.delenv("ANYSEARCH_API_KEY", raising=False)
    assert provider_is_configured(spec) is True
    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")
    assert provider_is_configured(spec) is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, WebErrorType.UNSUPPORTED_URL),
        (401, WebErrorType.FORBIDDEN),
        (402, WebErrorType.BUDGET_EXCEEDED),
        (422, WebErrorType.PROVIDER_ERROR),
        (429, WebErrorType.RATE_LIMITED),
        (503, WebErrorType.PROVIDER_ERROR),
    ],
)
async def test_extract_http_status_taxonomy(monkeypatch, status, expected):
    secret = "api_key=super-secret&password=do-not-return"

    async def fail_request(*args, **kwargs):
        raise _http_status_error(status, secret, EXTRACT_URL)

    monkeypatch.setenv("ANYSEARCH_API_KEY", "test-key")
    monkeypatch.setattr(anysearch, "request_json", fail_request)

    response = await anysearch.AnySearchFetchAdapter().fetch(
        FetchRequest(urls=[URL_OK]), {}
    )

    error = response.results[0].error
    assert error.type == expected
    assert error.http_status == status
    assert "super-secret" not in error.message
    assert "do-not-return" not in error.message
