"""The 030 exemption has to ask the same question ``brokerage_for_url`` asks.

030 left a row under the reserved name alone when its URL started with the
vendor's own address, as a plain case-sensitive text prefix. The app resolver
(``src/server/services/brokerages.py``) matches on the parsed host instead, so
the two disagree on any stored address with a different case -- exactly the bug
PR #375 fixed in ``032_free_moomoo_name`` for ``moomoo``. A row the app still
draws as the vendor would be renamed out from under a live OAuth connection,
while an address that merely shares the vendor's prefix belongs to somebody else
and must still move aside.

The migration's expression is verified here through the same host semantics, and
the generated SQL is checked for the case-insensitive host form so a revert to
``LIKE vendor_url || '%'`` cannot pass unnoticed.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import urlsplit

import pytest

_VERSIONS = Path(__file__).resolve().parents[3] / "migrations" / "versions"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migration(monkeypatch):
    module = _load("migration_030", _VERSIONS / "030_free_brokerage_names.py")
    op = MagicMock()
    monkeypatch.setattr(module, "op", op)
    return module, op


def _user_tier_sql(op) -> str:
    sql = " ".join(str(call.args[0]) for call in op.execute.call_args_list)
    return re.sub(r"\s+", " ", sql)


def _host_of(url: str | None) -> str | None:
    """Mirror the migration's host expression, as ``brokerage_for_url`` does."""
    if not url:
        return None
    authority = re.sub(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/?#]*)([/?#].*)?$", r"\1", url.strip()
    )
    authority = re.sub(r"^\[.*\]", "", authority)
    authority = re.sub(r":[0-9]+$", "", authority)
    return authority.strip().lower()


def _is_shipped(url: str, vendor_host: str) -> bool:
    return _host_of(url) == vendor_host


def test_registers_in_the_linear_chain(migration):
    module, _op = migration
    assert module.revision == "030"
    assert module.down_revision == "029"


@pytest.mark.parametrize(
    "url,vendor_host,expected",
    [
        ("https://agent.robinhood.com/mcp/trading", "agent.robinhood.com", True),
        ("HTTPS://AGENT.ROBINHOOD.COM/mcp/trading", "agent.robinhood.com", True),
        ("https://Agent.Robinhood.com/mcp/trading", "agent.robinhood.com", True),
        ("https://agent.robinhood.com:443/mcp", "agent.robinhood.com", True),
        ("http://AGENT.ROBINHOOD.COM/?a=b", "agent.robinhood.com", True),
        ("https://api.ibkr.com/v1/api/mcp-public", "api.ibkr.com", True),
        ("https://API.IBKR.COM", "api.ibkr.com", True),
        ("https://agent.robinhood.com.evil.com/x", "agent.robinhood.com", False),
        ("https://api.ibkr.com.evil.com", "api.ibkr.com", False),
        ("https://agent.robinhood.com/anything", "agent.robinhood.com", True),
    ],
)
def test_host_match_agrees_with_the_url_resolver(url, vendor_host, expected):
    if expected:
        assert urlsplit(url).hostname == vendor_host
    assert _is_shipped(url, vendor_host) is expected


def test_the_migration_pins_both_shipped_hosts(migration):
    module, _op = migration
    assert "'robinhood', 'agent.robinhood.com'" in module._SHIPPED
    assert "'ibkr', 'api.ibkr.com'" in module._SHIPPED
    # The old column carried a URL; the host form is what keeps the two in step.
    assert "vendor_url" not in module._SHIPPED


def test_the_exemption_is_a_case_insensitive_host_test(migration):
    module, op = migration
    module.upgrade()
    sql = _user_tier_sql(op)
    assert "lower(btrim(" in sql
    assert "<> shipped.vendor_host" in sql
    # The case-sensitive text prefix that shipped is gone.
    assert "NOT LIKE shipped.vendor_url" not in sql
