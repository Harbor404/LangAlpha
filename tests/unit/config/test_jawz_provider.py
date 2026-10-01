"""Contract for the optional, disabled-by-default Jawz bundled provider.

Jawz is a read-only remote MCP service with no credential. Shipping it disabled
keeps the new macro surface out of every existing agent prompt until an operator
opts in, while the bundle remains discoverable through the normal plugin path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptc_agent.config import plugins as bundles
from ptc_agent.config.utils import create_mcp_config

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _real_bundles(monkeypatch):
    monkeypatch.setattr(bundles, "BUNDLES_DIR", REPO_ROOT / "plugins")


def _section(servers: list[dict] | None = None) -> dict:
    section = {"tool_discovery_enabled": True}
    if servers is not None:
        section["servers"] = servers
    return section


def _bundled_jawz():
    return next(s for s in bundles.bundled_mcp_servers() if s.name == "jawz")


def test_jawz_bundle_is_discovered_as_an_optional_stdio_provider():
    jawz = _bundled_jawz()

    assert jawz.enabled is False
    assert jawz.transport == "stdio"
    assert jawz.command == "uv"
    assert jawz.args == ["run", "python", "plugins/jawz/jawz_mcp_server.py"]
    assert jawz.env == {}
    assert jawz.headers == {}
    assert jawz.vault_blueprints == []
    assert jawz.description
    assert jawz.instruction


def test_real_jawz_bundle_is_not_an_extra_agent_surface_until_enabled():
    servers = {s.name: s for s in create_mcp_config(_section()).servers}

    assert servers["jawz"].enabled is False
    # Every pre-existing bundle keeps its own enablement; adding Jawz must not
    # flip, replace, or silently suppress any provider that shipped before it.
    assert servers["price_data"].enabled is True
    assert servers["fundamentals"].enabled is True
    assert servers["macro"].enabled is True
    assert servers["options"].enabled is True


def test_existing_yaml_override_enables_jawz_without_restating_its_contract():
    servers = {
        s.name: s
        for s in create_mcp_config(
            _section([{"name": "jawz", "enabled": True}])
        ).servers
    }
    jawz = servers["jawz"]

    assert jawz.enabled is True
    assert jawz.transport == "stdio"
    assert jawz.command == "uv"
    assert jawz.args == ["run", "python", "plugins/jawz/jawz_mcp_server.py"]
