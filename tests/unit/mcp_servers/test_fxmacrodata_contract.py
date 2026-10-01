"""Manifest contract for the optional FXMacroData macro provider."""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_macro_manifest_passes_fxmacrodata_key_from_host_env():
    manifest = json.loads(
        (_REPO_ROOT / "plugins/langalpha_market_data/mcp.json").read_text()
    )

    assert manifest["mcpServers"]["macro"]["env"]["FXMACRODATA_API_KEY"] == (
        "${FXMACRODATA_API_KEY}"
    )


def test_env_example_documents_fxmacrodata_key():
    env_example = (_REPO_ROOT / ".env.example").read_text()

    assert "FXMACRODATA_API_KEY=" in env_example
