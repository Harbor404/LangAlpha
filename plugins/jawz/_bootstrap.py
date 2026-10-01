"""Reach the repo root, then hand back the real bootstrap.

The launched adapter gets its own bundle directory as ``sys.path[0]``; this shim
puts the repository root back on the path so it can import the shared MCP
bootstrap and envelope helpers.
"""

import sys
from pathlib import Path

_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file()
)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp_servers._bootstrap import MCPServer

__all__ = ["MCPServer"]
