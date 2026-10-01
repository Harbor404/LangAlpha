# jawz

An optional, disabled-by-default adapter for [Jawz](https://jawz.ai/docs/mcp), a
keyless read-only MCP service for macro regime, financial conditions, global
liquidity, and event calendars.

The adapter is intentionally narrower than Jawz's 22-tool public surface. It
exposes the four capabilities named in the integration issue through the same
standard success/error envelope as the other market-data servers. It talks to
the published streamable-HTTP MCP endpoint with `httpx`, validates the JSON-RPC
and `structuredContent` shape, applies a finite timeout, and never places remote
text or transport exception strings in the agent-visible error detail.

Enable it in `agent_config.yaml` when needed:

```yaml
mcp:
  servers:
    - name: "jawz"
      enabled: true
```
