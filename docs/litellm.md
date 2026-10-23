# Parapet With LiteLLM

Common chain:

```text
app -> Parapet -> LiteLLM -> model providers
```

Parapet supports either an HTTPS LiteLLM endpoint or an explicitly configured
plain-HTTP endpoint at a literal loopback IP. For local LiteLLM on port 4000:

```powershell
parapet-engine --config parapet.yaml --port 9800 `
  --upstream "openai=http://127.0.0.1:4000" `
  --allow-loopback-http-upstream
```

The opt-in accepts only literal loopback IPs (`127.0.0.0/8` or `[::1]`). It
does not enable `localhost`, LAN addresses, link-local addresses, or remote
plain-HTTP targets. Those configurations fail closed with
`upstream_config_invalid`; use `https://` instead.

Example (HTTPS LiteLLM endpoint):

```powershell
$env:PARAPET_API_OPENAI_COM_BASE_URL = "https://litellm.internal.example"
parapet-engine --config parapet.yaml --port 9800
```

Then point your client at:

```text
http://127.0.0.1:9800
```

Notes:

- In SDK sidecar mode, Parapet intercepts requests locally and routes `api.openai.com` to `PARAPET_API_OPENAI_COM_BASE_URL`.
- Zero-SDK mode (no `X-Parapet-Original-Host` header) uses `PARAPET_OPENAI_BASE_URL`.
- Ambient fallback variables (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`) never enable plain HTTP. Use an explicit `--upstream` or primary `PARAPET_<PROVIDER>_BASE_URL` literal-loopback value together with `--allow-loopback-http-upstream`, or use HTTPS/remove the fallback.
- Do not set `OPENAI_BASE_URL` to Parapet's own address in the engine's environment. Parapet reads it as an upstream fallback and refuses to start.
- If your app calls non-default hosts, add bare lowercase DNS hostnames to `PARAPET_EXTRA_HOSTS` so the SDK and engine host allowlists match.
