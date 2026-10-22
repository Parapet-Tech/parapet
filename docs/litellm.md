# Parapet With LiteLLM

Common chain:

```text
app -> Parapet -> LiteLLM (HTTPS) -> model providers
```

Parapet in this release forwards only to `https://` upstreams. A plain-HTTP value, including loopback, makes Parapet refuse to start with error code `upstream_config_invalid`:

```text
upstream configuration refused: PARAPET_API_OPENAI_COM_BASE_URL: non_https_upstream (value withheld, ...). Fix: plain-HTTP upstreams are not supported in this release; use an https:// URL (loopback HTTP support is planned behind an explicit flag)
```

Local loopback HTTP targets (LiteLLM, MLX) are **not yet supported**.

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
- Do not set `OPENAI_BASE_URL` to Parapet's own address in the engine's environment. Parapet reads it as an upstream fallback and refuses to start.
- If your app calls non-default hosts, add bare lowercase DNS hostnames to `PARAPET_EXTRA_HOSTS` so the SDK and engine host allowlists match.
