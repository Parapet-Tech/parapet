# Parapet With LiteLLM

Common chain:

```text
app -> Parapet -> LiteLLM -> model providers
```

Basic setup:

1. Run LiteLLM locally, for example on `http://127.0.0.1:4000`
2. Point Parapet's upstream OpenAI base URL at LiteLLM
3. Start Parapet normally
4. Keep using your existing OpenAI-compatible client path

Example:

```powershell
$env:PARAPET_API_OPENAI_COM_BASE_URL = "http://127.0.0.1:4000"
parapet-engine --config parapet.yaml --port 9800
```

Then point your client at:

```text
http://127.0.0.1:9800
```

Notes:

- In SDK sidecar mode, Parapet still intercepts requests locally before forwarding upstream.
- In proxy-only mode, Parapet becomes the explicit firewall hop in front of LiteLLM.
- If your app calls non-default hosts, make sure the SDK and engine host allowlists are configured accordingly.
