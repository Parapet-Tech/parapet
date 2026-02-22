# Copyright 2026 The Parapet Project
# SPDX-License-Identifier: Apache-2.0

"""
V4-11 spike: Provider compatibility test for Accept-Encoding: identity on streaming SSE.

Tests whether OpenAI-compatible providers honor Accept-Encoding: identity
and return uncompressed streaming responses.

Usage: python scripts/v4_11_provider_compat.py
Requires: .env with API keys (GEMINI_API_KEY, GROQ_API_KEY, etc.)
"""

import os
import time
import requests
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

PROVIDERS = [
    {
        "name": "Gemini (OpenAI-compat)",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash-lite",
    },
    {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant",
    },
    {
        "name": "Cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "key_env": "CEREBRAS_API_KEY",
        "model": "llama-3.3-70b",
    },
    {
        "name": "SambaNova",
        "url": "https://api.sambanova.ai/v1/chat/completions",
        "key_env": "SAMBANOVA_API_KEY",
        "model": "Meta-Llama-3.1-8B-Instruct",
    },
]

PROMPT = {"role": "user", "content": "Say hello in exactly 5 words."}


def test_provider(provider, accept_encoding):
    """Send streaming request with specified Accept-Encoding, return timing and headers."""
    key = os.environ.get(provider["key_env"], "")
    if not key:
        return {"status": "SKIP", "reason": f"no {provider['key_env']}"}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept-Encoding": accept_encoding,
    }
    body = {
        "model": provider["model"],
        "messages": [PROMPT],
        "stream": True,
        "max_tokens": 30,
    }

    try:
        t0 = time.monotonic()
        resp = requests.post(
            provider["url"],
            json=body,
            headers=headers,
            stream=True,
            timeout=15,
        )
        ttfb = time.monotonic() - t0

        # Read all chunks
        total_bytes = 0
        chunk_count = 0
        for chunk in resp.iter_content(chunk_size=None):
            total_bytes += len(chunk)
            chunk_count += 1
        total_time = time.monotonic() - t0

        content_encoding = resp.headers.get("Content-Encoding", "(none)")
        content_type = resp.headers.get("Content-Type", "(none)")
        transfer_encoding = resp.headers.get("Transfer-Encoding", "(none)")

        return {
            "status": resp.status_code,
            "content_encoding": content_encoding,
            "content_type": content_type,
            "transfer_encoding": transfer_encoding,
            "ttfb_ms": round(ttfb * 1000),
            "total_ms": round(total_time * 1000),
            "bytes": total_bytes,
            "chunks": chunk_count,
        }
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}


def main():
    print("V4-11 Provider Compatibility: Accept-Encoding: identity on streaming SSE")
    print("=" * 80)

    results = []

    for provider in PROVIDERS:
        print(f"\n--- {provider['name']} ({provider['model']}) ---")

        for ae in ["identity", "gzip, deflate, br"]:
            label = "identity" if ae == "identity" else "default"
            print(f"  Accept-Encoding: {ae}")
            result = test_provider(provider, ae)
            result["provider"] = provider["name"]
            result["accept_encoding"] = label
            results.append(result)

            if result["status"] == "SKIP":
                print(f"    SKIP: {result['reason']}")
            elif result["status"] == "ERROR":
                print(f"    ERROR: {result['reason']}")
            else:
                print(f"    HTTP {result['status']}")
                print(f"    Content-Encoding: {result['content_encoding']}")
                print(f"    Content-Type: {result['content_type']}")
                print(f"    Transfer-Encoding: {result['transfer_encoding']}")
                print(f"    TTFB: {result['ttfb_ms']}ms  Total: {result['total_ms']}ms")
                print(f"    Bytes: {result['bytes']}  Chunks: {result['chunks']}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Provider':<25} {'AE Mode':<10} {'HTTP':>5} {'Content-Enc':<15} {'TTFB':>7} {'Total':>7} {'Bytes':>7}")
    print("-" * 80)
    for r in results:
        if r["status"] in ("SKIP", "ERROR"):
            print(f"{r['provider']:<25} {r['accept_encoding']:<10} {r['status']}")
        else:
            print(
                f"{r['provider']:<25} {r['accept_encoding']:<10} {r['status']:>5} "
                f"{r['content_encoding']:<15} {r['ttfb_ms']:>5}ms {r['total_ms']:>5}ms {r['bytes']:>7}"
            )

    # Verdict
    print("\nVERDICT:")
    identity_results = [r for r in results if r["accept_encoding"] == "identity" and isinstance(r["status"], int)]
    compressed_count = sum(1 for r in identity_results if r["content_encoding"] not in ("(none)", "identity"))
    if compressed_count == 0:
        print("  All tested providers honored Accept-Encoding: identity (no compression on response).")
        print("  Option A is viable for all tested providers.")
    else:
        print(f"  {compressed_count}/{len(identity_results)} providers returned compressed despite identity request.")
        print("  Option A may need fallback for non-compliant providers.")


if __name__ == "__main__":
    main()
