// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import type { TransportOptions } from "./types.js";
import { buildBaggageHeader, buildTrustHeader } from "./header.js";
import { getContext } from "./context.js";
import { isJsonContentType } from "./trust.js";

export const DEFAULT_INTERCEPTED_HOSTS: readonly string[] = [
  "api.openai.com",
  "api.anthropic.com",
  "api.cerebras.ai",
  "api.groq.com",
  "generativelanguage.googleapis.com",
];

const DEFAULT_HOSTS: ReadonlySet<string> = new Set(DEFAULT_INTERCEPTED_HOSTS);

const DEFAULT_TIMEOUT_MS = 5000;

/**
 * Error thrown when the engine call is aborted (timeout or caller cancel).
 * Signals failclosed — the request was NOT sent to the provider.
 */
export class ParapetTimeoutError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "ParapetTimeoutError";
  }
}

/**
 * Walk the cause chain looking for ECONNREFUSED.
 * Node's fetch nests the system error as TypeError.cause.
 */
function isConnectionRefused(err: unknown): boolean {
  if (err == null || typeof err !== "object") return false;
  if ("code" in err && (err as { code: unknown }).code === "ECONNREFUSED") {
    return true;
  }
  if ("cause" in err) {
    return isConnectionRefused((err as { cause: unknown }).cause);
  }
  return false;
}

/**
 * Create a Parapet-wrapped fetch function that routes LLM API requests
 * through the engine sidecar.
 *
 * Non-intercepted hosts pass through unchanged.
 * Connection refused → failopen (retry direct to provider).
 * Timeout / caller abort → failclosed (throw ParapetTimeoutError).
 */
export function createParapetFetch(
  baseFetch: typeof globalThis.fetch,
  opts: TransportOptions,
): typeof globalThis.fetch {
  const hosts = new Set(opts.interceptedHosts ?? DEFAULT_HOSTS);
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;

  return async (
    input: string | URL | Request,
    init?: RequestInit,
  ): Promise<Response> => {
    // Extract URL without consuming the body stream.  new Request()
    // would disturb a Request input's body, making pass-through fail.
    const rawUrl =
      input instanceof Request ? input.url
      : input instanceof URL ? input.href
      : input;
    const url = new URL(rawUrl);

    // Non-intercepted host — pass through unchanged (body untouched).
    if (!hosts.has(url.hostname)) {
      return baseFetch(input, init);
    }

    // Intercepted — normalize to a Request so all properties (method,
    // body, headers, redirect, cache, credentials, etc.) are captured
    // in one object.
    const originalReq = new Request(input, init);

    // Rewrite to engine sidecar.
    const originalHost = url.hostname;
    url.protocol = "http:";
    url.hostname = "127.0.0.1";
    url.port = String(opts.port);

    // Fail fast if the caller already aborted before we start.
    if (originalReq.signal.aborted) {
      throw new ParapetTimeoutError("engine call aborted", {
        cause: originalReq.signal.reason,
      });
    }

    // Build headers: clone from original request, add ours.
    const headers = new Headers(originalReq.headers);
    headers.set("x-parapet-original-host", originalHost);

    // Read session context (AsyncLocalStorage).
    const ctx = getContext();
    if (ctx) {
      // Baggage from session context.
      const baggage = buildBaggageHeader({
        userId: ctx.userId,
        role: ctx.role,
      });
      if (baggage) {
        const existing = headers.get("baggage");
        headers.set("baggage", existing ? `${existing},${baggage}` : baggage);
      }

      // Trust spans: only when registry has entries AND body is JSON.
      if (ctx.trustRegistry.hasEntries) {
        const contentType = headers.get("content-type");
        if (isJsonContentType(contentType)) {
          const bodyText = await originalReq.clone().text();
          const spans = ctx.trustRegistry.findSpans(bodyText);
          const trustHeader = buildTrustHeader(spans);
          if (trustHeader) {
            headers.set("x-guard-trust", trustHeader);
          }
        }
      }
    }

    // Compose caller signal with engine timeout.
    const timeoutSignal = AbortSignal.timeout(timeoutMs);
    const signal = AbortSignal.any([originalReq.signal, timeoutSignal]);

    // Clone the request so the original body stream stays intact for
    // failopen retry. The clone is consumed by the engine attempt.
    // Using new Request(url, clone) preserves ALL request properties
    // (credentials, cache, integrity, keepalive, referrer, etc.).
    const engineReq = new Request(url.toString(), originalReq.clone() as RequestInit);
    // Override headers (we added x-parapet-original-host + baggage)
    // and signal (we composed caller + timeout).
    const engineInit: RequestInit = {
      headers,
      signal,
    };
    if (engineReq.body) {
      // Re-supply body + duplex since new Request() may have consumed it.
      engineInit.body = engineReq.body;
      (engineInit as Record<string, unknown>).duplex = "half";
    }

    try {
      return await baseFetch(
        new Request(engineReq, engineInit),
      );
    } catch (err) {
      // Timeout or caller abort → failclosed.
      if (err instanceof DOMException && err.name === "AbortError") {
        throw new ParapetTimeoutError("engine call aborted", { cause: err });
      }
      // Connection refused → failopen: retry direct to provider with
      // the original (unconsumed) request.
      if (err instanceof TypeError && isConnectionRefused(err)) {
        // eslint-disable-next-line no-console
        console.warn("[parapet] engine unreachable, failing open");
        return baseFetch(originalReq);
      }
      // Unknown error → failclosed.
      throw err;
    }
  };
}
