// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import type { SessionOptions, TrustSpan } from "./types.js";

const MAX_TRUST_HEADER_BYTES = 4096;

/**
 * Percent-encode a baggage value for W3C Baggage spec parity with
 * Python SDK (urllib.parse.quote with safe="").
 *
 * encodeURIComponent leaves !'()* unescaped. We encode those too
 * so both SDKs produce identical headers.
 */
function encodeBaggageValue(value: string): string {
  return encodeURIComponent(value).replace(
    /[!'()*]/g,
    (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`,
  );
}

/**
 * Build a W3C Baggage header value from session options.
 *
 * Returns undefined when no fields are provided.
 * Values are percent-encoded per RFC 3986 (strict, matching Python SDK).
 */
export function buildBaggageHeader(
  opts: SessionOptions,
): string | undefined {
  const parts: string[] = [];

  if (opts.userId != null) {
    parts.push(`user_id=${encodeBaggageValue(opts.userId)}`);
  }
  if (opts.role != null) {
    parts.push(`role=${encodeBaggageValue(opts.role)}`);
  }

  return parts.length > 0 ? parts.join(",") : undefined;
}

/**
 * Serialize trust spans as x-guard-trust header value.
 *
 * Format: `inline:<base64-encoded JSON>`
 * JSON schema: `[{"s": start, "e": end, "src": source}, ...]`
 *
 * Returns undefined when no spans are provided or the final header value
 * (inline: prefix + base64 payload) exceeds 4KB.
 */
export function buildTrustHeader(
  spans: readonly TrustSpan[],
): string | undefined {
  if (spans.length === 0) return undefined;

  const compact = spans.map((s) => ({ s: s.start, e: s.end, src: s.source }));
  const payload = JSON.stringify(compact);
  const base64 = Buffer.from(payload, "utf-8").toString("base64");
  const headerValue = `inline:${base64}`;

  // The final header value is all ASCII (inline: + base64), so
  // .length === byte length.
  if (headerValue.length > MAX_TRUST_HEADER_BYTES) {
    return undefined;
  }

  return headerValue;
}
