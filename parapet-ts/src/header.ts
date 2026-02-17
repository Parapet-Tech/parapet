// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import type { SessionOptions } from "./types.js";

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
