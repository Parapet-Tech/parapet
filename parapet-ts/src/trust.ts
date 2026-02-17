// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import type { TrustSpan } from "./types.js";

/**
 * Per-request registry tracking which strings are untrusted data payloads.
 *
 * SDK users register strings via `untrusted(content, source)`. On request
 * intercept, the transport calls `findSpans(body)` to locate those strings
 * in the serialized JSON body, then emits byte ranges as X-Guard-Trust spans.
 */
export class TrustRegistry {
  private static readonly MAX_ENTRIES = 1000;
  private readonly entries: Array<{ content: string; source: string }> = [];
  private readonly maxEntries: number;

  constructor(maxEntries: number = TrustRegistry.MAX_ENTRIES) {
    this.maxEntries = maxEntries;
  }

  /**
   * Register a string as untrusted with a provenance label.
   * Returns the string unchanged (for inline use in message construction).
   *
   * Empty strings are ignored (no-op).
   * Throws when the registry is full.
   */
  register(content: string, source: string): string {
    if (!content) return content;

    if (this.entries.length >= this.maxEntries) {
      throw new Error(
        `TrustRegistry full: ${this.maxEntries} entries. ` +
          "Increase maxEntries or reduce untrusted content registrations.",
      );
    }

    this.entries.push({ content, source });
    return content;
  }

  /**
   * Find all registered untrusted strings in a serialized JSON request body.
   *
   * For each registered string:
   * 1. JSON-escape it (the body is JSON — the string appears escaped)
   * 2. Search for the escaped form using indexOf
   * 3. Convert character offsets to UTF-8 byte offsets
   * 4. Return sorted by start offset
   */
  findSpans(body: string): TrustSpan[] {
    if (this.entries.length === 0) return [];

    const spans: TrustSpan[] = [];

    for (const entry of this.entries) {
      // JSON.stringify wraps in quotes; strip them to get the escaped interior.
      const escaped = JSON.stringify(entry.content).slice(1, -1);

      let searchFrom = 0;
      for (;;) {
        const idx = body.indexOf(escaped, searchFrom);
        if (idx === -1) break;

        const byteStart = Buffer.byteLength(body.slice(0, idx), "utf8");
        const byteEnd =
          byteStart + Buffer.byteLength(escaped, "utf8");

        spans.push({ start: byteStart, end: byteEnd, source: entry.source });
        searchFrom = idx + 1;
      }
    }

    // Sort by start offset, then by end offset descending (longer spans first
    // for same start — matches Python SDK).
    spans.sort((a, b) => a.start - b.start || b.end - a.end);
    return spans;
  }

  /** Number of registered entries. */
  get entryCount(): number {
    return this.entries.length;
  }

  /** Whether the registry has any entries. */
  get hasEntries(): boolean {
    return this.entries.length > 0;
  }

  /** Remove all registered entries. */
  clear(): void {
    this.entries.length = 0;
  }
}

/**
 * Check whether a Content-Type header value indicates JSON content.
 *
 * Matches `application/json` or any `+json` structured syntax suffix
 * (e.g., `application/vnd.api+json`). Ignores parameters like
 * `;charset=utf-8`. Case-insensitive.
 *
 * Returns false for null/undefined/non-JSON types.
 */
export function isJsonContentType(
  contentType: string | null | undefined,
): boolean {
  if (!contentType) return false;

  // Strip parameters: "application/json; charset=utf-8" → "application/json"
  const mediaType = contentType.split(";")[0].trim().toLowerCase();

  if (mediaType === "application/json") return true;

  // Match +json structured syntax suffix (RFC 6839).
  // Must be valid type/subtype form with a non-empty prefix before +json.
  const slash = mediaType.indexOf("/");
  if (slash < 1) return false;
  const subtype = mediaType.slice(slash + 1);
  return subtype.length > "+json".length && subtype.endsWith("+json");
}
