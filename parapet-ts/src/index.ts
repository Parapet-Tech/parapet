// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import type { SessionOptions } from "./types.js";
import type { ParapetContext } from "./context.js";
import { getContext, runWithContext } from "./context.js";
import { TrustRegistry } from "./trust.js";

export { createParapetFetch, ParapetTimeoutError } from "./transport.js";
export { buildBaggageHeader, buildTrustHeader } from "./header.js";
export { TrustRegistry } from "./trust.js";
export type { ParapetContext } from "./context.js";
export type {
  ParapetConfig,
  SessionOptions,
  TrustSpan,
  TransportOptions,
} from "./types.js";

/**
 * Run a callback within a Parapet session context.
 *
 * Session metadata (userId, role) is propagated via baggage headers.
 * A fresh TrustRegistry is created per session for trust span tracking.
 * Context propagates through all async boundaries (AsyncLocalStorage).
 *
 * Accepts both sync and async callbacks — if `fn` is async, returns
 * a Promise that resolves when the callback completes.
 */
export function session<T>(
  opts: SessionOptions,
  fn: () => T,
): T {
  const ctx: ParapetContext = {
    userId: opts.userId,
    role: opts.role,
    trustRegistry: new TrustRegistry(),
  };
  return runWithContext(ctx, fn);
}

/**
 * Mark content as untrusted with a provenance label.
 *
 * Registers the string in the current session's TrustRegistry so the
 * transport can locate it in serialized requests and emit byte-range
 * trust spans via the x-guard-trust header.
 *
 * Returns content unchanged (for inline use in message construction).
 *
 * Must be called inside a `session()` scope.
 */
export function untrusted(content: string, source: string): string {
  const ctx = getContext();
  if (!ctx) {
    throw new Error(
      "parapet.untrusted() called outside a session. " +
        "Wrap your code in parapet.session() first.",
    );
  }
  return ctx.trustRegistry.register(content, source);
}
