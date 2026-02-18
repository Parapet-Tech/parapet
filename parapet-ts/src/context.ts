// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import { AsyncLocalStorage } from "node:async_hooks";
import type { TrustRegistry } from "./trust.js";

/**
 * Per-session context propagated via AsyncLocalStorage.
 *
 * Created by `session()`, consumed by the transport to inject
 * baggage and trust headers automatically.
 */
export interface ParapetContext {
  readonly userId?: string;
  readonly role?: string;
  readonly trustRegistry: TrustRegistry;
}

const store = new AsyncLocalStorage<ParapetContext>();

/**
 * Get the current Parapet context, or undefined if not inside a session.
 */
export function getContext(): ParapetContext | undefined {
  return store.getStore();
}

/**
 * Run a callback within a Parapet context scope.
 *
 * The context is available via `getContext()` for the duration of `fn`,
 * including across async boundaries (AsyncLocalStorage propagation).
 */
export function runWithContext<T>(ctx: ParapetContext, fn: () => T): T {
  return store.run(ctx, fn);
}
