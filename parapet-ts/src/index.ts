// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import type { ParapetConfig, SessionOptions } from "./types.js";
import type { ParapetContext } from "./context.js";
import { getContext, runWithContext } from "./context.js";
import { TrustRegistry } from "./trust.js";
import { createParapetFetch, DEFAULT_INTERCEPTED_HOSTS } from "./transport.js";
import { EngineManager } from "./sidecar.js";
import { existsSync } from "node:fs";

export { createParapetFetch, ParapetTimeoutError, DEFAULT_INTERCEPTED_HOSTS } from "./transport.js";
export { buildBaggageHeader, buildTrustHeader } from "./header.js";
export { TrustRegistry } from "./trust.js";
export { EngineManager } from "./sidecar.js";
export type { SidecarDeps } from "./sidecar.js";
export type { ParapetContext } from "./context.js";
export type {
  ParapetConfig,
  SessionOptions,
  TrustSpan,
  TransportOptions,
} from "./types.js";

// ---------------------------------------------------------------------------
// Module-level state
// ---------------------------------------------------------------------------

let engine: EngineManager | null = null;
let parapetFetch: typeof globalThis.fetch | null = null;

/** @internal Injected deps for testing — used by EngineManager when non-null. */
let _testDeps: import("./sidecar.js").SidecarDeps | null = null;

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
  if (!parapetFetch) {
    throw new Error(
      "parapet.session() called before init(). " +
        "Call parapet.init() first.",
    );
  }
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
        "Call parapet.init() and wrap your code in parapet.session() first.",
    );
  }
  return ctx.trustRegistry.register(content, source);
}

// ---------------------------------------------------------------------------
// init / getParapetFetch
// ---------------------------------------------------------------------------

const DEFAULT_PORT = 9800;

/**
 * Initialize Parapet. Starts the engine sidecar and creates the fetch wrapper.
 *
 * Idempotent: same config file contents + canonical path → no-op.
 * Changed contents or path → stops old engine, starts fresh.
 */
export async function init(config: ParapetConfig): Promise<void> {
  const port = config.port ?? DEFAULT_PORT;
  const autoStart = config.autoStart ?? true;

  if (autoStart) {
    const exists = _testDeps ? _testDeps.exists(config.configPath) : existsSync(config.configPath);
    if (!exists) {
      throw new Error(`Config file not found: ${config.configPath}`);
    }
  }

  if (autoStart) {
    // Reuse the existing manager if port AND canonical config path
    // match. Uses realpath for path comparison — "./parapet.yaml" and
    // "/abs/path/parapet.yaml" are treated as identical if they resolve
    // to the same file. Content-level idempotency (hash check) is
    // delegated to engine.start().
    if (
      engine &&
      engine.port === port &&
      engine.matchesCanonicalConfig(config.configPath)
    ) {
      await engine.start(); // No-op if config content unchanged.
    } else {
      // Different port, different config, or no engine — stop old, create new.
      if (engine) {
        await engine.stop();
      }
      engine = new EngineManager({
        configPath: config.configPath,
        port,
        deps: _testDeps ?? undefined,
      });
      await engine.start();
    }
  } else if (engine) {
    // autoStart disabled but old engine exists — stop it.
    await engine.stop();
    engine = null;
  }

  const hosts = config.extraHosts
    ? [...DEFAULT_INTERCEPTED_HOSTS, ...config.extraHosts]
    : undefined;

  parapetFetch = createParapetFetch(globalThis.fetch, {
    port,
    interceptedHosts: hosts,
  });
}

/**
 * Get the Parapet-wrapped fetch function.
 * Use with OpenAI SDK: `new OpenAI({ fetch: getParapetFetch() })`
 *
 * Throws if `init()` has not been called.
 */
export function getParapetFetch(): typeof globalThis.fetch {
  if (!parapetFetch) {
    throw new Error(
      "parapet.getParapetFetch() called before init(). " +
        "Call parapet.init() first.",
    );
  }
  return parapetFetch;
}

/**
 * Shut down the engine sidecar (if managed).
 * Called automatically on process exit; exposed for explicit cleanup.
 */
export async function shutdown(): Promise<void> {
  if (engine) {
    await engine.stop();
    engine = null;
  }
  parapetFetch = null;
}

/** @internal Reset module state for testing. Optionally inject deps for EngineManager. */
export function _resetForTesting(
  deps?: import("./sidecar.js").SidecarDeps,
): void {
  engine = null;
  parapetFetch = null;
  _testDeps = deps ?? null;
}
