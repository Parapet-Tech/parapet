import { describe, it, expect, vi } from "vitest";
import {
  createParapetFetch,
  ParapetTimeoutError,
} from "../src/transport.js";
import type { TransportOptions } from "../src/types.js";
import { runWithContext } from "../src/context.js";
import type { ParapetContext } from "../src/context.js";
import { TrustRegistry } from "../src/trust.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockResponse(status = 200): Response {
  return new Response(null, { status });
}

function opts(overrides?: Partial<TransportOptions>): TransportOptions {
  return { port: 9800, ...overrides };
}

/** Extract the URL string from whatever baseFetch receives. */
function extractUrl(input: string | URL | Request): string {
  if (input instanceof Request) return input.url;
  if (input instanceof URL) return input.href;
  return input;
}

/** Extract headers from whatever baseFetch receives. */
function extractHeaders(
  input: string | URL | Request,
  init?: RequestInit,
): Headers {
  if (input instanceof Request) return new Headers(input.headers);
  return new Headers(init?.headers);
}

// ---------------------------------------------------------------------------
// URL rewriting
// ---------------------------------------------------------------------------

describe("URL rewriting", () => {
  it("rewrites intercepted host to localhost", async () => {
    let capturedUrl = "";
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      capturedUrl = extractUrl(input);
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://api.openai.com/v1/chat/completions");

    expect(capturedUrl).toBe(
      "http://127.0.0.1:9800/v1/chat/completions",
    );
  });

  it("preserves path, query, and fragment", async () => {
    let capturedUrl = "";
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      capturedUrl = extractUrl(input);
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://api.openai.com/v1/models?limit=10#top");

    expect(capturedUrl).toBe(
      "http://127.0.0.1:9800/v1/models?limit=10#top",
    );
  });

  it("uses custom port", async () => {
    let capturedUrl = "";
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      capturedUrl = extractUrl(input);
      return mockResponse();
    });

    const pfetch = createParapetFetch(
      baseFetch as typeof fetch,
      opts({ port: 8080 }),
    );
    await pfetch("https://api.openai.com/v1/chat/completions");

    expect(capturedUrl).toContain(":8080");
  });
});

// ---------------------------------------------------------------------------
// Host matching
// ---------------------------------------------------------------------------

describe("host matching", () => {
  it("passes through non-intercepted hosts unchanged", async () => {
    let capturedUrl = "";
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      capturedUrl = extractUrl(input);
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://example.com/api/data");

    expect(capturedUrl).toBe("https://example.com/api/data");
  });

  it("passes through non-intercepted Request with body undisturbed", async () => {
    let capturedBody: string | null = null;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      if (input instanceof Request) {
        capturedBody = await input.text();
      }
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({ prompt: "hello" });
    const request = new Request("https://example.com/api/data", {
      method: "POST",
      body,
    });

    await pfetch(request);

    expect(capturedBody).toBe(body);
  });

  it("does not intercept subdomains of known hosts", async () => {
    let capturedUrl = "";
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      capturedUrl = extractUrl(input);
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://evil.api.openai.com/v1/chat/completions");

    expect(capturedUrl).toBe(
      "https://evil.api.openai.com/v1/chat/completions",
    );
  });

  it("intercepts all default hosts", async () => {
    const defaultHosts = [
      "api.openai.com",
      "api.anthropic.com",
      "api.cerebras.ai",
      "api.groq.com",
      "generativelanguage.googleapis.com",
    ];

    for (const host of defaultHosts) {
      let capturedUrl = "";
      const baseFetch = vi.fn(async (input: string | URL | Request) => {
        capturedUrl = extractUrl(input);
        return mockResponse();
      });

      const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
      await pfetch(`https://${host}/v1/test`);

      expect(capturedUrl).toContain("127.0.0.1:9800");
    }
  });

  it("uses custom intercepted hosts", async () => {
    let capturedUrl = "";
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      capturedUrl = extractUrl(input);
      return mockResponse();
    });

    const pfetch = createParapetFetch(
      baseFetch as typeof fetch,
      opts({ interceptedHosts: ["custom.llm.io"] }),
    );
    await pfetch("https://custom.llm.io/v1/complete");

    expect(capturedUrl).toContain("127.0.0.1:9800");
  });

  it("custom hosts replace defaults, not extend", async () => {
    let capturedUrl = "";
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      capturedUrl = extractUrl(input);
      return mockResponse();
    });

    const pfetch = createParapetFetch(
      baseFetch as typeof fetch,
      opts({ interceptedHosts: ["custom.llm.io"] }),
    );
    // Default host should NOT be intercepted when custom list is provided.
    await pfetch("https://api.openai.com/v1/chat/completions");

    expect(capturedUrl).toBe(
      "https://api.openai.com/v1/chat/completions",
    );
  });
});

// ---------------------------------------------------------------------------
// Header injection
// ---------------------------------------------------------------------------

describe("header injection", () => {
  it("adds x-parapet-original-host header", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://api.openai.com/v1/chat/completions");

    expect(capturedHeaders?.get("x-parapet-original-host")).toBe(
      "api.openai.com",
    );
  });

  it("adds baggage header from session context", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await runWithContext(
      { userId: "u_1", role: "admin", trustRegistry: new TrustRegistry() },
      () => pfetch("https://api.openai.com/v1/chat/completions"),
    );

    expect(capturedHeaders?.get("baggage")).toBe("user_id=u_1,role=admin");
  });

  it("omits baggage header when no session context", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://api.openai.com/v1/chat/completions");

    expect(capturedHeaders?.has("baggage")).toBe(false);
  });

  it("preserves caller headers", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://api.openai.com/v1/chat/completions", {
      headers: { Authorization: "Bearer sk-test", "Content-Type": "application/json" },
    });

    expect(capturedHeaders?.get("authorization")).toBe("Bearer sk-test");
    expect(capturedHeaders?.get("content-type")).toBe("application/json");
    expect(capturedHeaders?.get("x-parapet-original-host")).toBe(
      "api.openai.com",
    );
  });
});

// ---------------------------------------------------------------------------
// Failopen: ECONNREFUSED
// ---------------------------------------------------------------------------

describe("failopen on ECONNREFUSED", () => {
  it("retries with original URL when engine is unreachable", async () => {
    const calls: string[] = [];
    let callCount = 0;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      callCount++;
      calls.push(extractUrl(input));
      if (callCount === 1) {
        const sysErr = Object.assign(new Error("connect ECONNREFUSED"), {
          code: "ECONNREFUSED",
        });
        throw new TypeError("fetch failed", { cause: sysErr });
      }
      return mockResponse();
    });

    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const response = await pfetch(
      "https://api.openai.com/v1/chat/completions",
    );

    expect(response.status).toBe(200);
    expect(calls).toHaveLength(2);
    expect(calls[0]).toContain("127.0.0.1:9800");
    expect(calls[1]).toBe("https://api.openai.com/v1/chat/completions");

    warnSpy.mockRestore();
  });

  it("detects ECONNREFUSED nested in cause chain", async () => {
    let callCount = 0;
    const baseFetch = vi.fn(async () => {
      callCount++;
      if (callCount === 1) {
        const innerErr = Object.assign(new Error("ECONNREFUSED"), {
          code: "ECONNREFUSED",
        });
        const outerCause = new Error("connect failed", { cause: innerErr });
        throw new TypeError("fetch failed", { cause: outerCause });
      }
      return mockResponse();
    });

    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const response = await pfetch(
      "https://api.openai.com/v1/chat/completions",
    );

    expect(response.status).toBe(200);
    expect(callCount).toBe(2);

    warnSpy.mockRestore();
  });

  it("failopen works with Request + body (body stream not consumed)", async () => {
    let callCount = 0;
    let fallbackBody: string | null = null;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      callCount++;
      if (callCount === 1) {
        const sysErr = Object.assign(new Error("connect ECONNREFUSED"), {
          code: "ECONNREFUSED",
        });
        throw new TypeError("fetch failed", { cause: sysErr });
      }
      if (input instanceof Request) {
        fallbackBody = await input.text();
      }
      return mockResponse();
    });

    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({ model: "gpt-4", messages: [] });
    const request = new Request(
      "https://api.openai.com/v1/chat/completions",
      { method: "POST", body },
    );

    const response = await pfetch(request);

    expect(response.status).toBe(200);
    expect(callCount).toBe(2);
    expect(fallbackBody).toBe(body);

    warnSpy.mockRestore();
  });
});

// ---------------------------------------------------------------------------
// Failclosed: timeout
// ---------------------------------------------------------------------------

describe("failclosed on timeout", () => {
  it("throws ParapetTimeoutError on AbortError", async () => {
    const baseFetch = vi.fn(async () => {
      throw new DOMException("The operation was aborted.", "AbortError");
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await expect(
      pfetch("https://api.openai.com/v1/chat/completions"),
    ).rejects.toThrow(ParapetTimeoutError);
  });

  it("does not failopen on timeout", async () => {
    const baseFetch = vi.fn(async () => {
      throw new DOMException("The operation was aborted.", "AbortError");
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await expect(
      pfetch("https://api.openai.com/v1/chat/completions"),
    ).rejects.toThrow("engine call aborted");

    expect(baseFetch).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Failclosed: caller signal abort
// ---------------------------------------------------------------------------

describe("failclosed on caller abort", () => {
  it("throws ParapetTimeoutError when caller aborts via init.signal", async () => {
    const controller = new AbortController();
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      // The signal is on the Request now. Check it.
      const req = input instanceof Request ? input : null;
      controller.abort();
      req?.signal.throwIfAborted();
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await expect(
      pfetch("https://api.openai.com/v1/chat/completions", {
        signal: controller.signal,
      }),
    ).rejects.toThrow(ParapetTimeoutError);

    expect(baseFetch).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Unknown errors → failclosed
// ---------------------------------------------------------------------------

describe("failclosed on unknown errors", () => {
  it("re-throws non-network errors without retry", async () => {
    const baseFetch = vi.fn(async () => {
      throw new Error("unexpected");
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await expect(
      pfetch("https://api.openai.com/v1/chat/completions"),
    ).rejects.toThrow("unexpected");

    expect(baseFetch).toHaveBeenCalledTimes(1);
  });

  it("re-throws TypeError without ECONNREFUSED as failclosed", async () => {
    const baseFetch = vi.fn(async () => {
      throw new TypeError("invalid URL");
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await expect(
      pfetch("https://api.openai.com/v1/chat/completions"),
    ).rejects.toThrow(TypeError);

    expect(baseFetch).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Request passthrough (string + init)
// ---------------------------------------------------------------------------

describe("request passthrough", () => {
  it("forwards method and body to engine", async () => {
    let capturedReq: Request | undefined;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      if (input instanceof Request) capturedReq = input;
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({ model: "gpt-4", messages: [] });
    await pfetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });

    expect(capturedReq?.method).toBe("POST");
    expect(capturedReq?.body).toBeDefined();
    // Read the body to verify content.
    const bodyText = await capturedReq!.clone().text();
    expect(bodyText).toBe(body);
  });
});

// ---------------------------------------------------------------------------
// Request object input
// ---------------------------------------------------------------------------

describe("Request object input", () => {
  it("preserves method and body from Request when no init", async () => {
    let capturedReq: Request | undefined;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      if (input instanceof Request) capturedReq = input;
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({ model: "gpt-4", messages: [] });
    const request = new Request(
      "https://api.openai.com/v1/chat/completions",
      { method: "POST", body },
    );

    await pfetch(request);

    expect(capturedReq?.method).toBe("POST");
    expect(capturedReq?.body).toBeDefined();
  });

  it("preserves headers from Request when no init headers", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const request = new Request(
      "https://api.openai.com/v1/chat/completions",
      {
        method: "POST",
        headers: { Authorization: "Bearer sk-test" },
      },
    );

    await pfetch(request);

    expect(capturedHeaders?.get("authorization")).toBe("Bearer sk-test");
    expect(capturedHeaders?.get("x-parapet-original-host")).toBe(
      "api.openai.com",
    );
  });

  it("respects Request.signal for caller cancellation", async () => {
    const controller = new AbortController();
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      const req = input instanceof Request ? input : null;
      controller.abort();
      req?.signal.throwIfAborted();
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const request = new Request(
      "https://api.openai.com/v1/chat/completions",
      { signal: controller.signal },
    );

    await expect(pfetch(request)).rejects.toThrow(ParapetTimeoutError);
    expect(baseFetch).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Pre-aborted signal → failclosed without calling baseFetch
// ---------------------------------------------------------------------------

describe("pre-aborted signal", () => {
  it("throws ParapetTimeoutError immediately when init.signal is already aborted", async () => {
    const baseFetch = vi.fn(async () => mockResponse());

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await expect(
      pfetch("https://api.openai.com/v1/chat/completions", {
        signal: AbortSignal.abort("cancelled"),
      }),
    ).rejects.toThrow(ParapetTimeoutError);

    // baseFetch must NOT be called — request was never sent.
    expect(baseFetch).toHaveBeenCalledTimes(0);
  });

  it("throws ParapetTimeoutError immediately when Request.signal is already aborted", async () => {
    const baseFetch = vi.fn(async () => mockResponse());

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const request = new Request(
      "https://api.openai.com/v1/chat/completions",
      { signal: AbortSignal.abort("cancelled") },
    );

    await expect(pfetch(request)).rejects.toThrow(ParapetTimeoutError);
    expect(baseFetch).toHaveBeenCalledTimes(0);
  });

  it("preserves the abort reason in ParapetTimeoutError.cause", async () => {
    const baseFetch = vi.fn(async () => mockResponse());

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    try {
      await pfetch("https://api.openai.com/v1/chat/completions", {
        signal: AbortSignal.abort("user-cancelled"),
      });
      expect.unreachable("should have thrown");
    } catch (err) {
      expect(err).toBeInstanceOf(ParapetTimeoutError);
      expect((err as ParapetTimeoutError).cause).toBe("user-cancelled");
    }
  });

  it("throws ParapetTimeoutError before body scanning when trust context is active", async () => {
    const baseFetch = vi.fn(async () => mockResponse());

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({
      messages: [{ role: "user", content: "untrusted data" }],
    });

    await expect(
      runWithContext(
        ctxWithTrust([{ content: "untrusted data", source: "rag" }]),
        () =>
          pfetch("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
            signal: AbortSignal.abort("cancelled"),
          }),
      ),
    ).rejects.toThrow(ParapetTimeoutError);

    // baseFetch must NOT be called — abort check fires before body scan.
    expect(baseFetch).toHaveBeenCalledTimes(0);
  });

  it("does not bail on pre-aborted signals for non-intercepted hosts", async () => {
    const baseFetch = vi.fn(async () => mockResponse());

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    // Non-intercepted host — should pass through unchanged (no signal check).
    await pfetch("https://example.com/api/data", {
      signal: AbortSignal.abort("cancelled"),
    });

    // baseFetch IS called — the signal check only applies to intercepted hosts.
    expect(baseFetch).toHaveBeenCalledTimes(1);
  });
});

// ---------------------------------------------------------------------------
// Engine request preserves all Request properties
// ---------------------------------------------------------------------------

describe("engine request property preservation", () => {
  it("preserves redirect policy on the engine request", async () => {
    let capturedReq: Request | undefined;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      if (input instanceof Request) capturedReq = input;
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://api.openai.com/v1/chat/completions", {
      redirect: "error",
    });

    expect(capturedReq?.redirect).toBe("error");
  });

  it("preserves credentials on the engine request", async () => {
    let capturedReq: Request | undefined;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      if (input instanceof Request) capturedReq = input;
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://api.openai.com/v1/chat/completions", {
      credentials: "include",
    });

    expect(capturedReq?.credentials).toBe("include");
  });

  it("preserves properties from a Request object input", async () => {
    let capturedReq: Request | undefined;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      if (input instanceof Request) capturedReq = input;
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const request = new Request(
      "https://api.openai.com/v1/chat/completions",
      {
        method: "POST",
        body: "{}",
        redirect: "manual",
        credentials: "omit",
      },
    );

    await pfetch(request);

    expect(capturedReq?.method).toBe("POST");
    expect(capturedReq?.redirect).toBe("manual");
    expect(capturedReq?.credentials).toBe("omit");
  });
});

// ---------------------------------------------------------------------------
// Trust header injection via context
// ---------------------------------------------------------------------------

/** Helper: create a context with trust entries registered. */
function ctxWithTrust(
  entries: Array<{ content: string; source: string }>,
  sessionOpts?: { userId?: string; role?: string },
): ParapetContext {
  const reg = new TrustRegistry();
  for (const e of entries) {
    reg.register(e.content, e.source);
  }
  return { ...sessionOpts, trustRegistry: reg };
}

describe("trust header injection", () => {
  it("adds x-guard-trust when context has entries and body is JSON", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({
      model: "gpt-4",
      messages: [{ role: "user", content: "untrusted data" }],
    });

    await runWithContext(
      ctxWithTrust([{ content: "untrusted data", source: "rag" }]),
      () =>
        pfetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        }),
    );

    expect(capturedHeaders?.has("x-guard-trust")).toBe(true);
    expect(capturedHeaders?.get("x-guard-trust")).toMatch(/^inline:/);
  });

  it("trust header decodes to correct span data", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({
      messages: [{ role: "user", content: "secret RAG" }],
    });

    await runWithContext(
      ctxWithTrust([{ content: "secret RAG", source: "rag" }]),
      () =>
        pfetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        }),
    );

    const header = capturedHeaders?.get("x-guard-trust")!;
    const base64 = header.slice("inline:".length);
    const spans = JSON.parse(Buffer.from(base64, "base64").toString("utf8"));

    expect(spans).toHaveLength(1);
    expect(spans[0].src).toBe("rag");

    // Verify byte offsets point to the right content in the body
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].s, spans[0].e).toString("utf8");
    expect(extracted).toBe("secret RAG");
  });

  it("omits x-guard-trust when no session context", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    await pfetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [] }),
    });

    expect(capturedHeaders?.has("x-guard-trust")).toBe(false);
  });

  it("omits x-guard-trust when trust registry is empty", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await runWithContext(
      { trustRegistry: new TrustRegistry() },
      () =>
        pfetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: [] }),
        }),
    );

    expect(capturedHeaders?.has("x-guard-trust")).toBe(false);
  });

  it("omits x-guard-trust when content-type is not JSON", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await runWithContext(
      ctxWithTrust([{ content: "data", source: "rag" }]),
      () =>
        pfetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "multipart/form-data" },
          body: "data",
        }),
    );

    expect(capturedHeaders?.has("x-guard-trust")).toBe(false);
  });

  it("omits x-guard-trust when no content-type header", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());

    await runWithContext(
      ctxWithTrust([{ content: "data", source: "rag" }]),
      () => pfetch("https://api.openai.com/v1/chat/completions"),
    );

    expect(capturedHeaders?.has("x-guard-trust")).toBe(false);
  });

  it("handles +json content-type subtypes", async () => {
    let capturedHeaders: Headers | undefined;
    const baseFetch = vi.fn(
      async (input: string | URL | Request, init?: RequestInit) => {
        capturedHeaders = extractHeaders(input, init);
        return mockResponse();
      },
    );

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({ data: "payload" });

    await runWithContext(
      ctxWithTrust([{ content: "payload", source: "web" }]),
      () =>
        pfetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/vnd.api+json; charset=utf-8" },
          body,
        }),
    );

    expect(capturedHeaders?.has("x-guard-trust")).toBe(true);
  });

  it("body is still forwarded correctly when trust spans are computed", async () => {
    let capturedBody: string | null = null;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      if (input instanceof Request) {
        capturedBody = await input.text();
      }
      return mockResponse();
    });

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({
      messages: [{ role: "user", content: "untrusted" }],
    });

    await runWithContext(
      ctxWithTrust([{ content: "untrusted", source: "rag" }]),
      () =>
        pfetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        }),
    );

    expect(capturedBody).toBe(body);
  });

  it("failopen still works when trust spans are active", async () => {
    let callCount = 0;
    let fallbackBody: string | null = null;
    const baseFetch = vi.fn(async (input: string | URL | Request) => {
      callCount++;
      if (callCount === 1) {
        const sysErr = Object.assign(new Error("connect ECONNREFUSED"), {
          code: "ECONNREFUSED",
        });
        throw new TypeError("fetch failed", { cause: sysErr });
      }
      if (input instanceof Request) {
        fallbackBody = await input.text();
      }
      return mockResponse();
    });

    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const pfetch = createParapetFetch(baseFetch as typeof fetch, opts());
    const body = JSON.stringify({ messages: [{ content: "data" }] });

    await runWithContext(
      ctxWithTrust([{ content: "data", source: "rag" }]),
      () =>
        pfetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        }),
    );

    expect(callCount).toBe(2);
    expect(fallbackBody).toBe(body);

    warnSpy.mockRestore();
  });
});
