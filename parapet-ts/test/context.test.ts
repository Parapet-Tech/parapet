import { describe, it, expect } from "vitest";
import { session, untrusted } from "../src/index.js";
import { getContext, runWithContext } from "../src/context.js";
import { TrustRegistry } from "../src/trust.js";

// ---------------------------------------------------------------------------
// session(): basic context propagation
// ---------------------------------------------------------------------------

describe("session context propagation", () => {
  it("provides context inside the callback", async () => {
    let captured: ReturnType<typeof getContext>;
    await session({ userId: "u_1", role: "admin" }, async () => {
      captured = getContext();
    });
    expect(captured!).toBeDefined();
    expect(captured!.userId).toBe("u_1");
    expect(captured!.role).toBe("admin");
    expect(captured!.trustRegistry).toBeInstanceOf(TrustRegistry);
  });

  it("propagates context through async chains", async () => {
    let captured: ReturnType<typeof getContext>;

    await session({ userId: "u_2" }, async () => {
      // Simulate async work (setTimeout, awaited promise, etc.)
      await new Promise((resolve) => setTimeout(resolve, 10));
      captured = getContext();
    });

    expect(captured!).toBeDefined();
    expect(captured!.userId).toBe("u_2");
  });

  it("propagates context through nested awaits", async () => {
    const results: string[] = [];

    await session({ userId: "u_3" }, async () => {
      results.push(getContext()!.userId!);
      await Promise.resolve();
      results.push(getContext()!.userId!);
      await new Promise((resolve) => setTimeout(resolve, 5));
      results.push(getContext()!.userId!);
    });

    expect(results).toEqual(["u_3", "u_3", "u_3"]);
  });

  it("context is undefined outside a session", () => {
    expect(getContext()).toBeUndefined();
  });

  it("context is cleaned up after session ends", async () => {
    await session({ userId: "u_4" }, async () => {
      expect(getContext()).toBeDefined();
    });
    expect(getContext()).toBeUndefined();
  });

  it("context is cleaned up after session throws", async () => {
    try {
      await session({}, async () => {
        throw new Error("boom");
      });
    } catch {
      // expected
    }
    expect(getContext()).toBeUndefined();
  });

  it("works with synchronous callbacks", () => {
    let captured: ReturnType<typeof getContext>;
    session({ userId: "sync_user" }, () => {
      captured = getContext();
    });
    expect(captured!).toBeDefined();
    expect(captured!.userId).toBe("sync_user");
  });

  it("returns the callback's return value", async () => {
    const result = await session({ userId: "u_5" }, async () => {
      return 42;
    });
    expect(result).toBe(42);
  });

  it("returns sync callback's return value", () => {
    const result = session({}, () => "hello");
    expect(result).toBe("hello");
  });
});

// ---------------------------------------------------------------------------
// session(): nested sessions
// ---------------------------------------------------------------------------

describe("nested sessions", () => {
  it("inner session overrides outer context", async () => {
    await session({ userId: "outer", role: "admin" }, async () => {
      expect(getContext()!.userId).toBe("outer");

      await session({ userId: "inner", role: "viewer" }, async () => {
        expect(getContext()!.userId).toBe("inner");
        expect(getContext()!.role).toBe("viewer");
      });

      // Outer context restored
      expect(getContext()!.userId).toBe("outer");
      expect(getContext()!.role).toBe("admin");
    });
  });

  it("each session gets a fresh TrustRegistry", async () => {
    await session({}, async () => {
      const outerRegistry = getContext()!.trustRegistry;
      outerRegistry.register("outer data", "rag");

      await session({}, async () => {
        const innerRegistry = getContext()!.trustRegistry;
        expect(innerRegistry).not.toBe(outerRegistry);
        expect(innerRegistry.entryCount).toBe(0);
      });

      // Outer registry unchanged
      expect(outerRegistry.entryCount).toBe(1);
    });
  });

  it("inner session does not leak to outer", async () => {
    await session({ userId: "outer" }, async () => {
      await session({ userId: "inner" }, async () => {
        untrusted("secret", "rag");
      });

      // Outer session's registry should be unaffected
      expect(getContext()!.trustRegistry.entryCount).toBe(0);
    });
  });
});

// ---------------------------------------------------------------------------
// untrusted()
// ---------------------------------------------------------------------------

describe("untrusted()", () => {
  it("registers content in session's trust registry", async () => {
    await session({}, async () => {
      untrusted("some data", "rag");
      expect(getContext()!.trustRegistry.entryCount).toBe(1);
    });
  });

  it("returns content unchanged", async () => {
    await session({}, async () => {
      const input = "RAG content here";
      expect(untrusted(input, "rag")).toBe(input);
    });
  });

  it("supports multiple registrations in one session", async () => {
    await session({}, async () => {
      untrusted("data1", "rag");
      untrusted("data2", "web");
      untrusted("data3", "user_input");
      expect(getContext()!.trustRegistry.entryCount).toBe(3);
    });
  });

  it("throws when called outside a session", () => {
    expect(() => untrusted("data", "rag")).toThrow(
      "parapet.untrusted() called outside a session",
    );
  });

  it("works with inline usage pattern", async () => {
    await session({}, async () => {
      const query = "What is Parapet?";
      const ragSnippet = "Parapet is a security proxy.";

      const prompt = `Context: ${untrusted(ragSnippet, "rag")}\n\nQ: ${query}`;

      expect(prompt).toContain(ragSnippet);
      expect(getContext()!.trustRegistry.entryCount).toBe(1);
    });
  });
});

// ---------------------------------------------------------------------------
// runWithContext (low-level)
// ---------------------------------------------------------------------------

describe("runWithContext", () => {
  it("makes context available via getContext()", () => {
    const registry = new TrustRegistry();
    runWithContext({ userId: "u_1", trustRegistry: registry }, () => {
      const ctx = getContext();
      expect(ctx).toBeDefined();
      expect(ctx!.userId).toBe("u_1");
      expect(ctx!.trustRegistry).toBe(registry);
    });
  });

  it("restores previous context after callback", () => {
    const outerRegistry = new TrustRegistry();
    const innerRegistry = new TrustRegistry();

    runWithContext({ trustRegistry: outerRegistry }, () => {
      expect(getContext()!.trustRegistry).toBe(outerRegistry);

      runWithContext({ trustRegistry: innerRegistry }, () => {
        expect(getContext()!.trustRegistry).toBe(innerRegistry);
      });

      expect(getContext()!.trustRegistry).toBe(outerRegistry);
    });
  });
});
