import { describe, it, expect } from "vitest";
import { TrustRegistry, isJsonContentType } from "../src/trust.js";

// ---------------------------------------------------------------------------
// TrustRegistry: registration
// ---------------------------------------------------------------------------

describe("TrustRegistry registration", () => {
  it("returns content unchanged", () => {
    const reg = new TrustRegistry();
    const content = "some RAG content";
    expect(reg.register(content, "rag")).toBe(content);
  });

  it("increments entry count", () => {
    const reg = new TrustRegistry();
    expect(reg.entryCount).toBe(0);
    expect(reg.hasEntries).toBe(false);

    reg.register("a", "rag");
    expect(reg.entryCount).toBe(1);
    expect(reg.hasEntries).toBe(true);

    reg.register("b", "web");
    expect(reg.entryCount).toBe(2);
  });

  it("ignores empty strings", () => {
    const reg = new TrustRegistry();
    const result = reg.register("", "rag");
    expect(result).toBe("");
    expect(reg.entryCount).toBe(0);
  });

  it("throws when full", () => {
    const reg = new TrustRegistry(2);
    reg.register("a", "rag");
    reg.register("b", "rag");
    expect(() => reg.register("c", "rag")).toThrow("TrustRegistry full");
  });

  it("clear removes all entries", () => {
    const reg = new TrustRegistry();
    reg.register("a", "rag");
    reg.register("b", "web");
    reg.clear();
    expect(reg.entryCount).toBe(0);
    expect(reg.hasEntries).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// TrustRegistry: findSpans — basic
// ---------------------------------------------------------------------------

describe("TrustRegistry findSpans", () => {
  it("returns empty array when no entries registered", () => {
    const reg = new TrustRegistry();
    expect(reg.findSpans('{"content":"hello"}')).toEqual([]);
  });

  it("returns empty array when registered string is not in body", () => {
    const reg = new TrustRegistry();
    reg.register("missing", "rag");
    expect(reg.findSpans('{"content":"hello"}')).toEqual([]);
  });

  it("finds a simple ASCII string in JSON body", () => {
    const reg = new TrustRegistry();
    reg.register("hello world", "rag");

    const body = '{"content":"hello world"}';
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    expect(spans[0].source).toBe("rag");

    // Verify byte offsets point to the right content
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe("hello world");
  });

  it("finds multiple occurrences of the same string", () => {
    const reg = new TrustRegistry();
    reg.register("foo", "rag");

    const body = '{"a":"foo","b":"foo"}';
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(2);
    expect(spans[0].start).toBeLessThan(spans[1].start);

    const buf = Buffer.from(body, "utf8");
    for (const span of spans) {
      expect(buf.slice(span.start, span.end).toString("utf8")).toBe("foo");
    }
  });

  it("finds spans from multiple registered entries", () => {
    const reg = new TrustRegistry();
    reg.register("alpha", "rag");
    reg.register("beta", "web");

    const body = '{"a":"alpha","b":"beta"}';
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(2);
    expect(spans[0].source).toBe("alpha" <= "beta" ? "rag" : "web");

    const buf = Buffer.from(body, "utf8");
    const values = spans.map((s) =>
      buf.slice(s.start, s.end).toString("utf8"),
    );
    expect(values).toContain("alpha");
    expect(values).toContain("beta");
  });

  it("returns spans sorted by start offset", () => {
    const reg = new TrustRegistry();
    // Register in reverse order of where they appear
    reg.register("zzz", "last");
    reg.register("aaa", "first");

    const body = '{"x":"aaa","y":"zzz"}';
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(2);
    expect(spans[0].start).toBeLessThan(spans[1].start);
    expect(spans[0].source).toBe("first");
    expect(spans[1].source).toBe("last");
  });
});

// ---------------------------------------------------------------------------
// TrustRegistry: findSpans — JSON escaping
// ---------------------------------------------------------------------------

describe("TrustRegistry findSpans — JSON escaping", () => {
  it("finds strings with newlines (escaped as \\n in JSON)", () => {
    const reg = new TrustRegistry();
    reg.register("line1\nline2", "rag");

    const body = JSON.stringify({ content: "line1\nline2" });
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe("line1\\nline2");
  });

  it("finds strings with tabs (escaped as \\t in JSON)", () => {
    const reg = new TrustRegistry();
    reg.register("col1\tcol2", "rag");

    const body = JSON.stringify({ content: "col1\tcol2" });
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe("col1\\tcol2");
  });

  it("finds strings with double quotes (escaped as \\\" in JSON)", () => {
    const reg = new TrustRegistry();
    reg.register('say "hello"', "rag");

    const body = JSON.stringify({ content: 'say "hello"' });
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe('say \\"hello\\"');
  });

  it("finds strings with backslashes (escaped as \\\\ in JSON)", () => {
    const reg = new TrustRegistry();
    reg.register("path\\to\\file", "rag");

    const body = JSON.stringify({ content: "path\\to\\file" });
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe("path\\\\to\\\\file");
  });
});

// ---------------------------------------------------------------------------
// TrustRegistry: findSpans — UTF-8 byte offsets
// ---------------------------------------------------------------------------

describe("TrustRegistry findSpans — byte offsets", () => {
  it("computes correct byte offsets for ASCII content", () => {
    const reg = new TrustRegistry();
    reg.register("abc", "rag");

    const body = '{"k":"abc"}';
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    // In ASCII, char offset === byte offset
    const charIdx = body.indexOf("abc");
    expect(spans[0].start).toBe(charIdx);
    expect(spans[0].end).toBe(charIdx + 3);
  });

  it("computes correct byte offsets with multi-byte prefix (emoji)", () => {
    const reg = new TrustRegistry();
    reg.register("target", "rag");

    // Emoji before the target: 🔥 is 4 bytes in UTF-8
    const body = '{"emoji":"🔥","data":"target"}';
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe("target");
  });

  it("computes correct byte offsets with CJK characters", () => {
    const reg = new TrustRegistry();
    reg.register("data", "rag");

    // 日本語 — each CJK char is 3 bytes in UTF-8
    const body = '{"label":"日本語","val":"data"}';
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe("data");
  });

  it("computes correct byte offsets for multi-byte content itself", () => {
    const reg = new TrustRegistry();
    reg.register("日本語", "rag");

    const body = '{"content":"日本語"}';
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe("日本語");
    // 3 CJK chars × 3 bytes each = 9 bytes
    expect(spans[0].end - spans[0].start).toBe(9);
  });

  it("handles mixed multi-byte and escaped content", () => {
    const reg = new TrustRegistry();
    reg.register("hello\nworld", "rag");

    // Emoji prefix + escaped newline in the target
    const body = JSON.stringify({ prefix: "🔥", content: "hello\nworld" });
    const spans = reg.findSpans(body);

    expect(spans).toHaveLength(1);
    const buf = Buffer.from(body, "utf8");
    const extracted = buf.slice(spans[0].start, spans[0].end).toString("utf8");
    expect(extracted).toBe("hello\\nworld");
  });
});

// ---------------------------------------------------------------------------
// isJsonContentType
// ---------------------------------------------------------------------------

describe("isJsonContentType", () => {
  it("returns true for application/json", () => {
    expect(isJsonContentType("application/json")).toBe(true);
  });

  it("returns true for application/json with charset parameter", () => {
    expect(isJsonContentType("application/json; charset=utf-8")).toBe(true);
  });

  it("returns true for application/json with extra whitespace", () => {
    expect(isJsonContentType("  application/json ; charset=utf-8  ")).toBe(
      true,
    );
  });

  it("returns true for case-insensitive APPLICATION/JSON", () => {
    expect(isJsonContentType("APPLICATION/JSON")).toBe(true);
    expect(isJsonContentType("Application/Json")).toBe(true);
  });

  it("returns true for +json subtypes", () => {
    expect(isJsonContentType("application/vnd.api+json")).toBe(true);
    expect(isJsonContentType("application/hal+json")).toBe(true);
    expect(isJsonContentType("application/ld+json")).toBe(true);
  });

  it("returns true for +json subtypes with parameters", () => {
    expect(
      isJsonContentType("application/vnd.api+json; charset=utf-8"),
    ).toBe(true);
  });

  it("returns false for null and undefined", () => {
    expect(isJsonContentType(null)).toBe(false);
    expect(isJsonContentType(undefined)).toBe(false);
  });

  it("returns false for empty string", () => {
    expect(isJsonContentType("")).toBe(false);
  });

  it("returns false for non-JSON types", () => {
    expect(isJsonContentType("text/plain")).toBe(false);
    expect(isJsonContentType("text/html")).toBe(false);
    expect(isJsonContentType("multipart/form-data")).toBe(false);
    expect(isJsonContentType("application/xml")).toBe(false);
    expect(isJsonContentType("application/octet-stream")).toBe(false);
  });

  it("returns false for text/json (not application/json or +json)", () => {
    expect(isJsonContentType("text/json")).toBe(false);
  });

  it("returns false for malformed media types without slash", () => {
    expect(isJsonContentType("foo+json")).toBe(false);
    expect(isJsonContentType("+json")).toBe(false);
    expect(isJsonContentType("json")).toBe(false);
  });

  it("returns false for empty subtype prefix before +json", () => {
    expect(isJsonContentType("application/+json")).toBe(false);
  });

  it("returns false for slash-only or missing type", () => {
    expect(isJsonContentType("/vnd.api+json")).toBe(false);
    expect(isJsonContentType("/+json")).toBe(false);
  });
});
