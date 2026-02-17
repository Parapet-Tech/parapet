import { describe, it, expect } from "vitest";
import { buildBaggageHeader, buildTrustHeader } from "../src/header.js";
import type { TrustSpan } from "../src/types.js";

describe("buildBaggageHeader", () => {
  it("returns userId and role", () => {
    expect(buildBaggageHeader({ userId: "u_1", role: "admin" })).toBe(
      "user_id=u_1,role=admin",
    );
  });

  it("returns userId only", () => {
    expect(buildBaggageHeader({ userId: "u_1" })).toBe("user_id=u_1");
  });

  it("returns role only", () => {
    expect(buildBaggageHeader({ role: "viewer" })).toBe("role=viewer");
  });

  it("returns undefined for empty options", () => {
    expect(buildBaggageHeader({})).toBeUndefined();
  });

  it("percent-encodes special characters", () => {
    expect(buildBaggageHeader({ userId: "user with spaces" })).toBe(
      "user_id=user%20with%20spaces",
    );
  });

  it("percent-encodes commas and semicolons", () => {
    expect(buildBaggageHeader({ role: "a,b;c=d" })).toBe(
      "role=a%2Cb%3Bc%3Dd",
    );
  });

  it("percent-encodes chars that encodeURIComponent leaves unescaped", () => {
    // !'()* are not encoded by encodeURIComponent but must be encoded
    // for strict RFC 3986 parity with Python SDK.
    expect(buildBaggageHeader({ userId: "it's!(a)*test" })).toBe(
      "user_id=it%27s%21%28a%29%2Atest",
    );
  });
});

// ---------------------------------------------------------------------------
// buildTrustHeader
// ---------------------------------------------------------------------------

describe("buildTrustHeader", () => {
  it("returns undefined for empty spans", () => {
    expect(buildTrustHeader([])).toBeUndefined();
  });

  it("returns inline: prefixed base64 value", () => {
    const spans: TrustSpan[] = [{ start: 0, end: 42, source: "rag" }];
    const result = buildTrustHeader(spans);

    expect(result).toBeDefined();
    expect(result).toMatch(/^inline:/);
  });

  it("uses compact keys (s, e, src) in the JSON payload", () => {
    const spans: TrustSpan[] = [{ start: 10, end: 50, source: "web" }];
    const result = buildTrustHeader(spans)!;

    // Decode to verify structure
    const base64 = result.slice("inline:".length);
    const json = Buffer.from(base64, "base64").toString("utf8");
    const parsed = JSON.parse(json);

    expect(parsed).toEqual([{ s: 10, e: 50, src: "web" }]);
  });

  it("serializes multiple spans preserving order", () => {
    const spans: TrustSpan[] = [
      { start: 0, end: 10, source: "rag" },
      { start: 20, end: 30, source: "user_input" },
    ];
    const result = buildTrustHeader(spans)!;

    const base64 = result.slice("inline:".length);
    const parsed = JSON.parse(Buffer.from(base64, "base64").toString("utf8"));

    expect(parsed).toEqual([
      { s: 0, e: 10, src: "rag" },
      { s: 20, e: 30, src: "user_input" },
    ]);
  });

  it("roundtrips through base64 correctly", () => {
    const spans: TrustSpan[] = [
      { start: 100, end: 200, source: "search" },
    ];
    const result = buildTrustHeader(spans)!;

    const base64 = result.slice("inline:".length);
    const decoded = Buffer.from(base64, "base64").toString("utf8");
    const reparsed = JSON.parse(decoded);

    expect(reparsed[0].s).toBe(100);
    expect(reparsed[0].e).toBe(200);
    expect(reparsed[0].src).toBe("search");
  });

  it("returns undefined when final header value exceeds 4KB", () => {
    // Each span produces ~30+ bytes of JSON. 200 spans with large offsets
    // produce well over 4KB after base64 encoding.
    const spans: TrustSpan[] = Array.from({ length: 200 }, (_, i) => ({
      start: i * 1000,
      end: (i + 1) * 1000,
      source: "rag_source_label",
    }));
    expect(buildTrustHeader(spans)).toBeUndefined();
  });

  it("4KB limit is on final header value, not raw JSON", () => {
    // Find the boundary: binary search for max span count that fits
    const makeSpans = (n: number): TrustSpan[] =>
      Array.from({ length: n }, (_, i) => ({
        start: i,
        end: i + 1,
        source: "x",
      }));

    let lo = 1;
    let hi = 300;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (buildTrustHeader(makeSpans(mid)) !== undefined) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }

    // lo is max count that fits — verify boundary
    const fitting = buildTrustHeader(makeSpans(lo))!;
    expect(fitting).toBeDefined();
    expect(fitting.length).toBeLessThanOrEqual(4096);

    expect(buildTrustHeader(makeSpans(lo + 1))).toBeUndefined();
  });
});
