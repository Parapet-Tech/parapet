import { describe, it, expect } from "vitest";
import { buildBaggageHeader } from "../src/header.js";

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
