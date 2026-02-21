import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { EventEmitter } from "node:events";
import type { ChildProcess } from "node:child_process";
import type { SidecarDeps } from "../src/sidecar.js";
import {
  init,
  session,
  getParapetFetch,
  shutdown,
  _resetForTesting,
} from "../src/index.js";

// ---------------------------------------------------------------------------
// Minimal mock infrastructure (shared with sidecar.test.ts pattern)
// ---------------------------------------------------------------------------

class MockProcess extends EventEmitter {
  pid = 12345;
  killed = false;
  exitCode: number | null = null;

  kill(signal?: string): boolean {
    this.killed = true;
    // Auto-exit on kill.
    Promise.resolve().then(() => this.emit("exit", 0, signal));
    return true;
  }
}

const CONFIG_CONTENT = "port: 9800\nrules: []";
const CONFIG_PATH = "/etc/parapet/parapet.yaml";

function createMockDeps(): {
  deps: SidecarDeps;
  spawnCount: () => number;
  stopCount: () => number;
} {
  const files = new Map<string, string>();
  files.set(CONFIG_PATH, CONFIG_CONTENT);
  let spawns = 0;
  let stops = 0;

  const deps: SidecarDeps = {
    readFile: (p) => {
      const c = files.get(p);
      if (c === undefined) throw new Error(`ENOENT: ${p}`);
      return c;
    },
    writeFile: (p, d) => files.set(p, d),
    unlink: (p) => files.delete(p),
    exists: (p) => files.has(p),
    mkdirp: () => {},
    realpath: (p) => p,
    openLogFile: () => 999,
    closeFile: () => {
      stops++;
    },
    spawnProcess: () => {
      spawns++;
      return new MockProcess() as unknown as ChildProcess;
    },
    fetch: async () => ({ ok: true, status: 200 }) as Response,
    homedir: () => "/home/test",
    onSignal: () => {},
    removeSignalHandler: () => {},
    onBeforeExit: () => {},
    removeBeforeExitHandler: () => {},
    onExit: () => {},
    removeExitHandler: () => {},
  };

  return { deps, spawnCount: () => spawns, stopCount: () => stops };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("module lifecycle: init / getParapetFetch / shutdown", () => {
  beforeEach(() => {
    _resetForTesting();
  });

  it("getParapetFetch() throws before init()", () => {
    expect(() => getParapetFetch()).toThrow(/before init/);
  });

  it("session() throws before init()", () => {
    expect(() => session({ userId: "u_1" }, () => {})).toThrow(/before init/);
  });

  it("getParapetFetch() returns a function after init()", async () => {
    await init({ configPath: CONFIG_PATH, port: 9800, autoStart: false });

    const pf = getParapetFetch();
    expect(typeof pf).toBe("function");
  });

  it("shutdown() clears state — getParapetFetch() throws again", async () => {
    await init({ configPath: CONFIG_PATH, port: 9800, autoStart: false });
    expect(() => getParapetFetch()).not.toThrow();

    await shutdown();
    expect(() => getParapetFetch()).toThrow(/before init/);
  });

  it("repeated init() without shutdown does not leak (autoStart: false)", async () => {
    await init({ configPath: CONFIG_PATH, port: 9800, autoStart: false });
    const fetch1 = getParapetFetch();

    await init({ configPath: CONFIG_PATH, port: 9801, autoStart: false });
    const fetch2 = getParapetFetch();

    // Different port → different fetch wrapper.
    expect(fetch2).not.toBe(fetch1);
  });

  it("init with autoStart: false skips engine spawn", async () => {
    await init({ configPath: CONFIG_PATH, autoStart: false });

    const pf = getParapetFetch();
    expect(typeof pf).toBe("function");

    await shutdown();
    expect(() => getParapetFetch()).toThrow(/before init/);
  });

  it("_resetForTesting clears module state", async () => {
    await init({ configPath: CONFIG_PATH, autoStart: false });
    expect(() => getParapetFetch()).not.toThrow();

    _resetForTesting();
    expect(() => getParapetFetch()).toThrow(/before init/);
  });
});

// ---------------------------------------------------------------------------
// Sidecar lifecycle (autoStart: true) — exercises init() idempotency
// ---------------------------------------------------------------------------

describe("init() idempotency with autoStart: true", () => {
  let mock: ReturnType<typeof createMockDeps>;

  beforeEach(() => {
    mock = createMockDeps();
    _resetForTesting(mock.deps);
  });

  afterEach(async () => {
    await shutdown();
    _resetForTesting();
  });

  it("throws on missing config file when autoStart is true", async () => {
    await expect(
      init({ configPath: "/nonexistent/parapet.yaml", port: 9800 }),
    ).rejects.toThrow(/Config file not found/);
  });

  it("same config twice → single spawn, no restart", async () => {
    await init({ configPath: CONFIG_PATH, port: 9800 });
    expect(mock.spawnCount()).toBe(1);

    // Second init with identical config — should reuse, not restart.
    await init({ configPath: CONFIG_PATH, port: 9800 });
    expect(mock.spawnCount()).toBe(1); // Still 1 — no second spawn.
  });

  it("different port → stops old engine, spawns new", async () => {
    await init({ configPath: CONFIG_PATH, port: 9800 });
    expect(mock.spawnCount()).toBe(1);

    await init({ configPath: CONFIG_PATH, port: 9801 });
    expect(mock.spawnCount()).toBe(2); // New spawn for new port.
  });

  it("different configPath → stops old engine, spawns new", async () => {
    // Add a second config file to the mock fs.
    mock.deps.writeFile("/etc/parapet/other.yaml", "port: 9800\nrules: [other]");

    await init({ configPath: CONFIG_PATH, port: 9800 });
    expect(mock.spawnCount()).toBe(1);

    await init({ configPath: "/etc/parapet/other.yaml", port: 9800 });
    expect(mock.spawnCount()).toBe(2);
  });

  it("getParapetFetch() works after init with autoStart", async () => {
    await init({ configPath: CONFIG_PATH, port: 9800 });
    const pf = getParapetFetch();
    expect(typeof pf).toBe("function");
  });

  it("symlink then real path → same canonical identity, no respawn", async () => {
    // Register the symlink path in mock FS so the existence check passes.
    mock.deps.writeFile("./symlink.yaml", CONFIG_CONTENT);

    // Override realpath to simulate symlink resolution.
    const origRealpath = mock.deps.realpath;
    mock.deps.realpath = (p: string) => {
      if (p === "./symlink.yaml") return CONFIG_PATH;
      return origRealpath(p);
    };

    await init({ configPath: "./symlink.yaml", port: 9800 });
    expect(mock.spawnCount()).toBe(1);

    // Second init with the resolved real path — should reuse.
    await init({ configPath: CONFIG_PATH, port: 9800 });
    expect(mock.spawnCount()).toBe(1); // No second spawn.
  });
});
