import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";
import { join } from "node:path";
import { EngineManager, type SidecarDeps } from "../src/sidecar.js";

// ---------------------------------------------------------------------------
// Mock ChildProcess
// ---------------------------------------------------------------------------

class MockProcess extends EventEmitter {
  pid = 12345;
  killed = false;
  killSignals: string[] = [];
  exitCode: number | null = null;
  /** If true, SIGTERM does not cause auto-exit (for testing SIGKILL escalation). */
  stubborn = false;
  private _exited = false;

  kill(signal?: string): boolean {
    const sig = signal ?? "SIGTERM";
    this.killSignals.push(sig);
    this.killed = true;
    // Auto-exit unless stubborn on SIGTERM.
    if (!this._exited && !(this.stubborn && sig === "SIGTERM")) {
      this._exited = true;
      Promise.resolve().then(() => this.emit("exit", 0, sig));
    }
    return true;
  }

  exitAfter(ms: number, code = 0): void {
    setTimeout(() => {
      if (!this._exited) {
        this._exited = true;
        this.emit("exit", code, null);
      }
    }, ms);
  }

  exitNow(code = 0): void {
    if (!this._exited) {
      this._exited = true;
      this.emit("exit", code, null);
    }
  }
}

// ---------------------------------------------------------------------------
// Path helpers — works on both Windows (backslash) and Unix (forward slash)
// ---------------------------------------------------------------------------

const HOME_DIR = "/home/test";
const PARAPET_DIR = join(HOME_DIR, ".parapet");
const PID_PATH = join(PARAPET_DIR, "engine.pid");
const META_PATH = join(PARAPET_DIR, "engine.meta");

// ---------------------------------------------------------------------------
// Mock SidecarDeps factory
// ---------------------------------------------------------------------------

interface MockFs {
  files: Map<string, string>;
}

function createMockDeps(overrides?: Partial<SidecarDeps>): {
  deps: SidecarDeps;
  fs: MockFs;
  spawnedProcesses: MockProcess[];
  signalHandlers: Map<string, Set<() => void>>;
  beforeExitHandlers: Set<() => void>;
  exitHandlers: Set<() => void>;
} {
  const fs: MockFs = { files: new Map() };
  const spawnedProcesses: MockProcess[] = [];
  const signalHandlers = new Map<string, Set<() => void>>();
  const beforeExitHandlers = new Set<() => void>();
  const exitHandlers = new Set<() => void>();

  // Default fetch: always returns 200 OK.
  const defaultFetch = async () => ({ ok: true, status: 200 }) as Response;

  const deps: SidecarDeps = {
    readFile: (p) => {
      const content = fs.files.get(p);
      if (content === undefined) throw new Error(`ENOENT: ${p}`);
      return content;
    },
    writeFile: (p, d) => {
      fs.files.set(p, d);
    },
    unlink: (p) => {
      fs.files.delete(p);
    },
    exists: (p) => fs.files.has(p),
    mkdirp: () => {},
    realpath: (p) => p,
    openLogFile: () => 999,
    closeFile: () => {},
    spawnProcess: () => {
      const proc = new MockProcess();
      spawnedProcesses.push(proc);
      return proc as unknown as ChildProcess;
    },
    fetch: defaultFetch,
    homedir: () => HOME_DIR,
    onSignal: (sig, handler) => {
      if (!signalHandlers.has(sig)) signalHandlers.set(sig, new Set());
      signalHandlers.get(sig)!.add(handler);
    },
    removeSignalHandler: (sig, handler) => {
      signalHandlers.get(sig)?.delete(handler);
    },
    onBeforeExit: (handler) => {
      beforeExitHandlers.add(handler);
    },
    removeBeforeExitHandler: (handler) => {
      beforeExitHandlers.delete(handler);
    },
    onExit: (handler) => {
      exitHandlers.add(handler);
    },
    removeExitHandler: (handler) => {
      exitHandlers.delete(handler);
    },
    ...overrides,
  };

  return { deps, fs, spawnedProcesses, signalHandlers, beforeExitHandlers, exitHandlers };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const CONFIG_CONTENT = "port: 9800\nrules: []";
const CONFIG_PATH = "/etc/parapet/parapet.yaml";

function setupConfigFile(
  fs: MockFs,
  path = CONFIG_PATH,
  content = CONFIG_CONTENT,
): void {
  fs.files.set(path, content);
}

function writeMeta(
  fs: MockFs,
  overrides?: Partial<{ port: number; configPath: string; configHash: string }>,
): void {
  const { createHash } = require("node:crypto");
  const meta = {
    port: overrides?.port ?? 9800,
    configPath: overrides?.configPath ?? CONFIG_PATH,
    configHash:
      overrides?.configHash ??
      createHash("sha256").update(CONFIG_CONTENT, "utf-8").digest("hex"),
    startedAt: new Date().toISOString(),
  };
  fs.files.set(META_PATH, JSON.stringify(meta));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("EngineManager", () => {
  // -----------------------------------------------------------------------
  // Process spawn
  // -----------------------------------------------------------------------

  describe("process spawn", () => {
    it("spawns engine and writes PID + meta files", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(spawnedProcesses).toHaveLength(1);
      expect(fs.files.has(PID_PATH)).toBe(true);
      expect(fs.files.get(PID_PATH)).toBe("12345");

      const meta = JSON.parse(fs.files.get(META_PATH)!);
      expect(meta.port).toBe(9800);
      expect(meta.configPath).toBe(CONFIG_PATH);
      expect(meta.configHash).toBeTypeOf("string");
      expect(meta.startedAt).toBeTypeOf("string");
      expect(mgr.isInitialized).toBe(true);

      await mgr.stop();
    });

    it("passes --config and --port to engine binary", async () => {
      let capturedArgs: string[] = [];
      const { deps, fs } = createMockDeps({
        spawnProcess: (_cmd, args) => {
          capturedArgs = args;
          return new MockProcess() as unknown as ChildProcess;
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 4242, deps });
      await mgr.start();

      expect(capturedArgs).toContain("--config");
      expect(capturedArgs).toContain(CONFIG_PATH);
      expect(capturedArgs).toContain("--port");
      expect(capturedArgs).toContain("4242");

      await mgr.stop();
    });
  });

  // -----------------------------------------------------------------------
  // Idempotent reuse
  // -----------------------------------------------------------------------

  describe("idempotent reuse", () => {
    it("same config → no-op (no second spawn)", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();
      expect(spawnedProcesses).toHaveLength(1);

      await mgr.start();
      expect(spawnedProcesses).toHaveLength(1); // No second spawn.

      await mgr.stop();
    });

    it("changed config content → restart", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();
      expect(spawnedProcesses).toHaveLength(1);

      // Change config content — new hash.
      fs.files.set(CONFIG_PATH, "port: 9800\nrules: [new_rule]");
      spawnedProcesses[0].exitAfter(10);

      await mgr.start();
      expect(spawnedProcesses).toHaveLength(2);

      spawnedProcesses[1].exitAfter(10);
      await mgr.stop();
    });
  });

  // -----------------------------------------------------------------------
  // Reuse from PID file
  // -----------------------------------------------------------------------

  describe("PID file reuse", () => {
    it("reuses existing engine when PID file + meta match and heartbeat succeeds", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      // Pre-existing PID + meta from a previous run.
      fs.files.set(PID_PATH, "99999");
      writeMeta(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(spawnedProcesses).toHaveLength(0); // Reused, no new spawn.
      expect(mgr.isInitialized).toBe(true);

      await mgr.stop();
    });

    it("cleans up stale PID file when heartbeat fails, then spawns new", async () => {
      let fetchCallCount = 0;
      const { deps, fs, spawnedProcesses } = createMockDeps({
        fetch: async () => {
          fetchCallCount++;
          if (fetchCallCount === 1) {
            // First heartbeat (reuse check) fails.
            throw new Error("connection refused");
          }
          // Subsequent calls (readiness) succeed.
          return { ok: true, status: 200 } as Response;
        },
      });
      setupConfigFile(fs);
      fs.files.set(PID_PATH, "99999");
      writeMeta(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(spawnedProcesses).toHaveLength(1); // Spawned new.

      await mgr.stop();
    });

    it("spawns new engine when PID file meta has mismatched config and port is free", async () => {
      let fetchCallCount = 0;
      const { deps, fs, spawnedProcesses } = createMockDeps({
        fetch: async () => {
          fetchCallCount++;
          // First call: mismatch port-check → port free (old engine dead).
          if (fetchCallCount === 1) throw new Error("connection refused");
          // Subsequent: readiness → success.
          return { ok: true, status: 200 } as Response;
        },
      });
      setupConfigFile(fs);
      fs.files.set(PID_PATH, "99999");
      writeMeta(fs, { configHash: "deadbeef" }); // Wrong hash.

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(spawnedProcesses).toHaveLength(1); // Port free → new spawn.

      await mgr.stop();
    });
  });

  // -----------------------------------------------------------------------
  // Config path canonicalization
  // -----------------------------------------------------------------------

  describe("config path canonicalization", () => {
    it("uses realpath for config identity — symlink resolves to canonical", async () => {
      const { deps, fs } = createMockDeps({
        realpath: (p) => {
          if (p.includes("symlink")) return CONFIG_PATH;
          return p;
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({
        configPath: "/etc/parapet/symlink.yaml",
        port: 9800,
        deps,
      });
      await mgr.start();

      const meta = JSON.parse(fs.files.get(META_PATH)!);
      expect(meta.configPath).toBe(CONFIG_PATH);

      await mgr.stop();
    });

    it("relative vs absolute paths with same realpath → same identity", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps({
        realpath: () => CONFIG_PATH,
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: "./parapet.yaml", port: 9800, deps });
      await mgr.start();
      expect(spawnedProcesses).toHaveLength(1);

      await mgr.start();
      expect(spawnedProcesses).toHaveLength(1); // Same canonical path — reuse.

      await mgr.stop();
    });
  });

  // -----------------------------------------------------------------------
  // Readiness barrier
  // -----------------------------------------------------------------------

  describe("readiness barrier", () => {
    it("polls with exponential backoff until engine is ready", async () => {
      let fetchCount = 0;
      const { deps, fs } = createMockDeps({
        fetch: async () => {
          fetchCount++;
          if (fetchCount < 4) throw new Error("not ready");
          return { ok: true, status: 200 } as Response;
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(fetchCount).toBeGreaterThanOrEqual(4);

      await mgr.stop();
    });

    it("throws if engine does not become ready within timeout", async () => {
      const { deps, fs } = createMockDeps({
        fetch: async () => {
          throw new Error("not ready");
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await expect(mgr.start()).rejects.toThrow(/did not become ready/);
    }, 15_000);
  });

  // -----------------------------------------------------------------------
  // Heartbeat
  // -----------------------------------------------------------------------

  describe("background heartbeat", () => {
    beforeEach(() => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    it("starts heartbeat after successful init", async () => {
      let heartbeatCount = 0;
      const { deps, fs } = createMockDeps({
        fetch: async () => {
          heartbeatCount++;
          return { ok: true, status: 200 } as Response;
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      const countAfterStart = heartbeatCount; // ≥1 (readiness check).

      await vi.advanceTimersByTimeAsync(10_000);
      expect(heartbeatCount).toBeGreaterThan(countAfterStart);

      await mgr.stop();
    });

    it("stops heartbeat on stop()", async () => {
      let heartbeatCount = 0;
      const { deps, fs } = createMockDeps({
        fetch: async () => {
          heartbeatCount++;
          return { ok: true, status: 200 } as Response;
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();
      await mgr.stop();
      const countAfterStop = heartbeatCount;

      await vi.advanceTimersByTimeAsync(30_000);
      expect(heartbeatCount).toBe(countAfterStop);
    });
  });

  // -----------------------------------------------------------------------
  // Graceful shutdown
  // -----------------------------------------------------------------------

  describe("graceful shutdown", () => {
    it("sends SIGTERM then waits for exit", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      const proc = spawnedProcesses[0];
      proc.exitAfter(50);

      await mgr.stop();

      expect(proc.killSignals).toContain("SIGTERM");
      expect(proc.killSignals).not.toContain("SIGKILL");
    });

    it("escalates to SIGKILL after grace period", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });

      const stubbornProc = new MockProcess();
      stubbornProc.stubborn = true; // Won't auto-exit on SIGTERM.
      const { deps, fs } = createMockDeps({
        spawnProcess: () => stubbornProc as unknown as ChildProcess,
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      const stopPromise = mgr.stop();
      await vi.advanceTimersByTimeAsync(6000);
      await stopPromise;

      expect(stubbornProc.killSignals).toContain("SIGTERM");
      expect(stubbornProc.killSignals).toContain("SIGKILL");

      vi.useRealTimers();
    });

    it("cleans up PID + meta files on stop", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(fs.files.has(PID_PATH)).toBe(true);
      expect(fs.files.has(META_PATH)).toBe(true);

      spawnedProcesses[0].exitAfter(10);
      await mgr.stop();

      expect(fs.files.has(PID_PATH)).toBe(false);
      expect(fs.files.has(META_PATH)).toBe(false);
    });
  });

  // -----------------------------------------------------------------------
  // Shutdown handler registration
  // -----------------------------------------------------------------------

  describe("shutdown handler management", () => {
    it("registers signal handlers on start", async () => {
      const { deps, fs, signalHandlers, beforeExitHandlers, exitHandlers } =
        createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(signalHandlers.get("SIGINT")?.size).toBe(1);
      expect(signalHandlers.get("SIGTERM")?.size).toBe(1);
      expect(beforeExitHandlers.size).toBe(1);
      expect(exitHandlers.size).toBe(1);

      await mgr.stop();
    });

    it("removes handlers on stop", async () => {
      const { deps, fs, signalHandlers, beforeExitHandlers, exitHandlers } =
        createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();
      await mgr.stop();

      expect(signalHandlers.get("SIGINT")?.size ?? 0).toBe(0);
      expect(signalHandlers.get("SIGTERM")?.size ?? 0).toBe(0);
      expect(beforeExitHandlers.size).toBe(0);
      expect(exitHandlers.size).toBe(0);
    });

    it("no handler leak on repeated init()", async () => {
      const { deps, fs, spawnedProcesses, signalHandlers } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();
      expect(signalHandlers.get("SIGINT")?.size).toBe(1);

      // Change config to force restart.
      fs.files.set(CONFIG_PATH, "port: 9800\nrules: [v2]");
      spawnedProcesses[0].exitAfter(10);
      await mgr.start();

      // Still exactly 1 handler per signal.
      expect(signalHandlers.get("SIGINT")?.size).toBe(1);
      expect(signalHandlers.get("SIGTERM")?.size).toBe(1);

      spawnedProcesses[1].exitAfter(10);
      await mgr.stop();
    });
  });

  // -----------------------------------------------------------------------
  // Engine binary resolution
  // -----------------------------------------------------------------------

  describe("engine binary resolution", () => {
    it("uses engineBinary option when provided", async () => {
      let capturedCmd = "";
      const { deps, fs } = createMockDeps({
        spawnProcess: (cmd) => {
          capturedCmd = cmd;
          return new MockProcess() as unknown as ChildProcess;
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9800,
        engineBinary: "/usr/local/bin/parapet-engine",
        deps,
      });
      await mgr.start();

      expect(capturedCmd).toBe("/usr/local/bin/parapet-engine");
      await mgr.stop();
    });

    it("falls back to PARAPET_ENGINE_PATH env var", async () => {
      const originalEnv = process.env["PARAPET_ENGINE_PATH"];
      process.env["PARAPET_ENGINE_PATH"] = "/custom/engine";

      let capturedCmd = "";
      const { deps, fs } = createMockDeps({
        spawnProcess: (cmd) => {
          capturedCmd = cmd;
          return new MockProcess() as unknown as ChildProcess;
        },
      });
      setupConfigFile(fs);

      try {
        const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
        await mgr.start();
        expect(capturedCmd).toBe("/custom/engine");
        await mgr.stop();
      } finally {
        if (originalEnv === undefined) delete process.env["PARAPET_ENGINE_PATH"];
        else process.env["PARAPET_ENGINE_PATH"] = originalEnv;
      }
    });

    it("defaults to 'parapet-engine' on PATH", async () => {
      const origEnv = process.env["PARAPET_ENGINE_PATH"];
      delete process.env["PARAPET_ENGINE_PATH"];

      let capturedCmd = "";
      const { deps, fs } = createMockDeps({
        spawnProcess: (cmd) => {
          capturedCmd = cmd;
          return new MockProcess() as unknown as ChildProcess;
        },
      });
      setupConfigFile(fs);

      try {
        const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
        await mgr.start();
        expect(capturedCmd).toBe("parapet-engine");
        await mgr.stop();
      } finally {
        if (origEnv !== undefined) process.env["PARAPET_ENGINE_PATH"] = origEnv;
      }
    });
  });

  // -----------------------------------------------------------------------
  // Default port
  // -----------------------------------------------------------------------

  describe("default port", () => {
    it("uses 9800 when port not specified", () => {
      const { deps } = createMockDeps();
      const mgr = new EngineManager({ configPath: CONFIG_PATH, deps });
      expect(mgr.port).toBe(9800);
    });
  });

  // -----------------------------------------------------------------------
  // stopSync (exit handler)
  // -----------------------------------------------------------------------

  describe("stopSync", () => {
    it("kills process and cleans up files", async () => {
      const { deps, fs } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(fs.files.has(PID_PATH)).toBe(true);

      mgr.stopSync();

      expect(mgr.isInitialized).toBe(false);
      expect(fs.files.has(PID_PATH)).toBe(false);
      expect(fs.files.has(META_PATH)).toBe(false);
    });
  });

  // -----------------------------------------------------------------------
  // Readiness failure rollback (finding 2)
  // -----------------------------------------------------------------------

  describe("readiness failure rollback", () => {
    it("cleans up process, log FD, and PID/meta on readiness failure", async () => {
      let closeFileCalled = false;
      const { deps, fs, spawnedProcesses } = createMockDeps({
        fetch: async () => {
          throw new Error("not ready");
        },
        closeFile: () => {
          closeFileCalled = true;
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await expect(mgr.start()).rejects.toThrow(/did not become ready/);

      // Process should have been killed.
      expect(spawnedProcesses[0].killed).toBe(true);
      // PID + meta files should be cleaned up.
      expect(fs.files.has(PID_PATH)).toBe(false);
      expect(fs.files.has(META_PATH)).toBe(false);
      // Log FD should have been closed.
      expect(closeFileCalled).toBe(true);
      // Manager should not be initialized.
      expect(mgr.isInitialized).toBe(false);
    }, 15_000);
  });

  // -----------------------------------------------------------------------
  // Ownership tracking (finding 3)
  // -----------------------------------------------------------------------

  describe("ownership: reused engines", () => {
    it("stop() does NOT delete PID/meta files for a reused engine", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      // Pre-existing PID + meta from another process.
      fs.files.set(PID_PATH, "99999");
      writeMeta(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      // Reused — no spawn.
      expect(spawnedProcesses).toHaveLength(0);

      await mgr.stop();

      // PID + meta should still exist — we don't own them.
      expect(fs.files.has(PID_PATH)).toBe(true);
      expect(fs.files.has(META_PATH)).toBe(true);
    });

    it("stop() DOES delete PID/meta files for a spawned engine", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(spawnedProcesses).toHaveLength(1);

      await mgr.stop();

      // We spawned it — files should be cleaned up.
      expect(fs.files.has(PID_PATH)).toBe(false);
      expect(fs.files.has(META_PATH)).toBe(false);
    });
  });

  // -----------------------------------------------------------------------
  // Port conflict detection (P1: incompatible engine on same port)
  // -----------------------------------------------------------------------

  describe("port conflict detection", () => {
    it("fails fast when mismatched PID/meta but port is occupied", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      // Pre-existing PID + meta with DIFFERENT config hash.
      fs.files.set(PID_PATH, "99999");
      writeMeta(fs, { configHash: "deadbeef" });

      // Heartbeat succeeds → port is occupied by incompatible engine.
      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await expect(mgr.start()).rejects.toThrow(/already in use/);

      // Should NOT have spawned a new process.
      expect(spawnedProcesses).toHaveLength(0);
    });

    it("spawns normally when mismatched PID/meta and port is free", async () => {
      let fetchCallCount = 0;
      const { deps, fs, spawnedProcesses } = createMockDeps({
        fetch: async () => {
          fetchCallCount++;
          // First call: PID mismatch check → port free.
          if (fetchCallCount === 1) throw new Error("connection refused");
          // Subsequent: readiness → success.
          return { ok: true, status: 200 } as Response;
        },
      });
      setupConfigFile(fs);

      fs.files.set(PID_PATH, "99999");
      writeMeta(fs, { configHash: "deadbeef" });

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await mgr.start();

      expect(spawnedProcesses).toHaveLength(1);
      await mgr.stop();
    });

    it("detects child process exit after readiness (stale port listener)", async () => {
      const deadProc = new MockProcess();
      // Simulate process that exits immediately after spawn.
      Object.defineProperty(deadProc, "exitCode", { value: 1 });

      const { deps, fs } = createMockDeps({
        spawnProcess: () => deadProc as unknown as ChildProcess,
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });
      await expect(mgr.start()).rejects.toThrow(/exited immediately/);
    });
  });

  // -----------------------------------------------------------------------
  // Canonical identity accessors
  // -----------------------------------------------------------------------

  describe("canonical identity", () => {
    it("exposes canonicalConfigPath and configHash after start()", async () => {
      const { deps, fs } = createMockDeps();
      setupConfigFile(fs);

      const mgr = new EngineManager({ configPath: CONFIG_PATH, port: 9800, deps });

      // Before start — null.
      expect(mgr.canonicalConfigPath).toBeNull();
      expect(mgr.configHash).toBeNull();

      await mgr.start();

      expect(mgr.canonicalConfigPath).toBe(CONFIG_PATH);
      expect(mgr.configHash).toBeTypeOf("string");
      expect(mgr.configHash!.length).toBe(64); // SHA-256 hex.

      await mgr.stop();
    });

    it("matchesCanonicalConfig resolves via realpath", async () => {
      const { deps, fs } = createMockDeps({
        realpath: (p) => {
          if (p.includes("symlink")) return CONFIG_PATH;
          return p;
        },
      });
      setupConfigFile(fs);

      const mgr = new EngineManager({
        configPath: "/etc/parapet/symlink.yaml",
        port: 9800,
        deps,
      });
      await mgr.start();

      // Both resolve to the same canonical path.
      expect(mgr.matchesCanonicalConfig("/etc/parapet/symlink.yaml")).toBe(true);
      expect(mgr.matchesCanonicalConfig(CONFIG_PATH)).toBe(true);
      // Different canonical path → false.
      expect(mgr.matchesCanonicalConfig("/other/config.yaml")).toBe(false);

      await mgr.stop();
    });
  });
});
