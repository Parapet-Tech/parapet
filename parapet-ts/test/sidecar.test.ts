import { describe, it, expect, vi, beforeEach, afterEach, type MockInstance } from "vitest";
import type { ChildProcess } from "node:child_process";
import { spawn as spawnChild } from "node:child_process";
import { EventEmitter } from "node:events";
import {
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
  chmodSync,
} from "node:fs";
import { createServer } from "node:net";
import { join } from "node:path";
import {
  EngineManager,
  EngineRecoveryError,
  defaultRequestControlDisable,
  defaultRequestControlDisableSync,
  type ControlGateResult,
  type SidecarDeps,
} from "../src/sidecar.js";

const APPROVED_CLEAR: ControlGateResult = { kind: "approved", fileCleanup: "clear" };
const APPROVED_NOT_APPLICABLE: ControlGateResult = {
  kind: "approved",
  fileCleanup: "not_applicable",
};
const REFUSED: ControlGateResult = { kind: "refused" };

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

    it("uses the SDK source hash when engine.owner.json carries a different contract hash", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps({
        requestControlDisable: async () => APPROVED_CLEAR,
      });
      setupConfigFile(fs);

      const stateDir = "/var/tmp/parapet-sdk-config-hash";
      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9800,
        stateDir,
        deps,
      });
      await mgr.start();
      expect(spawnedProcesses).toHaveLength(1);

      const metaPath = join(stateDir, "engine.meta");
      const ownerPath = join(stateDir, "engine.owner.json");
      const meta = JSON.parse(fs.files.get(metaPath)!);
      meta.configHash = "engine-contract-hash";
      fs.files.set(metaPath, JSON.stringify(meta));
      fs.files.set(
        ownerPath,
        JSON.stringify({
          schema: "parapet-engine-owner/v0",
          pid: 12345,
          port: 9800,
          config_hash: "engine-contract-hash",
        }),
      );

      fs.files.set(CONFIG_PATH, "port: 9800\nrules: [changed]");
      await mgr.start();

      expect(spawnedProcesses).toHaveLength(2);
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

  // -----------------------------------------------------------------------
  // Explicit stateDir and engineArgs (1D T1)
  // -----------------------------------------------------------------------

  describe("stateDir and engineArgs", () => {
    it("uses stateDir instead of ~/.parapet and forwards --state-dir + engineArgs", async () => {
      let capturedArgs: string[] = [];
      let mkdirpPath = "";
      const { deps, fs } = createMockDeps({
        mkdirp: (p) => {
          mkdirpPath = p;
        },
        spawnProcess: (_cmd, args) => {
          capturedArgs = args;
          return new MockProcess() as unknown as ChildProcess;
        },
      });
      setupConfigFile(fs);

      const stateDir = "/var/tmp/parapet-state-1";
      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9811,
        stateDir,
        engineArgs: ["--offline-fixture", "v0", "--ready-file", "ready.port"],
        deps,
      });

      await mgr.start();

      expect(mgr.stateDir).toBe(stateDir);
      expect(mgr.engineArgs).toEqual([
        "--offline-fixture",
        "v0",
        "--ready-file",
        "ready.port",
      ]);
      expect(mkdirpPath).toBe(stateDir);
      expect(capturedArgs).toEqual([
        "--config",
        CONFIG_PATH,
        "--port",
        "9811",
        "--state-dir",
        stateDir,
        "--offline-fixture",
        "v0",
        "--ready-file",
        "ready.port",
      ]);
      // Writes PID + meta inside stateDir, never ~/.parapet.
      expect(fs.files.has(join(stateDir, "engine.pid"))).toBe(true);
      expect(fs.files.has(join(stateDir, "engine.meta"))).toBe(true);
      expect(fs.files.has(PID_PATH)).toBe(false);
      expect(fs.files.has(META_PATH)).toBe(false);

      await mgr.stop();
      expect(fs.files.has(join(stateDir, "engine.pid"))).toBe(false);
      expect(fs.files.has(join(stateDir, "engine.meta"))).toBe(false);
    });

    it("extracts --state-dir from engineArgs without duplicating the flag", async () => {
      let capturedArgs: string[] = [];
      const { deps, fs } = createMockDeps({
        spawnProcess: (_cmd, args) => {
          capturedArgs = args;
          return new MockProcess() as unknown as ChildProcess;
        },
      });
      setupConfigFile(fs);

      const stateDir = "/var/tmp/parapet-state-from-args";
      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9812,
        engineArgs: ["--state-dir", stateDir, "--offline-fixture", "v0"],
        deps,
      });

      await mgr.start();

      expect(mgr.stateDir).toBe(stateDir);
      expect(capturedArgs.filter((a) => a === "--state-dir")).toHaveLength(1);
      expect(fs.files.has(join(stateDir, "engine.pid"))).toBe(true);

      await mgr.stop();
    });
  });

  // -----------------------------------------------------------------------
  // Reused engines across all six shutdown paths (1D T1)
  // -----------------------------------------------------------------------

  describe("reused engine across all six shutdown paths", () => {
    it.each(["stop", "stopSync", "SIGINT", "SIGTERM", "beforeExit", "exit"] as const)(
      "never signals process, calls control gate, or unlinks files on %s when reused",
      async (pathName) => {
        let asyncGateCalls = 0;
        let syncGateCalls = 0;
        const {
          deps,
          fs,
          spawnedProcesses,
          signalHandlers,
          beforeExitHandlers,
          exitHandlers,
        } = createMockDeps({
          requestControlDisable: async () => {
            asyncGateCalls++;
            return APPROVED_CLEAR;
          },
          requestControlDisableSync: () => {
            syncGateCalls++;
            return APPROVED_CLEAR;
          },
        });
        setupConfigFile(fs);

        const stateDir = "/var/tmp/parapet-reused-state";
        const pidFile = join(stateDir, "engine.pid");
        const metaFile = join(stateDir, "engine.meta");
        const sockFile = join(stateDir, "control.sock");
        const tokenFile = join(stateDir, "control.token");
        const ownerFile = join(stateDir, "engine.owner.json");

        const { createHash } = require("node:crypto");
        const hash = createHash("sha256")
          .update(CONFIG_CONTENT, "utf-8")
          .digest("hex");

        fs.files.set(pidFile, "77777");
        fs.files.set(
          metaFile,
          JSON.stringify({
            port: 9800,
            configPath: CONFIG_PATH,
            configHash: hash,
            startedAt: new Date().toISOString(),
          }),
        );
        fs.files.set(sockFile, "");
        fs.files.set(tokenFile, "a".repeat(64));
        fs.files.set(
          ownerFile,
          JSON.stringify({
            schema: "parapet-engine-owner/v0",
            pid: 77777,
            port: 9800,
            config_hash: hash,
          }),
        );

        const killSpy = vi
          .spyOn(process, "kill")
          .mockImplementation(() => true);
        const exitSpy = vi
          .spyOn(process, "exit")
          .mockImplementation((() => undefined) as never);

        try {
          const mgr = new EngineManager({
            configPath: CONFIG_PATH,
            port: 9800,
            stateDir,
            deps,
          });
          await mgr.start();
          expect(spawnedProcesses).toHaveLength(0);
          expect(mgr.isInitialized).toBe(true);

          if (pathName === "stop") {
            await mgr.stop();
          } else if (pathName === "stopSync") {
            mgr.stopSync();
          } else if (pathName === "SIGINT") {
            const h = [...(signalHandlers.get("SIGINT") ?? [])][0];
            h();
            await new Promise((r) => setTimeout(r, 10));
          } else if (pathName === "SIGTERM") {
            const h = [...(signalHandlers.get("SIGTERM") ?? [])][0];
            h();
            await new Promise((r) => setTimeout(r, 10));
          } else if (pathName === "beforeExit") {
            const h = [...beforeExitHandlers][0];
            h();
            await new Promise((r) => setTimeout(r, 10));
          } else if (pathName === "exit") {
            const h = [...exitHandlers][0];
            h();
          }

          expect(asyncGateCalls).toBe(0);
          expect(syncGateCalls).toBe(0);
          expect(killSpy).not.toHaveBeenCalled();
          expect(fs.files.has(pidFile)).toBe(true);
          expect(fs.files.has(metaFile)).toBe(true);
          expect(fs.files.has(sockFile)).toBe(true);
          expect(fs.files.has(tokenFile)).toBe(true);
          expect(fs.files.has(ownerFile)).toBe(true);
        } finally {
          killSpy.mockRestore();
          exitSpy.mockRestore();
        }
      },
    );

    it("reuses engine from engine.owner.json when engine.pid is absent", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps();
      setupConfigFile(fs);

      const stateDir = "/var/tmp/parapet-owner-only";
      const ownerFile = join(stateDir, "engine.owner.json");
      const sockFile = join(stateDir, "control.sock");
      const tokenFile = join(stateDir, "control.token");
      fs.files.set(
        ownerFile,
        JSON.stringify({
          schema: "parapet-engine-owner/v0",
          pid: 88888,
          port: 9800,
          config_hash: "abc",
        }),
      );
      fs.files.set(sockFile, "");
      fs.files.set(tokenFile, "b".repeat(64));

      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9800,
        stateDir,
        deps,
      });
      await mgr.start();

      expect(spawnedProcesses).toHaveLength(0);
      expect(mgr.isInitialized).toBe(true);

      await mgr.stop();
      expect(fs.files.has(ownerFile)).toBe(true);
      expect(fs.files.has(sockFile)).toBe(true);
      expect(fs.files.has(tokenFile)).toBe(true);
    });
  });

  // -----------------------------------------------------------------------
  // Authenticated cleanup gate across all six shutdown paths (1D T1 / AC-1D-11)
  // -----------------------------------------------------------------------

  describe("authenticated cleanup gate across all six shutdown paths", () => {
    it.each(["stop", "SIGINT", "SIGTERM", "beforeExit"] as const)(
      "async path %s refuses to kill or unlink when gate refuses, then stops cleanly after restore approves",
      async (pathName) => {
        let gateApproved = false;
        const gateDirs: string[] = [];
        const {
          deps,
          fs,
          spawnedProcesses,
          signalHandlers,
          beforeExitHandlers,
        } = createMockDeps({
          requestControlDisable: async (dir) => {
            gateDirs.push(dir);
            return gateApproved ? APPROVED_CLEAR : REFUSED;
          },
        });
        setupConfigFile(fs);

        const stateDir = `/var/tmp/parapet-async-gate-${pathName}`;
        const pidFile = join(stateDir, "engine.pid");
        const metaFile = join(stateDir, "engine.meta");
        const sockFile = join(stateDir, "control.sock");
        const tokenFile = join(stateDir, "control.token");
        const ownerFile = join(stateDir, "engine.owner.json");
        const controlJsonFile = join(stateDir, "control.json");

        const exitSpy = vi
          .spyOn(process, "exit")
          .mockImplementation((() => undefined) as never);

        try {
          const mgr = new EngineManager({
            configPath: CONFIG_PATH,
            port: 9800,
            stateDir,
            engineArgs: ["--effect-root", "/var/tmp/parapet-effect"],
            deps,
          });
          await mgr.start();

          fs.files.set(sockFile, "");
          fs.files.set(tokenFile, "c".repeat(64));
          fs.files.set(
            ownerFile,
            JSON.stringify({ schema: "parapet-engine-owner/v0", pid: 12345 }),
          );
          fs.files.set(
            controlJsonFile,
            JSON.stringify({
              schema: "parapet-control-state/v0",
              effect_root_must_be_new: true,
            }),
          );

          const proc = spawnedProcesses[0];

          const invokePath = async (): Promise<void> => {
            if (pathName === "stop") {
              await mgr.stop();
            } else if (pathName === "SIGINT") {
              const h = [...(signalHandlers.get("SIGINT") ?? [])][0];
              h();
              await new Promise((r) => setTimeout(r, 15));
            } else if (pathName === "SIGTERM") {
              const h = [...(signalHandlers.get("SIGTERM") ?? [])][0];
              h();
              await new Promise((r) => setTimeout(r, 15));
            } else if (pathName === "beforeExit") {
              const h = [...beforeExitHandlers][0];
              h();
              await new Promise((r) => setTimeout(r, 15));
            }
          };

          // 1. Gate refuses (file cleanup not_run / fail / indeterminate).
          await invokePath();

          expect(gateDirs).toEqual([stateDir]);
          expect(proc.killed).toBe(false);
          expect(proc.killSignals).toEqual([]);
          expect(exitSpy).not.toHaveBeenCalled();
          expect(mgr.isInitialized).toBe(true);
          expect(fs.files.has(pidFile)).toBe(true);
          expect(fs.files.has(metaFile)).toBe(true);
          expect(fs.files.has(sockFile)).toBe(true);
          expect(fs.files.has(tokenFile)).toBe(true);
          expect(fs.files.has(ownerFile)).toBe(true);
          expect(fs.files.has(controlJsonFile)).toBe(true);

          // 2. Subsequent restore -> clear (gateApproved = true) allows clean stop.
          gateApproved = true;
          await invokePath();

          expect(gateDirs).toEqual([stateDir, stateDir]);
          expect(proc.killSignals).not.toContain("SIGKILL");
          expect(mgr.isInitialized).toBe(false);
          expect(fs.files.has(pidFile)).toBe(false);
          expect(fs.files.has(metaFile)).toBe(false);
          expect(fs.files.has(sockFile)).toBe(false);
          expect(fs.files.has(tokenFile)).toBe(false);
          expect(fs.files.has(ownerFile)).toBe(false);
          // Persistent state metadata is retained.
          expect(fs.files.has(controlJsonFile)).toBe(true);
        } finally {
          exitSpy.mockRestore();
        }
      },
    );

    it.each(["stopSync", "exit"] as const)(
      "sync path %s refuses to signal or unlink when gate refuses, and approval permits SIGTERM only without unlink",
      async (pathName) => {
        let syncApproved = false;
        const syncDirs: string[] = [];
        const { deps, fs, spawnedProcesses, exitHandlers } = createMockDeps({
          requestControlDisableSync: (dir) => {
            syncDirs.push(dir);
            return syncApproved ? APPROVED_CLEAR : REFUSED;
          },
        });
        setupConfigFile(fs);

        const stateDir = `/var/tmp/parapet-sync-gate-${pathName}`;
        const pidFile = join(stateDir, "engine.pid");
        const metaFile = join(stateDir, "engine.meta");
        const sockFile = join(stateDir, "control.sock");
        const tokenFile = join(stateDir, "control.token");
        const ownerFile = join(stateDir, "engine.owner.json");
        const controlJsonFile = join(stateDir, "control.json");

        const killSpy = vi
          .spyOn(process, "kill")
          .mockImplementation(() => true);

        try {
          const mgr = new EngineManager({
            configPath: CONFIG_PATH,
            port: 9800,
            stateDir,
            engineArgs: ["--effect-root", "/var/tmp/parapet-effect"],
            deps,
          });
          await mgr.start();

          fs.files.set(sockFile, "");
          fs.files.set(tokenFile, "d".repeat(64));
          fs.files.set(
            ownerFile,
            JSON.stringify({ schema: "parapet-engine-owner/v0", pid: 12345 }),
          );
          fs.files.set(
            controlJsonFile,
            JSON.stringify({
              schema: "parapet-control-state/v0",
              effect_root_must_be_new: true,
            }),
          );

          const proc = spawnedProcesses[0];
          const invokeSync = (): void => {
            if (pathName === "stopSync") {
              mgr.stopSync();
            } else {
              const h = [...exitHandlers][0];
              h();
            }
          };

          // 1. Refused synchronous gate.
          invokeSync();

          expect(syncDirs).toEqual([stateDir]);
          expect(killSpy).not.toHaveBeenCalled();
          expect(proc.killed).toBe(false);
          expect(mgr.isInitialized).toBe(true);
          expect(fs.files.has(pidFile)).toBe(true);
          expect(fs.files.has(metaFile)).toBe(true);
          expect(fs.files.has(sockFile)).toBe(true);
          expect(fs.files.has(tokenFile)).toBe(true);
          expect(fs.files.has(ownerFile)).toBe(true);

          // 2. Approved synchronous gate after restore -> clear: SIGTERM only,
          // and no unlink because exit cannot be confirmed synchronously.
          syncApproved = true;
          invokeSync();

          expect(syncDirs).toEqual([stateDir, stateDir]);
          expect(killSpy).toHaveBeenCalledOnce();
          expect(killSpy).toHaveBeenCalledWith(proc.pid, "SIGTERM");
          expect(proc.killSignals).toEqual([]);
          expect(mgr.isInitialized).toBe(true);
          expect(fs.files.has(pidFile)).toBe(true);
          expect(fs.files.has(metaFile)).toBe(true);
          expect(fs.files.has(sockFile)).toBe(true);
          expect(fs.files.has(tokenFile)).toBe(true);
          expect(fs.files.has(ownerFile)).toBe(true);
          expect(fs.files.has(controlJsonFile)).toBe(true);
        } finally {
          killSpy.mockRestore();
        }
      },
    );

    it("synchronous no-handshake hook (missing requestControlDisableSync) leaves engine and control files intact", async () => {
      const { deps, fs, spawnedProcesses, exitHandlers } = createMockDeps({
        requestControlDisableSync: undefined,
      });
      setupConfigFile(fs);

      const stateDir = "/var/tmp/parapet-no-sync-hook";
      const pidFile = join(stateDir, "engine.pid");
      const metaFile = join(stateDir, "engine.meta");
      const sockFile = join(stateDir, "control.sock");
      const tokenFile = join(stateDir, "control.token");

      const killSpy = vi.spyOn(process, "kill").mockImplementation(() => true);

      try {
        const mgr = new EngineManager({
          configPath: CONFIG_PATH,
          port: 9800,
          stateDir,
          deps,
        });
        await mgr.start();

        fs.files.set(sockFile, "");
        fs.files.set(tokenFile, "e".repeat(64));

        const exitFn = [...exitHandlers][0];
        exitFn();
        mgr.stopSync();

        expect(killSpy).not.toHaveBeenCalled();
        expect(spawnedProcesses[0].killed).toBe(false);
        expect(mgr.isInitialized).toBe(true);
        expect(fs.files.has(pidFile)).toBe(true);
        expect(fs.files.has(metaFile)).toBe(true);
        expect(fs.files.has(sockFile)).toBe(true);
        expect(fs.files.has(tokenFile)).toBe(true);
      } finally {
        killSpy.mockRestore();
      }
    });

    it("approved async gate preserves live state when the child does not exit", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        const { deps, fs, spawnedProcesses } = createMockDeps({
          requestControlDisable: async () => APPROVED_CLEAR,
        });
        setupConfigFile(fs);
        const stateDir = "/var/tmp/parapet-approved-timeout";
        const mgr = new EngineManager({
          configPath: CONFIG_PATH,
          stateDir,
          engineArgs: ["--effect-root", "/var/tmp/parapet-effect"],
          deps,
        });
        await mgr.start();
        fs.files.set(join(stateDir, "control.sock"), "");
        fs.files.set(join(stateDir, "control.token"), "a".repeat(64));
        fs.files.set(
          join(stateDir, "engine.owner.json"),
          JSON.stringify({ schema: "parapet-engine-owner/v0", pid: 12345 }),
        );

        const proc = spawnedProcesses[0];
        proc.stubborn = true;
        const stopping = mgr.stop();
        await vi.advanceTimersByTimeAsync(5_000);
        await stopping;

        expect(proc.killSignals).toEqual(["SIGTERM"]);
        expect(mgr.isInitialized).toBe(true);
        expect(fs.files.has(join(stateDir, "engine.pid"))).toBe(true);
        expect(fs.files.has(join(stateDir, "control.sock"))).toBe(true);
        expect(fs.files.has(join(stateDir, "control.token"))).toBe(true);
        expect(fs.files.has(join(stateDir, "engine.owner.json"))).toBe(true);

        proc.stubborn = false;
        await mgr.stop();
        expect(mgr.isInitialized).toBe(false);
      } finally {
        vi.useRealTimers();
      }
    });

    it("gated child already exited by signal is a confirmed exit: no grace wait, no SIGKILL", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps({
        requestControlDisable: async () => APPROVED_CLEAR,
      });
      setupConfigFile(fs);
      const stateDir = "/var/tmp/parapet-signal-exited";
      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        stateDir,
        engineArgs: ["--effect-root", "/var/tmp/parapet-effect"],
        deps,
      });
      await mgr.start();
      fs.files.set(join(stateDir, "control.sock"), "");
      fs.files.set(join(stateDir, "control.token"), "b".repeat(64));
      fs.files.set(
        join(stateDir, "engine.owner.json"),
        JSON.stringify({ schema: "parapet-engine-owner/v0", pid: 12345 }),
      );

      // The child was already reaped by a signal: exitCode stays null, and a
      // dead child never emits another "exit" event.
      const proc = spawnedProcesses[0];
      proc.stubborn = true;
      Object.defineProperty(proc, "signalCode", { value: "SIGTERM" });

      vi.useFakeTimers();
      try {
        let settled = false;
        const stopping = mgr.stop().then(() => {
          settled = true;
        });
        await vi.advanceTimersByTimeAsync(1);
        expect(settled).toBe(true);
        await stopping;
      } finally {
        vi.useRealTimers();
      }

      expect(proc.killSignals).toEqual([]);
      expect(proc.killSignals).not.toContain("SIGKILL");
      expect(mgr.isInitialized).toBe(false);
      for (const f of ["engine.pid", "engine.meta", "control.sock", "control.token", "engine.owner.json"]) {
        expect(fs.files.has(join(stateDir, f))).toBe(false);
      }
    });
  });

  // -----------------------------------------------------------------------
  // Startup rollback cannot bypass the cleanup gate (1D Rev 9 D4)
  // -----------------------------------------------------------------------

  describe("readiness rollback obeys effect cleanup gate Rev9", () => {
    const STATE_DIR = "/var/tmp/parapet-rev9-rollback";
    const PID = join(STATE_DIR, "engine.pid");
    const META = join(STATE_DIR, "engine.meta");
    const SOCK = join(STATE_DIR, "control.sock");
    const TOKEN = join(STATE_DIR, "control.token");
    const OWNER = join(STATE_DIR, "engine.owner.json");
    const CONTROL_JSON = join(STATE_DIR, "control.json");
    const GATED_FILES = [PID, META, SOCK, TOKEN, OWNER, CONTROL_JSON];

    /** Readiness timeout, child exit after readiness, or both. */
    type Failure = "readiness" | "exited" | "exited_unready";

    interface Harness {
      mgr: EngineManager;
      fs: MockFs;
      proc: () => MockProcess;
      events: string[];
      unlinks: string[];
      asyncGateCalls: () => number;
      killSpy: MockInstance<typeof process.kill>;
    }

    let killSpy: MockInstance<typeof process.kill>;

    beforeEach(() => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      killSpy = vi.spyOn(process, "kill").mockImplementation(() => true);
    });
    afterEach(() => {
      killSpy.mockRestore();
      vi.useRealTimers();
    });

    function harness(opts: {
      gate?: SidecarDeps["requestControlDisable"];
      failure: Failure;
      effectRoot?: boolean;
      artifacts?: boolean;
      ownerPid?: number | "unparseable";
      stubborn?: boolean;
    }): Harness {
      const events: string[] = [];
      const unlinks: string[] = [];
      let gateCalls = 0;
      let spawned: MockProcess | null = null;
      const { deps, fs } = createMockDeps({
        requestControlDisable:
          opts.gate === undefined
            ? undefined
            : async (dir) => {
                gateCalls++;
                events.push("gate");
                return opts.gate!(dir);
              },
        fetch: async () => {
          if (opts.failure !== "exited") throw new Error("not ready");
          return { ok: true, status: 200 } as Response;
        },
        spawnProcess: () => {
          const proc = new MockProcess();
          proc.stubborn = opts.stubborn ?? false;
          if (opts.failure !== "readiness") {
            Object.defineProperty(proc, "exitCode", { value: 1 });
          }
          proc.on("exit", () => events.push("exit"));
          spawned = proc;
          // The engine writes its control artifacts once it is running.
          if (opts.artifacts ?? true) {
            fs.files.set(SOCK, "");
            fs.files.set(TOKEN, "7".repeat(64));
            fs.files.set(
              OWNER,
              opts.ownerPid === "unparseable"
                ? "{not json"
                : JSON.stringify({
                    schema: "parapet-engine-owner/v0",
                    pid: opts.ownerPid ?? proc.pid,
                  }),
            );
            fs.files.set(CONTROL_JSON, "{}");
          }
          return proc as unknown as ChildProcess;
        },
      });
      const baseUnlink = deps.unlink;
      deps.unlink = (p) => {
        unlinks.push(p);
        events.push(`unlink:${p}`);
        baseUnlink(p);
      };
      setupConfigFile(fs);
      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9800,
        stateDir: STATE_DIR,
        engineArgs:
          opts.effectRoot ?? true
            ? ["--effect-root", "/var/tmp/parapet-rev9-effect"]
            : [],
        deps,
      });
      return {
        mgr,
        fs,
        proc: () => spawned!,
        events,
        unlinks,
        asyncGateCalls: () => gateCalls,
        killSpy,
      };
    }

    async function startAndCollect(mgr: EngineManager): Promise<unknown> {
      const outcome = mgr.start().then(
        () => null,
        (err: unknown) => err,
      );
      await vi.advanceTimersByTimeAsync(30_000);
      return outcome;
    }

    function expectZeroKillZeroUnlink(h: Harness): void {
      expect(h.proc().killSignals).toEqual([]);
      expect(h.proc().killed).toBe(false);
      expect(h.killSpy).not.toHaveBeenCalled();
      expect(h.unlinks).toEqual([]);
    }

    function expectFilesPreserved(h: Harness, files = GATED_FILES): void {
      for (const f of files) {
        expect(h.fs.files.has(f)).toBe(true);
      }
    }

    it.each([
      ["clear", APPROVED_CLEAR],
      ["not_applicable", APPROVED_NOT_APPLICABLE],
    ] as const)(
      "async approved %s permits SIGTERM only and removes controls only after confirmed exit",
      async (_label, gate) => {
        const h = harness({ gate: async () => gate, failure: "readiness" });
        const err = await startAndCollect(h.mgr);

        expect(err).toBeInstanceOf(Error);
        expect(err).not.toBeInstanceOf(EngineRecoveryError);
        expect((err as Error).message).toMatch(/did not become ready/);
        expect(h.asyncGateCalls()).toBe(1);
        expect(h.proc().killSignals).toEqual(["SIGTERM"]);
        expect(h.killSpy).not.toHaveBeenCalled();

        const exitAt = h.events.indexOf("exit");
        const firstUnlink = h.events.findIndex((e) => e.startsWith("unlink:"));
        expect(h.events.indexOf("gate")).toBeLessThan(exitAt);
        expect(exitAt).toBeGreaterThanOrEqual(0);
        expect(firstUnlink).toBeGreaterThan(exitAt);

        for (const f of [PID, META, SOCK, TOKEN, OWNER]) {
          expect(h.fs.files.has(f)).toBe(false);
        }
        expect(h.fs.files.has(CONTROL_JSON)).toBe(true);
        expect(h.mgr.isInitialized).toBe(false);
      },
    );

    it("async approved without confirmed exit sends no SIGKILL and unlinks nothing", async () => {
      const h = harness({
        gate: async () => APPROVED_CLEAR,
        failure: "readiness",
        stubborn: true,
      });
      const err = await startAndCollect(h.mgr);

      expect(err).toBeInstanceOf(EngineRecoveryError);
      expect((err as EngineRecoveryError).reason).toBe("exit_unconfirmed");
      expect(h.proc().killSignals).toEqual(["SIGTERM"]);
      expect(h.killSpy).not.toHaveBeenCalled();
      expect(h.unlinks).toEqual([]);
      expectFilesPreserved(h);
    });

    it.each([
      ["refused", async () => ({ kind: "refused" }) as ControlGateResult, "refused"],
      ["unavailable", async () => ({ kind: "unavailable" }) as ControlGateResult, "unavailable"],
      ["timeout", async () => ({ kind: "timeout" }) as ControlGateResult, "timeout"],
      ["malformed", async () => ({ kind: "malformed" }) as ControlGateResult, "malformed"],
      ["bare boolean true", async () => true as unknown as ControlGateResult, "malformed"],
      [
        "approved without fileCleanup",
        async () => ({ kind: "approved" }) as unknown as ControlGateResult,
        "malformed",
      ],
      [
        "approved with fail cleanup",
        async () => ({ kind: "approved", fileCleanup: "fail" }) as unknown as ControlGateResult,
        "malformed",
      ],
      [
        "approved with extra field",
        async () =>
          ({ kind: "approved", fileCleanup: "clear", force: true }) as unknown as ControlGateResult,
        "malformed",
      ],
      [
        "throwing client",
        async (): Promise<ControlGateResult> => {
          throw new Error("socket exploded");
        },
        "unavailable",
      ],
    ] as const)(
      "gate %s performs zero kill and zero unlink and returns a recovery error",
      async (_label, gate, reason) => {
        const h = harness({ gate, failure: "readiness" });
        const err = await startAndCollect(h.mgr);

        expect(err).toBeInstanceOf(EngineRecoveryError);
        expect((err as EngineRecoveryError).reason).toBe(reason);
        expect((err as EngineRecoveryError).stateDir).toBe(STATE_DIR);
        expect((err as EngineRecoveryError).startupError?.message).toMatch(
          /did not become ready/,
        );
        expect(h.asyncGateCalls()).toBe(1);
        expectZeroKillZeroUnlink(h);
        expectFilesPreserved(h);
        expect(h.mgr.isInitialized).toBe(false);
      },
    );

    it("missing async control client is unavailable: zero kill, zero unlink", async () => {
      const h = harness({ gate: undefined, failure: "readiness" });
      const err = await startAndCollect(h.mgr);

      expect(err).toBeInstanceOf(EngineRecoveryError);
      expect((err as EngineRecoveryError).reason).toBe("unavailable");
      expectZeroKillZeroUnlink(h);
      expectFilesPreserved(h);
    });

    it("effect root alone gates rollback even before any control artifact exists", async () => {
      const h = harness({
        gate: async () => REFUSED,
        failure: "readiness",
        artifacts: false,
      });
      const err = await startAndCollect(h.mgr);

      expect(err).toBeInstanceOf(EngineRecoveryError);
      expect((err as EngineRecoveryError).reason).toBe("refused");
      expectZeroKillZeroUnlink(h);
      expectFilesPreserved(h, [PID, META]);
    });

    it("control artifacts alone gate rollback without an effect root", async () => {
      const h = harness({
        gate: async () => REFUSED,
        failure: "readiness",
        effectRoot: false,
      });
      const err = await startAndCollect(h.mgr);

      expect(err).toBeInstanceOf(EngineRecoveryError);
      expect((err as EngineRecoveryError).reason).toBe("refused");
      expectZeroKillZeroUnlink(h);
      expectFilesPreserved(h);
    });

    it.each([
      ["owner pid mismatch", 99999],
      ["unparseable owner record", "unparseable"],
    ] as const)(
      "ownership unknown (%s) skips the gate and performs zero kill and zero unlink",
      async (_label, ownerPid) => {
        const h = harness({
          gate: async () => APPROVED_CLEAR,
          failure: "readiness",
          ownerPid,
        });
        const err = await startAndCollect(h.mgr);

        expect(err).toBeInstanceOf(EngineRecoveryError);
        expect((err as EngineRecoveryError).reason).toBe("ownership_unknown");
        expect(h.asyncGateCalls()).toBe(0);
        expectZeroKillZeroUnlink(h);
        expectFilesPreserved(h);
      },
    );

    it.each(["exited", "exited_unready"] as const)(
      "child exited before handshake (%s) skips the gate and preserves evidence/control state",
      async (failure) => {
        const h = harness({ gate: async () => APPROVED_CLEAR, failure });
        const err = await startAndCollect(h.mgr);

        expect(err).toBeInstanceOf(EngineRecoveryError);
        expect((err as EngineRecoveryError).reason).toBe("exited_before_handshake");
        expect(h.asyncGateCalls()).toBe(0);
        expectZeroKillZeroUnlink(h);
        expectFilesPreserved(h);
      },
    );

    it("sync approved permits at most SIGTERM and never unlinks", async () => {
      let syncCalls = 0;
      const { deps, fs, spawnedProcesses } = createMockDeps({
        requestControlDisableSync: () => {
          syncCalls++;
          return APPROVED_NOT_APPLICABLE;
        },
      });
      const unlinks: string[] = [];
      const baseUnlink = deps.unlink;
      deps.unlink = (p) => {
        unlinks.push(p);
        baseUnlink(p);
      };
      setupConfigFile(fs);
      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9800,
        stateDir: STATE_DIR,
        engineArgs: ["--effect-root", "/var/tmp/parapet-rev9-effect"],
        deps,
      });
      await mgr.start();
      fs.files.set(SOCK, "");
      fs.files.set(TOKEN, "8".repeat(64));
      fs.files.set(OWNER, JSON.stringify({ pid: spawnedProcesses[0].pid }));

      mgr.stopSync();

      expect(syncCalls).toBe(1);
      expect(killSpy).toHaveBeenCalledOnce();
      expect(killSpy).toHaveBeenCalledWith(spawnedProcesses[0].pid, "SIGTERM");
      expect(killSpy).not.toHaveBeenCalledWith(expect.anything(), "SIGKILL");
      expect(spawnedProcesses[0].killSignals).toEqual([]);
      expect(unlinks).toEqual([]);
      for (const f of [PID, META, SOCK, TOKEN, OWNER]) {
        expect(fs.files.has(f)).toBe(true);
      }
      expect(mgr.isInitialized).toBe(true);
    });

    it("legacy ungated rollback applies only without effect root and without control artifacts", async () => {
      const h = harness({
        gate: async () => REFUSED,
        failure: "readiness",
        effectRoot: false,
        artifacts: false,
      });
      const err = await startAndCollect(h.mgr);

      expect(err).not.toBeInstanceOf(EngineRecoveryError);
      expect((err as Error).message).toMatch(/did not become ready/);
      expect(h.asyncGateCalls()).toBe(0);
      expect(h.proc().killSignals).toContain("SIGTERM");
      expect(h.fs.files.has(PID)).toBe(false);
      expect(h.fs.files.has(META)).toBe(false);
    });

    it("stale gated control state is never unlinked ungated on restart", async () => {
      const { deps, fs, spawnedProcesses } = createMockDeps({
        fetch: async () => {
          throw new Error("connection refused");
        },
      });
      const unlinks: string[] = [];
      deps.unlink = (p) => {
        unlinks.push(p);
        fs.files.delete(p);
      };
      setupConfigFile(fs);
      fs.files.set(PID, "4242");
      fs.files.set(SOCK, "");
      fs.files.set(TOKEN, "9".repeat(64));
      fs.files.set(OWNER, JSON.stringify({ pid: 4242 }));

      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9800,
        stateDir: STATE_DIR,
        deps,
      });
      const err = await startAndCollect(mgr);

      expect(err).toBeInstanceOf(EngineRecoveryError);
      expect((err as EngineRecoveryError).reason).toBe("stale_control_state");
      expect(spawnedProcesses).toHaveLength(0);
      expect(unlinks).toEqual([]);
      expect(killSpy).not.toHaveBeenCalled();
      for (const f of [PID, SOCK, TOKEN, OWNER]) {
        expect(fs.files.has(f)).toBe(true);
      }
    });

    it("stale PID alone is cleaned and startup proceeds even with a newly requested effect root", async () => {
      let fetchCalls = 0;
      const { deps, fs, spawnedProcesses } = createMockDeps({
        // The stale-PID heartbeat finds the port free; the new child is ready.
        fetch: async () => {
          fetchCalls++;
          if (fetchCalls === 1) throw new Error("connection refused");
          return { ok: true, status: 200 } as Response;
        },
      });
      setupConfigFile(fs);
      fs.files.set(PID, "4242");

      const mgr = new EngineManager({
        configPath: CONFIG_PATH,
        port: 9800,
        stateDir: STATE_DIR,
        engineArgs: ["--effect-root", "/var/tmp/parapet-rev9-effect"],
        deps,
      });
      const err = await startAndCollect(mgr);

      expect(err).toBeNull();
      expect(spawnedProcesses).toHaveLength(1);
      expect(fs.files.get(PID)).toBe(String(spawnedProcesses[0].pid));
      expect(mgr.isInitialized).toBe(true);
      expect(killSpy).not.toHaveBeenCalled();
    });

    it.each([
      ["control.sock", SOCK, ""],
      ["control.token", TOKEN, "9".repeat(64)],
      ["engine.owner.json", OWNER, JSON.stringify({ pid: 4242 })],
    ] as const)(
      "stale PID plus %s with a newly requested effect root remains fail closed",
      async (_label, artifact, content) => {
        const { deps, fs, spawnedProcesses } = createMockDeps({
          fetch: async () => {
            throw new Error("connection refused");
          },
        });
        const unlinks: string[] = [];
        deps.unlink = (p) => {
          unlinks.push(p);
          fs.files.delete(p);
        };
        setupConfigFile(fs);
        fs.files.set(PID, "4242");
        fs.files.set(artifact, content);

        const mgr = new EngineManager({
          configPath: CONFIG_PATH,
          port: 9800,
          stateDir: STATE_DIR,
          engineArgs: ["--effect-root", "/var/tmp/parapet-rev9-effect"],
          deps,
        });
        const err = await startAndCollect(mgr);

        expect(err).toBeInstanceOf(EngineRecoveryError);
        expect((err as EngineRecoveryError).reason).toBe("stale_control_state");
        expect(spawnedProcesses).toHaveLength(0);
        expect(unlinks).toEqual([]);
        expect(killSpy).not.toHaveBeenCalled();
        expect(fs.files.has(PID)).toBe(true);
        expect(fs.files.has(artifact)).toBe(true);
      },
    );
  });

  // -----------------------------------------------------------------------
  // Default Unix domain socket control-gate helpers (parapet-control/v0)
  // -----------------------------------------------------------------------

  describe("defaultRequestControlDisable and defaultRequestControlDisableSync", () => {
    it("returns unavailable when control.sock or control.token is missing, empty, or not listening", async () => {
      const tmpDir = mkdtempSync("/tmp/p1d_ts_ctrl_missing_");
      chmodSync(tmpDir, 0o700);
      try {
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "unavailable" });
        expect(defaultRequestControlDisableSync(tmpDir)).toEqual({ kind: "unavailable" });

        writeFileSync(join(tmpDir, "control.token"), "   \n", { mode: 0o600 });
        writeFileSync(join(tmpDir, "control.sock"), "", { mode: 0o600 });
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "unavailable" });
        expect(defaultRequestControlDisableSync(tmpDir)).toEqual({ kind: "unavailable" });

        // Token present but control.sock is a plain file, not a listener.
        writeFileSync(join(tmpDir, "control.token"), "a".repeat(64), { mode: 0o600 });
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "unavailable" });
        expect(defaultRequestControlDisableSync(tmpDir)).toEqual({ kind: "unavailable" });
      } finally {
        rmSync(tmpDir, { recursive: true, force: true });
      }
    });

    it("authenticates over control.sock and enforces stop_approved + file_cleanup", async () => {
      const tmpDir = mkdtempSync("/tmp/p1d_ts_ctrl_");
      chmodSync(tmpDir, 0o700);
      const sockPath = join(tmpDir, "control.sock");
      const tokenPath = join(tmpDir, "control.token");
      const expectedToken = "f".repeat(64);
      writeFileSync(tokenPath, expectedToken, { mode: 0o600 });

      let nextResponse: Record<string, unknown> | string = {
        schema: "parapet-control/v0",
        ok: false,
        stop_approved: false,
        file_cleanup: "not_run",
        evidence_retained: true,
      };
      const receivedRequests: Array<Record<string, unknown>> = [];

      const server = createServer((conn) => {
        let buf = "";
        conn.on("data", (chunk) => {
          buf += chunk.toString("utf-8");
          if (buf.includes("\n")) {
            const line = buf.split("\n", 1)[0].trim();
            receivedRequests.push(JSON.parse(line));
            conn.end(
              (typeof nextResponse === "string"
                ? nextResponse
                : JSON.stringify(nextResponse)) + "\n",
            );
          }
        });
      });

      await new Promise<void>((resolve) => server.listen(sockPath, resolve));
      chmodSync(sockPath, 0o600);

      try {
        // 1. Refused (not_run).
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "refused" });
        expect(receivedRequests).toHaveLength(1);
        expect(receivedRequests[0]).toEqual({
          schema: "parapet-control/v0",
          token: expectedToken,
          action: "disable",
        });

        // 2. Approved (clear).
        nextResponse = {
          schema: "parapet-control/v0",
          ok: true,
          stop_approved: true,
          file_cleanup: "clear",
          evidence_retained: true,
        };
        expect(await defaultRequestControlDisable(tmpDir)).toEqual(APPROVED_CLEAR);
        expect(receivedRequests).toHaveLength(2);

        // 3. Approved (not_applicable).
        nextResponse = { ...nextResponse, file_cleanup: "not_applicable" };
        expect(await defaultRequestControlDisable(tmpDir)).toEqual(
          APPROVED_NOT_APPLICABLE,
        );

        // 4. Stop approved but file cleanup failed -> refused.
        nextResponse = { ...nextResponse, file_cleanup: "fail" };
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "refused" });

        // 5. Non-JSON, wrong schema, and untyped fields -> malformed.
        nextResponse = "not json";
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "malformed" });
        nextResponse = {
          schema: "parapet-control/v1",
          ok: true,
          stop_approved: true,
          file_cleanup: "clear",
        };
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "malformed" });
        nextResponse = {
          schema: "parapet-control/v0",
          ok: "true",
          stop_approved: true,
          file_cleanup: "clear",
        };
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "malformed" });

        // 6. Oversized response line -> malformed.
        nextResponse = "x".repeat(5000);
        expect(await defaultRequestControlDisable(tmpDir)).toEqual({ kind: "malformed" });
      } finally {
        await new Promise<void>((resolve) => server.close(() => resolve()));
        rmSync(tmpDir, { recursive: true, force: true });
      }
    });

    it("performs the synchronous authenticated handshake in a separate server process", async () => {
      const tmpDir = mkdtempSync("/tmp/p1d_ts_ctrl_sync_");
      chmodSync(tmpDir, 0o700);
      const sockPath = join(tmpDir, "control.sock");
      const tokenPath = join(tmpDir, "control.token");
      const requestPath = join(tmpDir, "request.json");
      const expectedToken = "9".repeat(64);
      writeFileSync(tokenPath, expectedToken, { mode: 0o600 });

      const serverScript = [
        'const fs = require("node:fs");',
        'const net = require("node:net");',
        "const [sockPath, requestPath] = process.argv.slice(1);",
        "const server = net.createServer((conn) => {",
        '  let buf = "";',
        '  conn.on("data", (chunk) => {',
        '    buf += chunk.toString("utf-8");',
        '    if (!buf.includes("\\n")) return;',
        '    fs.writeFileSync(requestPath, buf.split("\\n", 1)[0]);',
        '    conn.end(JSON.stringify({ schema: "parapet-control/v0", ok: true, stop_approved: true, file_cleanup: "clear" }) + "\\n");',
        "    server.close();",
        "  });",
        "});",
        'server.listen(sockPath, () => process.stdout.write("READY\\n"));',
      ].join("\n");
      const server = spawnChild(
        process.execPath,
        ["-e", serverScript, sockPath, requestPath],
        { stdio: ["ignore", "pipe", "inherit"] },
      );
      const exited = new Promise<void>((resolve, reject) => {
        server.once("error", reject);
        server.once("exit", (code) => {
          if (code === 0) resolve();
          else reject(new Error(`control test server exited ${code}`));
        });
      });

      try {
        await new Promise<void>((resolve, reject) => {
          server.stdout!.once("data", (chunk) => {
            if (chunk.toString("utf-8").includes("READY")) resolve();
            else reject(new Error("control test server did not become ready"));
          });
          server.once("error", reject);
        });
        chmodSync(sockPath, 0o600);

        expect(defaultRequestControlDisableSync(tmpDir)).toEqual(APPROVED_CLEAR);
        await exited;
        expect(JSON.parse(readFileSync(requestPath, "utf-8"))).toEqual({
          schema: "parapet-control/v0",
          token: expectedToken,
          action: "disable",
        });
      } finally {
        if (server.exitCode === null) server.kill();
        rmSync(tmpDir, { recursive: true, force: true });
      }
    });
  });
});
