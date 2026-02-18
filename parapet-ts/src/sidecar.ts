// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import { createHash } from "node:crypto";
import { spawn, type ChildProcess } from "node:child_process";
import {
  readFileSync,
  writeFileSync,
  unlinkSync,
  existsSync,
  mkdirSync,
  realpathSync,
  openSync,
  closeSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// ---------------------------------------------------------------------------
// Dependency injection
// ---------------------------------------------------------------------------

/** Abstraction over OS/FS/process operations for testability. */
export interface SidecarDeps {
  readFile(path: string): string;
  writeFile(path: string, data: string): void;
  unlink(path: string): void;
  exists(path: string): boolean;
  mkdirp(path: string): void;
  realpath(path: string): string;
  openLogFile(path: string): number;
  closeFile(fd: number): void;
  spawnProcess(
    cmd: string,
    args: string[],
    logFd: number,
  ): ChildProcess;
  fetch(url: string, opts: { signal?: AbortSignal }): Promise<Response>;
  homedir(): string;
  onSignal(signal: string, handler: () => void): void;
  removeSignalHandler(signal: string, handler: () => void): void;
  onBeforeExit(handler: () => void): void;
  removeBeforeExitHandler(handler: () => void): void;
  onExit(handler: () => void): void;
  removeExitHandler(handler: () => void): void;
}

/** Default deps wrapping Node built-ins. */
export function defaultDeps(): SidecarDeps {
  return {
    readFile: (p) => readFileSync(p, "utf-8"),
    writeFile: (p, d) => writeFileSync(p, d, "utf-8"),
    unlink: (p) => unlinkSync(p),
    exists: (p) => existsSync(p),
    mkdirp: (p) => mkdirSync(p, { recursive: true }),
    realpath: (p) => realpathSync(p),
    openLogFile: (p) => openSync(p, "a"),
    closeFile: (fd) => closeSync(fd),
    spawnProcess: (cmd, args, logFd) =>
      spawn(cmd, args, { stdio: ["ignore", logFd, logFd], detached: false }),
    fetch: (url, opts) => globalThis.fetch(url, opts),
    homedir: () => homedir(),
    onSignal: (sig, handler) => process.on(sig, handler),
    removeSignalHandler: (sig, handler) => process.removeListener(sig, handler),
    onBeforeExit: (handler) => process.on("beforeExit", handler),
    removeBeforeExitHandler: (handler) =>
      process.removeListener("beforeExit", handler),
    onExit: (handler) => process.on("exit", handler),
    removeExitHandler: (handler) => process.removeListener("exit", handler),
  };
}

// ---------------------------------------------------------------------------
// Engine metadata
// ---------------------------------------------------------------------------

interface EngineMeta {
  port: number;
  configPath: string;
  configHash: string;
  startedAt: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_PORT = 9800;
const HEARTBEAT_INTERVAL_MS = 10_000;
const READINESS_TIMEOUT_MS = 10_000;
const READINESS_INITIAL_DELAY_MS = 50;
const SHUTDOWN_GRACE_MS = 5_000;

// ---------------------------------------------------------------------------
// EngineManager
// ---------------------------------------------------------------------------

export interface EngineManagerOptions {
  configPath: string;
  port?: number;
  engineBinary?: string;
  deps?: SidecarDeps;
}

export class EngineManager {
  private readonly deps: SidecarDeps;
  private readonly parapetDir: string;
  private readonly pidPath: string;
  private readonly metaPath: string;
  private readonly logPath: string;

  private process: ChildProcess | null = null;
  private logFd: number | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private initialized = false;
  /** True when we spawned the process; false when reusing from PID file. */
  private ownsProcess = false;
  /** Canonical config path resolved via realpath, set during start(). */
  private _canonicalConfigPath: string | null = null;
  /** SHA-256 of config contents, set during start(). */
  private _configHash: string | null = null;

  // Store handler refs so repeated init() doesn't leak.
  private sigintHandler: (() => void) | null = null;
  private sigtermHandler: (() => void) | null = null;
  private beforeExitHandler: (() => void) | null = null;
  private exitHandler: (() => void) | null = null;
  private cleaning = false;

  constructor(private readonly opts: EngineManagerOptions) {
    this.deps = opts.deps ?? defaultDeps();
    this.parapetDir = join(this.deps.homedir(), ".parapet");
    this.pidPath = join(this.parapetDir, "engine.pid");
    this.metaPath = join(this.parapetDir, "engine.meta");
    this.logPath = join(this.parapetDir, "engine.log");
  }

  get configPath(): string {
    return this.opts.configPath;
  }

  /** Canonical config path (after realpath). Only available after start(). */
  get canonicalConfigPath(): string | null {
    return this._canonicalConfigPath;
  }

  /** SHA-256 of config file contents. Only available after start(). */
  get configHash(): string | null {
    return this._configHash;
  }

  get port(): number {
    return this.opts.port ?? DEFAULT_PORT;
  }

  get isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Check if a given configPath resolves to the same canonical identity
   * as this manager's config. Uses realpath for comparison.
   */
  matchesCanonicalConfig(configPath: string): boolean {
    if (!this._canonicalConfigPath) return false;
    try {
      const resolved = this.deps.realpath(configPath);
      return resolved === this._canonicalConfigPath;
    } catch {
      return false;
    }
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * Start the engine sidecar and wait for readiness.
   *
   * Idempotent: same config → no-op. Changed config → restart.
   */
  async start(): Promise<void> {
    const canonicalPath = this.deps.realpath(this.opts.configPath);
    const configContents = this.deps.readFile(canonicalPath);
    const configHash = sha256(configContents);

    this._canonicalConfigPath = canonicalPath;
    this._configHash = configHash;

    // Already running in this process — check for config identity.
    if (this.initialized && this.process !== null) {
      const meta = this.readMeta();
      if (
        meta &&
        meta.configPath === canonicalPath &&
        meta.configHash === configHash &&
        meta.port === this.port
      ) {
        return; // Same config, reuse.
      }
      // Config changed — stop old engine, start fresh.
      await this.stop();
    }

    // Check for existing PID file from a previous run.
    if (this.deps.exists(this.pidPath)) {
      const meta = this.readMeta();
      const pidText = this.tryReadPid();

      if (
        pidText !== null &&
        meta &&
        meta.configPath === canonicalPath &&
        meta.configHash === configHash &&
        meta.port === this.port
      ) {
        // Config matches — check if still alive via heartbeat.
        if (await this.heartbeatOnce()) {
          this.initialized = true;
          this.ownsProcess = false; // We did not spawn it.
          this.startHeartbeat();
          this.registerShutdownHandlers();
          return; // Reuse existing engine.
        }
      } else if (await this.heartbeatOnce()) {
        // Mismatch but port is occupied — an incompatible engine is
        // running. Fail fast rather than spawning into a port conflict.
        throw new Error(
          `Port ${this.port} is already in use by an engine with ` +
            `different config. Stop the existing engine first, or ` +
            `use a different port.`,
        );
      }

      // Stale or mismatched and port is free — clean up files.
      this.cleanupFiles();
    }

    // Ensure parapet dir exists.
    this.deps.mkdirp(this.parapetDir);

    // Resolve engine binary.
    const binary = this.resolveEngineBinary();

    // Open log file.
    this.logFd = this.deps.openLogFile(this.logPath);

    // Spawn.
    this.process = this.deps.spawnProcess(
      binary,
      ["--config", canonicalPath, "--port", String(this.port)],
      this.logFd,
    );
    this.ownsProcess = true;

    // Write PID + meta.
    this.deps.writeFile(this.pidPath, String(this.process.pid));
    this.deps.writeFile(
      this.metaPath,
      JSON.stringify({
        port: this.port,
        configPath: canonicalPath,
        configHash,
        startedAt: new Date().toISOString(),
      } satisfies EngineMeta),
    );

    // Wait for readiness. If it fails, clean up the process/files we just
    // created so we don't leak a child process or stale metadata.
    try {
      await this.waitForReady();
    } catch (err) {
      await this.stop();
      throw err;
    }

    // Verify our child process is still alive — if it exited immediately,
    // the readiness heartbeat may have hit a stale listener on the port.
    const exitCode = this.process.exitCode;
    if (exitCode !== null) {
      await this.stop();
      throw new Error(
        `Engine process exited immediately with code ${exitCode}. ` +
          `Check ${this.logPath} for details.`,
      );
    }

    this.initialized = true;
    this.startHeartbeat();
    this.registerShutdownHandlers();
  }

  /** Stop the engine and clean up all resources. */
  async stop(): Promise<void> {
    if (this.cleaning) return;
    this.cleaning = true;

    try {
      this.stopHeartbeat();
      this.removeShutdownHandlers();

      if (this.process !== null) {
        await this.terminateProcess(this.process);
        this.process = null;
      }

      if (this.logFd !== null) {
        try {
          this.deps.closeFile(this.logFd);
        } catch {
          // Best effort.
        }
        this.logFd = null;
      }

      // Only remove PID/meta files if we spawned the process.
      // Reused engines are not ours to clean up.
      if (this.ownsProcess) {
        this.cleanupFiles();
      }

      this.initialized = false;
      this.ownsProcess = false;
    } finally {
      this.cleaning = false;
    }
  }

  /** Synchronous last-resort kill (for process.on("exit")). */
  stopSync(): void {
    this.stopHeartbeat();
    if (this.process?.pid) {
      try {
        process.kill(this.process.pid, "SIGKILL");
      } catch {
        // Best effort.
      }
    }
    if (this.ownsProcess) {
      this.cleanupFiles();
    }
    this.initialized = false;
    this.ownsProcess = false;
  }

  // -----------------------------------------------------------------------
  // Internals
  // -----------------------------------------------------------------------

  private resolveEngineBinary(): string {
    if (this.opts.engineBinary) return this.opts.engineBinary;
    const fromEnv = process.env["PARAPET_ENGINE_PATH"];
    if (fromEnv) return fromEnv;
    return "parapet-engine";
  }

  private async waitForReady(): Promise<void> {
    const deadline = Date.now() + READINESS_TIMEOUT_MS;
    let delay = READINESS_INITIAL_DELAY_MS;

    while (Date.now() < deadline) {
      if (await this.heartbeatOnce()) return;
      const remaining = deadline - Date.now();
      if (remaining <= 0) break;
      await sleep(Math.min(delay, remaining));
      delay *= 2;
    }

    throw new Error(
      `Engine did not become ready within ${READINESS_TIMEOUT_MS}ms. ` +
        `Check ${this.logPath} for details.`,
    );
  }

  private async heartbeatOnce(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 2000);
      try {
        const res = await this.deps.fetch(
          `http://127.0.0.1:${this.port}/v1/heartbeat`,
          { signal: controller.signal },
        );
        return res.ok;
      } finally {
        clearTimeout(timeout);
      }
    } catch {
      return false;
    }
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      this.heartbeatOnce().catch(() => {
        // Log only, don't kill the SDK.
      });
    }, HEARTBEAT_INTERVAL_MS);
    // Allow process to exit even if timer is running.
    if (this.heartbeatTimer && typeof this.heartbeatTimer === "object" && "unref" in this.heartbeatTimer) {
      this.heartbeatTimer.unref();
    }
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private registerShutdownHandlers(): void {
    // Remove any previous handlers first to prevent accumulation.
    this.removeShutdownHandlers();

    this.sigintHandler = () => {
      void this.stop().then(() => {
        process.exit(130);
      });
    };
    this.sigtermHandler = () => {
      void this.stop().then(() => {
        process.exit(143);
      });
    };
    this.beforeExitHandler = () => {
      void this.stop();
    };
    this.exitHandler = () => {
      this.stopSync();
    };

    this.deps.onSignal("SIGINT", this.sigintHandler);
    this.deps.onSignal("SIGTERM", this.sigtermHandler);
    this.deps.onBeforeExit(this.beforeExitHandler);
    this.deps.onExit(this.exitHandler);
  }

  private removeShutdownHandlers(): void {
    if (this.sigintHandler) {
      this.deps.removeSignalHandler("SIGINT", this.sigintHandler);
      this.sigintHandler = null;
    }
    if (this.sigtermHandler) {
      this.deps.removeSignalHandler("SIGTERM", this.sigtermHandler);
      this.sigtermHandler = null;
    }
    if (this.beforeExitHandler) {
      this.deps.removeBeforeExitHandler(this.beforeExitHandler);
      this.beforeExitHandler = null;
    }
    if (this.exitHandler) {
      this.deps.removeExitHandler(this.exitHandler);
      this.exitHandler = null;
    }
  }

  private async terminateProcess(proc: ChildProcess): Promise<void> {
    return new Promise<void>((resolve) => {
      const timeout = setTimeout(() => {
        try {
          proc.kill("SIGKILL");
        } catch {
          // Already dead.
        }
        resolve();
      }, SHUTDOWN_GRACE_MS);

      proc.once("exit", () => {
        clearTimeout(timeout);
        resolve();
      });

      try {
        proc.kill("SIGTERM");
      } catch {
        // Already dead.
        clearTimeout(timeout);
        resolve();
      }
    });
  }

  private readMeta(): EngineMeta | null {
    try {
      if (!this.deps.exists(this.metaPath)) return null;
      return JSON.parse(this.deps.readFile(this.metaPath)) as EngineMeta;
    } catch {
      return null;
    }
  }

  private tryReadPid(): number | null {
    try {
      if (!this.deps.exists(this.pidPath)) return null;
      const text = this.deps.readFile(this.pidPath).trim();
      const pid = parseInt(text, 10);
      return Number.isFinite(pid) ? pid : null;
    } catch {
      return null;
    }
  }

  private cleanupFiles(): void {
    try {
      if (this.deps.exists(this.pidPath)) this.deps.unlink(this.pidPath);
    } catch {
      // Best effort.
    }
    try {
      if (this.deps.exists(this.metaPath)) this.deps.unlink(this.metaPath);
    } catch {
      // Best effort.
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function sha256(data: string): string {
  return createHash("sha256").update(data, "utf-8").digest("hex");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
