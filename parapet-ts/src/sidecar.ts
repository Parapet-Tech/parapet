// Copyright 2026 The Parapet Project
// SPDX-License-Identifier: Apache-2.0

import { createHash } from "node:crypto";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import {
  readFileSync,
  writeFileSync,
  unlinkSync,
  existsSync,
  mkdirSync,
  chmodSync,
  realpathSync,
  openSync,
  closeSync,
} from "node:fs";
import { createConnection } from "node:net";
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
  requestControlDisable?(stateDir: string): Promise<boolean>;
  requestControlDisableSync?(stateDir: string): boolean;
}

const CONTROL_SCHEMA = "parapet-control/v0";
const CONTROL_TIMEOUT_MS = 5_000;
const MAX_CONTROL_LINE_BYTES = 4096;

interface ControlDisableResponse {
  schema?: unknown;
  ok?: unknown;
  stop_approved?: unknown;
  file_cleanup?: unknown;
}

function isApprovedControlDisable(resp: unknown): boolean {
  if (resp === null || typeof resp !== "object") return false;
  const obj = resp as ControlDisableResponse;
  return (
    obj.schema === CONTROL_SCHEMA &&
    obj.ok === true &&
    obj.stop_approved === true &&
    (obj.file_cleanup === "clear" || obj.file_cleanup === "not_applicable")
  );
}

/** Default async authenticated control-socket disable handshake. */
export function defaultRequestControlDisable(
  stateDir: string,
): Promise<boolean> {
  if (process.platform === "win32") {
    return Promise.resolve(false);
  }
  const sockPath = join(stateDir, "control.sock");
  const tokenPath = join(stateDir, "control.token");
  let token: string;
  try {
    if (!existsSync(sockPath) || !existsSync(tokenPath)) {
      return Promise.resolve(false);
    }
    token = readFileSync(tokenPath, "utf-8").trim();
    if (!token) {
      return Promise.resolve(false);
    }
  } catch {
    return Promise.resolve(false);
  }

  const payload =
    JSON.stringify({
      schema: CONTROL_SCHEMA,
      token,
      action: "disable",
    }) + "\n";

  return new Promise<boolean>((resolve) => {
    let settled = false;
    const finish = (result: boolean): void => {
      if (settled) return;
      settled = true;
      resolve(result);
    };

    const sock = createConnection(sockPath);
    const chunks: Buffer[] = [];
    let received = 0;

    const timer = setTimeout(() => {
      sock.destroy();
      finish(false);
    }, CONTROL_TIMEOUT_MS);

    sock.on("connect", () => {
      sock.write(payload, "utf-8");
    });

    sock.on("data", (chunk: Buffer) => {
      chunks.push(chunk);
      received += chunk.length;
      if (received > MAX_CONTROL_LINE_BYTES) {
        clearTimeout(timer);
        sock.destroy();
        finish(false);
        return;
      }
      if (chunk.includes(0x0a)) {
        clearTimeout(timer);
        sock.end();
        const rawLine = Buffer.concat(chunks)
          .toString("utf-8")
          .split("\n", 1)[0]
          .trim();
        try {
          finish(isApprovedControlDisable(JSON.parse(rawLine)));
        } catch {
          finish(false);
        }
      }
    });

    sock.on("end", () => {
      clearTimeout(timer);
      if (chunks.length === 0 || received > MAX_CONTROL_LINE_BYTES) {
        finish(false);
        return;
      }
      const rawLine = Buffer.concat(chunks)
        .toString("utf-8")
        .split("\n", 1)[0]
        .trim();
      try {
        finish(isApprovedControlDisable(JSON.parse(rawLine)));
      } catch {
        finish(false);
      }
    });

    sock.on("error", () => {
      clearTimeout(timer);
      finish(false);
    });
  });
}

/** Default synchronous authenticated control-socket disable handshake. */
export function defaultRequestControlDisableSync(stateDir: string): boolean {
  if (process.platform === "win32") {
    return false;
  }
  const sockPath = join(stateDir, "control.sock");
  const tokenPath = join(stateDir, "control.token");
  let token: string;
  try {
    if (!existsSync(sockPath) || !existsSync(tokenPath)) {
      return false;
    }
    token = readFileSync(tokenPath, "utf-8").trim();
    if (!token) {
      return false;
    }
  } catch {
    return false;
  }

  const script = [
    'const net = require("node:net");',
    "const [sockPath, token] = process.argv.slice(1);",
    'const req = JSON.stringify({ schema: "parapet-control/v0", token, action: "disable" }) + "\\n";',
    "const s = net.createConnection(sockPath);",
    'let buf = "";',
    "const t = setTimeout(() => { s.destroy(); process.exit(1); }, 5000);",
    's.on("connect", () => s.write(req));',
    's.on("data", (d) => {',
    '  buf += d.toString("utf-8");',
    "  if (buf.length > 4096) { clearTimeout(t); s.destroy(); process.exit(1); }",
    '  if (buf.includes("\\n")) {',
    "    clearTimeout(t);",
    "    s.end();",
    "    try {",
    '      const r = JSON.parse(buf.split("\\n", 1)[0].trim());',
    '      const ok = r && r.schema === "parapet-control/v0" && r.ok === true && r.stop_approved === true && (r.file_cleanup === "clear" || r.file_cleanup === "not_applicable");',
    "      process.exit(ok ? 0 : 1);",
    "    } catch { process.exit(1); }",
    "  }",
    "});",
    's.on("error", () => { clearTimeout(t); process.exit(1); });',
  ].join("\n");

  try {
    const res = spawnSync(process.execPath, ["-e", script, sockPath, token], {
      timeout: CONTROL_TIMEOUT_MS,
      stdio: "ignore",
    });
    return res.status === 0;
  } catch {
    return false;
  }
}

/** Default deps wrapping Node built-ins. */
export function defaultDeps(): SidecarDeps {
  return {
    readFile: (p) => readFileSync(p, "utf-8"),
    writeFile: (p, d) => writeFileSync(p, d, "utf-8"),
    unlink: (p) => unlinkSync(p),
    exists: (p) => existsSync(p),
    mkdirp: (p) => {
      mkdirSync(p, { recursive: true, mode: 0o700 });
      if (process.platform !== "win32") {
        try {
          chmodSync(p, 0o700);
        } catch {
          // Best effort.
        }
      }
    },
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
    requestControlDisable: (stateDir) =>
      defaultRequestControlDisable(stateDir),
    requestControlDisableSync: (stateDir) =>
      defaultRequestControlDisableSync(stateDir),
  };
}

// ---------------------------------------------------------------------------
// Engine metadata
// ---------------------------------------------------------------------------

interface EngineMeta {
  port: number;
  configPath: string;
  configHash: string;
  sdkConfigHash?: string;
  startedAt: string;
}

interface EngineOwnerRecord {
  schema?: string;
  pid?: number;
  port?: number;
  config_hash?: string;
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
  stateDir?: string;
  engineBinary?: string;
  engineArgs?: string[];
  deps?: SidecarDeps;
}

export class EngineManager {
  private readonly deps: SidecarDeps;
  private readonly _stateDir: string | null;
  private readonly _engineArgs: string[];
  private readonly parapetDir: string;
  private readonly pidPath: string;
  private readonly metaPath: string;
  private readonly logPath: string;
  private readonly controlSockPath: string;
  private readonly controlTokenPath: string;
  private readonly ownerPath: string;

  private process: ChildProcess | null = null;
  private logFd: number | null = null;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private initialized = false;
  /** True when we spawned the process; false when reusing from PID/owner file. */
  private ownsProcess = false;
  /** True during startup readiness rollback before control gate is active. */
  private rollingBackStartup = false;
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
    this._engineArgs = opts.engineArgs ? [...opts.engineArgs] : [];
    const resolvedStateDir =
      opts.stateDir ?? extractStateDirFromArgs(this._engineArgs);
    this._stateDir = resolvedStateDir;
    this.parapetDir = resolvedStateDir ?? join(this.deps.homedir(), ".parapet");
    this.pidPath = join(this.parapetDir, "engine.pid");
    this.metaPath = join(this.parapetDir, "engine.meta");
    this.logPath = join(this.parapetDir, "engine.log");
    this.controlSockPath = join(this.parapetDir, "control.sock");
    this.controlTokenPath = join(this.parapetDir, "control.token");
    this.ownerPath = join(this.parapetDir, "engine.owner.json");
  }

  get configPath(): string {
    return this.opts.configPath;
  }

  get stateDir(): string | null {
    return this._stateDir;
  }

  get engineArgs(): readonly string[] {
    return this._engineArgs;
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

  /**
   * Check if runtime options (stateDir, engineBinary, engineArgs) match
   * this manager's configured options.
   */
  matchesRuntimeOptions(opts: {
    stateDir?: string;
    engineBinary?: string;
    engineArgs?: string[];
  }): boolean {
    const candidateArgs = opts.engineArgs ?? [];
    const candidateStateDir =
      opts.stateDir ?? extractStateDirFromArgs(candidateArgs);
    if (candidateStateDir !== this._stateDir) return false;
    if ((opts.engineBinary ?? undefined) !== (this.opts.engineBinary ?? undefined)) {
      return false;
    }
    if (candidateArgs.length !== this._engineArgs.length) return false;
    for (let i = 0; i < candidateArgs.length; i++) {
      if (candidateArgs[i] !== this._engineArgs[i]) return false;
    }
    return true;
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
        (meta.sdkConfigHash ?? meta.configHash) === configHash &&
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
        (meta.sdkConfigHash ?? meta.configHash) === configHash &&
        meta.port === this.port
      ) {
        // Config matches — check if still alive via heartbeat.
        if (await this.heartbeatOnce()) {
          this.process = null;
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
    } else if (this._stateDir !== null && this.deps.exists(this.ownerPath)) {
      const owner = this.readOwnerRecord();
      if (
        owner !== null &&
        owner.port === this.port &&
        (await this.heartbeatOnce())
      ) {
        this.process = null;
        this.initialized = true;
        this.ownsProcess = false;
        this.startHeartbeat();
        this.registerShutdownHandlers();
        return;
      } else if (await this.heartbeatOnce()) {
        throw new Error(
          `Port ${this.port} is already in use by an engine with ` +
            `different config. Stop the existing engine first, or ` +
            `use a different port.`,
        );
      }
    }

    // Ensure parapet / state dir exists.
    this.deps.mkdirp(this.parapetDir);

    // Resolve engine binary.
    const binary = this.resolveEngineBinary();

    // Open log file.
    this.logFd = this.deps.openLogFile(this.logPath);

    // Build spawn arguments.
    const spawnArgs = ["--config", canonicalPath, "--port", String(this.port)];
    if (this._stateDir !== null && !hasFlag(this._engineArgs, "--state-dir")) {
      spawnArgs.push("--state-dir", this._stateDir);
    }
    if (this._engineArgs.length > 0) {
      spawnArgs.push(...this._engineArgs);
    }

    // Spawn.
    this.process = this.deps.spawnProcess(binary, spawnArgs, this.logFd);
    this.ownsProcess = true;

    const startedAt = new Date().toISOString();
    // Write PID + meta.
    this.deps.writeFile(this.pidPath, String(this.process.pid));
    this.deps.writeFile(
      this.metaPath,
      JSON.stringify({
        port: this.port,
        configPath: canonicalPath,
        configHash,
        sdkConfigHash: configHash,
        startedAt,
      } satisfies EngineMeta),
    );

    // Wait for readiness. If it fails, clean up the process/files we just
    // created so we don't leak a child process or stale metadata.
    try {
      await this.waitForReady();
    } catch (err) {
      this.rollingBackStartup = true;
      try {
        await this.stop();
      } finally {
        this.rollingBackStartup = false;
      }
      throw err;
    }

    // Verify our child process is still alive — if it exited immediately,
    // the readiness heartbeat may have hit a stale listener on the port.
    const exitCode = this.process.exitCode;
    if (exitCode !== null) {
      this.rollingBackStartup = true;
      try {
        await this.stop();
      } finally {
        this.rollingBackStartup = false;
      }
      throw new Error(
        `Engine process exited immediately with code ${exitCode}. ` +
          `Check ${this.logPath} for details.`,
      );
    }

    const owner = this.readOwnerRecord();
    if (owner?.config_hash) {
      this.deps.writeFile(
        this.metaPath,
        JSON.stringify({
          port: this.port,
          configPath: canonicalPath,
          configHash: owner.config_hash,
          sdkConfigHash: configHash,
          startedAt,
        } satisfies EngineMeta),
      );
    }

    this.initialized = true;
    this.startHeartbeat();
    this.registerShutdownHandlers();
  }

  /** Stop the engine and clean up all resources (if owned and permitted by cleanup gate). */
  async stop(): Promise<void> {
    if (this.cleaning) return;
    this.cleaning = true;

    try {
      // Reused / unowned engines are never signaled or unlinked.
      if (!this.ownsProcess) {
        this.stopHeartbeat();
        this.removeShutdownHandlers();
        this.closeLogFd();
        this.process = null;
        this.initialized = false;
        this.ownsProcess = false;
        return;
      }

      const gated = !this.rollingBackStartup && this.requiresControlGate();
      if (gated) {
        const approved = await this.checkControlGateAsync();
        if (!approved) {
          // Refused by cleanup gate: leave child process and control files intact.
          return;
        }
      }

      if (this.process !== null) {
        const stopped = await this.terminateProcess(this.process, !gated);
        if (!stopped) {
          // An approved control response does not prove process exit. Preserve
          // the live manager and every control file so a later stop can retry.
          return;
        }
        this.process = null;
      }

      this.stopHeartbeat();
      this.removeShutdownHandlers();
      this.closeLogFd();
      this.cleanupFiles();

      this.initialized = false;
      this.ownsProcess = false;
    } finally {
      this.cleaning = false;
    }
  }

  /** Synchronous shutdown hook (for process.on("exit") and explicit stopSync()). */
  stopSync(): void {
    // Reused / unowned engines are never signaled or unlinked.
    if (!this.ownsProcess) {
      this.stopHeartbeat();
      this.removeShutdownHandlers();
      this.closeLogFd();
      this.process = null;
      this.initialized = false;
      this.ownsProcess = false;
      return;
    }

    if (this.requiresControlGate()) {
      const approved = this.checkControlGateSync();
      if (!approved) {
        // Refused or unable to complete synchronous handshake: leave engine
        // and all control state intact and never send SIGKILL or unlink files.
        return;
      }
      if (this.process?.pid && this.process.exitCode === null) {
        try {
          process.kill(this.process.pid, "SIGKILL");
        } catch {
          // Without proof that the process stopped, preserve its reachable
          // state instead of claiming successful synchronous shutdown.
          return;
        }
      }
      this.stopHeartbeat();
      this.removeShutdownHandlers();
      this.closeLogFd();
      this.cleanupFiles();
      this.process = null;
      this.initialized = false;
      this.ownsProcess = false;
      return;
    }

    this.stopHeartbeat();
    this.removeShutdownHandlers();
    if (this.process?.pid) {
      try {
        process.kill(this.process.pid, "SIGKILL");
      } catch {
        // Best effort.
      }
    }
    this.closeLogFd();
    this.cleanupFiles();
    this.process = null;
    this.initialized = false;
    this.ownsProcess = false;
  }

  // -----------------------------------------------------------------------
  // Internals
  // -----------------------------------------------------------------------

  private requiresControlGate(): boolean {
    if (hasFlag(this._engineArgs, "--effect-root")) {
      return true;
    }
    return (
      this.deps.exists(this.controlSockPath) ||
      this.deps.exists(this.controlTokenPath) ||
      this.deps.exists(this.ownerPath)
    );
  }

  private async checkControlGateAsync(): Promise<boolean> {
    if (!this.deps.requestControlDisable) {
      return false;
    }
    try {
      return await this.deps.requestControlDisable(this.parapetDir);
    } catch {
      return false;
    }
  }

  private checkControlGateSync(): boolean {
    if (!this.deps.requestControlDisableSync) {
      return false;
    }
    try {
      return this.deps.requestControlDisableSync(this.parapetDir);
    } catch {
      return false;
    }
  }

  private closeLogFd(): void {
    if (this.logFd !== null) {
      try {
        this.deps.closeFile(this.logFd);
      } catch {
        // Best effort.
      }
      this.logFd = null;
    }
  }

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
    if (
      this.heartbeatTimer &&
      typeof this.heartbeatTimer === "object" &&
      "unref" in this.heartbeatTimer
    ) {
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
        if (!this.initialized) {
          process.exit(130);
        }
      });
    };
    this.sigtermHandler = () => {
      void this.stop().then(() => {
        if (!this.initialized) {
          process.exit(143);
        }
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

  private async terminateProcess(
    proc: ChildProcess,
    allowSigkill = true,
  ): Promise<boolean> {
    if (proc.exitCode !== null) {
      return true;
    }
    return new Promise<boolean>((resolve) => {
      const onExit = (): void => {
        clearTimeout(timeout);
        resolve(true);
      };
      const timeout = setTimeout(() => {
        proc.removeListener("exit", onExit);
        if (allowSigkill) {
          try {
            proc.kill("SIGKILL");
          } catch {
            // Already dead.
          }
          resolve(true);
          return;
        }
        resolve(false);
      }, SHUTDOWN_GRACE_MS);

      proc.once("exit", onExit);

      try {
        proc.kill("SIGTERM");
      } catch {
        // Already dead.
        clearTimeout(timeout);
        proc.removeListener("exit", onExit);
        resolve(true);
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

  private readOwnerRecord(): EngineOwnerRecord | null {
    try {
      if (!this.deps.exists(this.ownerPath)) return null;
      const parsed = JSON.parse(
        this.deps.readFile(this.ownerPath),
      ) as EngineOwnerRecord;
      if (parsed === null || typeof parsed !== "object") return null;
      return parsed;
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
    for (const p of [
      this.pidPath,
      this.metaPath,
      this.controlSockPath,
      this.controlTokenPath,
      this.ownerPath,
    ]) {
      try {
        if (this.deps.exists(p)) this.deps.unlink(p);
      } catch {
        // Best effort.
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function hasFlag(args: readonly string[], flag: string): boolean {
  return args.some((a) => a === flag || a.startsWith(`${flag}=`));
}

function extractStateDirFromArgs(args: readonly string[]): string | null {
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--state-dir" && i + 1 < args.length) {
      return args[i + 1];
    }
    if (arg.startsWith("--state-dir=")) {
      return arg.slice("--state-dir=".length);
    }
  }
  return null;
}

function sha256(data: string): string {
  return createHash("sha256").update(data, "utf-8").digest("hex");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
