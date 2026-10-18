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
  requestControlDisable?(stateDir: string): Promise<ControlGateResult>;
  requestControlDisableSync?(stateDir: string): ControlGateResult;
}

/**
 * Closed result of one authenticated control-socket disable handshake.
 *
 * Only `approved` permits a signal; every other kind leaves the engine and
 * all control state untouched.
 */
export type ControlGateResult =
  | { kind: "approved"; fileCleanup: "clear" | "not_applicable" }
  | { kind: "refused" }
  | { kind: "unavailable" }
  | { kind: "timeout" }
  | { kind: "malformed" };

/** Raised when startup rollback cannot prove it may stop and clean up. */
export class EngineRecoveryError extends Error {
  constructor(
    readonly reason:
      | ControlGateResult["kind"]
      | "ownership_unknown"
      | "exited_before_handshake"
      | "exit_unconfirmed"
      | "stale_control_state",
    readonly stateDir: string,
    readonly startupError?: Error,
  ) {
    super(
      `Parapet engine recovery required (${reason}): engine and control ` +
        `state were preserved in ${stateDir}.` +
        (startupError ? ` Startup failure: ${startupError.message}` : ""),
    );
    this.name = "EngineRecoveryError";
  }
}

const CONTROL_SCHEMA = "parapet-control/v0";
const CONTROL_TIMEOUT_MS = 5_000;
const MAX_CONTROL_LINE_BYTES = 4096;

const UNAVAILABLE: ControlGateResult = { kind: "unavailable" };
const TIMEOUT: ControlGateResult = { kind: "timeout" };
const MALFORMED: ControlGateResult = { kind: "malformed" };
const REFUSED: ControlGateResult = { kind: "refused" };

/** Classify one raw `parapet-control/v0` disable response line. */
function classifyControlDisableLine(rawLine: string): ControlGateResult {
  let resp: unknown;
  try {
    resp = JSON.parse(rawLine);
  } catch {
    return MALFORMED;
  }
  if (resp === null || typeof resp !== "object" || Array.isArray(resp)) {
    return MALFORMED;
  }
  const obj = resp as Record<string, unknown>;
  if (
    obj.schema !== CONTROL_SCHEMA ||
    typeof obj.ok !== "boolean" ||
    typeof obj.stop_approved !== "boolean"
  ) {
    return MALFORMED;
  }
  if (
    obj.ok &&
    obj.stop_approved &&
    (obj.file_cleanup === "clear" || obj.file_cleanup === "not_applicable")
  ) {
    return { kind: "approved", fileCleanup: obj.file_cleanup };
  }
  return REFUSED;
}

/**
 * Accept only an exact `ControlGateResult` from an injected client; any other
 * value (including a bare boolean) is `malformed`.
 */
function normalizeControlGateResult(value: unknown): ControlGateResult {
  if (value === null || typeof value !== "object") return MALFORMED;
  const obj = value as Record<string, unknown>;
  const keys = Object.keys(obj).sort().join(",");
  switch (obj.kind) {
    case "approved":
      if (
        keys === "fileCleanup,kind" &&
        (obj.fileCleanup === "clear" || obj.fileCleanup === "not_applicable")
      ) {
        return { kind: "approved", fileCleanup: obj.fileCleanup };
      }
      return MALFORMED;
    case "refused":
    case "unavailable":
    case "timeout":
    case "malformed":
      return keys === "kind" ? { kind: obj.kind } : MALFORMED;
    default:
      return MALFORMED;
  }
}

function readControlToken(stateDir: string): string | null {
  const sockPath = join(stateDir, "control.sock");
  const tokenPath = join(stateDir, "control.token");
  try {
    if (!existsSync(sockPath) || !existsSync(tokenPath)) {
      return null;
    }
    const token = readFileSync(tokenPath, "utf-8").trim();
    return token ? token : null;
  } catch {
    return null;
  }
}

/** Default async authenticated control-socket disable handshake. */
export function defaultRequestControlDisable(
  stateDir: string,
): Promise<ControlGateResult> {
  if (process.platform === "win32") {
    return Promise.resolve(UNAVAILABLE);
  }
  const token = readControlToken(stateDir);
  if (token === null) {
    return Promise.resolve(UNAVAILABLE);
  }
  const sockPath = join(stateDir, "control.sock");

  const payload =
    JSON.stringify({
      schema: CONTROL_SCHEMA,
      token,
      action: "disable",
    }) + "\n";

  return new Promise<ControlGateResult>((resolve) => {
    let settled = false;
    const finish = (result: ControlGateResult): void => {
      if (settled) return;
      settled = true;
      resolve(result);
    };

    const sock = createConnection(sockPath);
    const chunks: Buffer[] = [];
    let received = 0;

    const firstLine = (): string =>
      Buffer.concat(chunks).toString("utf-8").split("\n", 1)[0].trim();

    const timer = setTimeout(() => {
      sock.destroy();
      finish(TIMEOUT);
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
        finish(MALFORMED);
        return;
      }
      if (chunk.includes(0x0a)) {
        clearTimeout(timer);
        sock.end();
        finish(classifyControlDisableLine(firstLine()));
      }
    });

    sock.on("end", () => {
      clearTimeout(timer);
      finish(chunks.length === 0 ? UNAVAILABLE : classifyControlDisableLine(firstLine()));
    });

    sock.on("error", () => {
      clearTimeout(timer);
      finish(UNAVAILABLE);
    });
  });
}

// Exit codes of the synchronous handshake helper process. On success the
// helper prints the raw response line, which the parent classifies.
const SYNC_EXIT_LINE = 0;
const SYNC_EXIT_UNAVAILABLE = 30;
const SYNC_EXIT_TIMEOUT = 40;
const SYNC_EXIT_MALFORMED = 50;

/** Default synchronous authenticated control-socket disable handshake. */
export function defaultRequestControlDisableSync(
  stateDir: string,
): ControlGateResult {
  if (process.platform === "win32") {
    return UNAVAILABLE;
  }
  const token = readControlToken(stateDir);
  if (token === null) {
    return UNAVAILABLE;
  }
  const sockPath = join(stateDir, "control.sock");

  const script = [
    'const net = require("node:net");',
    "const [sockPath, token] = process.argv.slice(1);",
    'const req = JSON.stringify({ schema: "parapet-control/v0", token, action: "disable" }) + "\\n";',
    "const s = net.createConnection(sockPath);",
    "const chunks = [];",
    "let received = 0;",
    "let finished = false;",
    `const done = (code, line) => { if (finished) return; finished = true; clearTimeout(t); s.destroy(); if (line !== undefined) process.stdout.write(line + "\\n"); process.exitCode = code; };`,
    `const t = setTimeout(() => done(${SYNC_EXIT_TIMEOUT}), ${CONTROL_TIMEOUT_MS});`,
    's.on("connect", () => s.write(req));',
    's.on("data", (d) => {',
    "  chunks.push(d);",
    "  received += d.length;",
    `  if (received > ${MAX_CONTROL_LINE_BYTES}) return done(${SYNC_EXIT_MALFORMED});`,
    "  const buf = Buffer.concat(chunks);",
    "  if (buf.includes(0x0a)) {",
    `    done(${SYNC_EXIT_LINE}, buf.toString("utf-8").split("\\n", 1)[0].trim());`,
    "  }",
    "});",
    `s.on("end", () => { if (chunks.length === 0) return done(${SYNC_EXIT_UNAVAILABLE}); done(${SYNC_EXIT_LINE}, Buffer.concat(chunks).toString("utf-8").split("\\n", 1)[0].trim()); });`,
    `s.on("error", () => done(${SYNC_EXIT_UNAVAILABLE}));`,
  ].join("\n");

  try {
    const res = spawnSync(process.execPath, ["-e", script, sockPath, token], {
      timeout: CONTROL_TIMEOUT_MS + 1_000,
      stdio: ["ignore", "pipe", "ignore"],
      maxBuffer: MAX_CONTROL_LINE_BYTES * 2,
      encoding: "utf-8",
    });
    if (res.error) {
      return (res.error as NodeJS.ErrnoException).code === "ETIMEDOUT"
        ? TIMEOUT
        : UNAVAILABLE;
    }
    switch (res.status) {
      case SYNC_EXIT_LINE:
        return classifyControlDisableLine(String(res.stdout).split("\n", 1)[0]);
      case SYNC_EXIT_UNAVAILABLE:
        return UNAVAILABLE;
      case SYNC_EXIT_TIMEOUT:
        return TIMEOUT;
      default:
        return MALFORMED;
    }
  } catch {
    return UNAVAILABLE;
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

      // Stale or mismatched and port is free — clean up files, unless they
      // may belong to a gated engine whose evidence must be recovered first.
      if (this.requiresControlGate()) {
        throw new EngineRecoveryError("stale_control_state", this.parapetDir);
      }
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
      throw await this.rollbackStartup(
        err instanceof Error ? err : new Error(String(err)),
      );
    }

    // Verify our child process is still alive — if it exited immediately,
    // the readiness heartbeat may have hit a stale listener on the port.
    const exitCode = this.process.exitCode;
    if (exitCode !== null) {
      throw await this.rollbackStartup(
        new Error(
          `Engine process exited immediately with code ${exitCode}. ` +
            `Check ${this.logPath} for details.`,
        ),
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

      const gated = this.requiresControlGate();
      if (gated) {
        if (!this.ownershipConfirmed()) {
          // Cannot prove the gated engine is ours: leave everything intact.
          return;
        }
        const gate = await this.checkControlGateAsync();
        if (gate.kind !== "approved") {
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
      if (!this.ownershipConfirmed()) {
        return;
      }
      const gate = this.checkControlGateSync();
      if (gate.kind !== "approved") {
        // Refused or unable to complete synchronous handshake: leave engine
        // and all control state intact and never signal or unlink files.
        return;
      }
      // Approved synchronously: request a graceful stop at most. Exit cannot
      // be confirmed synchronously, so every control file and the manager's
      // live state are preserved for recovery.
      const proc = this.process;
      if (proc?.pid && !hasExited(proc)) {
        try {
          process.kill(proc.pid, "SIGTERM");
        } catch {
          // Preserve state either way.
        }
      }
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

  /**
   * True only when this manager spawned the live child it holds and any
   * engine owner record names that same pid.
   */
  private ownershipConfirmed(): boolean {
    const pid = this.process?.pid;
    if (!this.ownsProcess || typeof pid !== "number") {
      return false;
    }
    if (!this.deps.exists(this.ownerPath)) {
      return true;
    }
    return this.readOwnerRecord()?.pid === pid;
  }

  /**
   * Roll back a spawned engine whose startup failed; returns the error the
   * caller throws.
   *
   * Without an effect root or any control/token/owner artifact this is the
   * legacy ungated stop. Otherwise only an async approved gate permits
   * SIGTERM, and controls are removed only after confirmed exit. Every other
   * state sends no signal, unlinks nothing, and yields a recovery error.
   */
  private async rollbackStartup(startupError: Error): Promise<Error> {
    if (!this.requiresControlGate()) {
      await this.stop();
      return startupError;
    }
    const recovery = (reason: EngineRecoveryError["reason"]): Error =>
      new EngineRecoveryError(reason, this.parapetDir, startupError);

    const proc = this.process;
    if (proc === null || !this.ownershipConfirmed()) {
      return recovery("ownership_unknown");
    }
    if (hasExited(proc)) {
      return recovery("exited_before_handshake");
    }
    const gate = await this.checkControlGateAsync();
    if (gate.kind !== "approved") {
      return recovery(gate.kind);
    }
    if (!(await this.terminateProcess(proc, false))) {
      return recovery("exit_unconfirmed");
    }
    this.process = null;
    this.stopHeartbeat();
    this.removeShutdownHandlers();
    this.closeLogFd();
    this.cleanupFiles();
    this.initialized = false;
    this.ownsProcess = false;
    return startupError;
  }

  private async checkControlGateAsync(): Promise<ControlGateResult> {
    if (!this.deps.requestControlDisable) {
      return UNAVAILABLE;
    }
    try {
      return normalizeControlGateResult(
        await this.deps.requestControlDisable(this.parapetDir),
      );
    } catch {
      return UNAVAILABLE;
    }
  }

  private checkControlGateSync(): ControlGateResult {
    if (!this.deps.requestControlDisableSync) {
      return UNAVAILABLE;
    }
    try {
      return normalizeControlGateResult(
        this.deps.requestControlDisableSync(this.parapetDir),
      );
    } catch {
      return UNAVAILABLE;
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
        // Ungated stop treats a failed signal as already dead; a gated stop
        // requires observed exit before controls may be removed.
        clearTimeout(timeout);
        proc.removeListener("exit", onExit);
        resolve(allowSigkill || hasExited(proc));
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

function hasExited(proc: ChildProcess): boolean {
  return proc.exitCode !== null || (proc.signalCode ?? null) !== null;
}

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
