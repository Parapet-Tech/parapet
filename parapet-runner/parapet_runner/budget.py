"""Configurable host budgets and best-effort POSIX process containment."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import resource
import stat
from contextlib import contextmanager
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ResourceBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    max_rss_bytes: int = Field(default=12 * 1024**3, gt=0)
    max_processes: int = Field(default=32, gt=0)
    max_workers: int = Field(default=2, gt=0)
    wall_seconds: float = Field(default=14400, gt=0)
    min_free_bytes: int = Field(default=4 * 1024**3, ge=0)
    expected_rss_bytes: int | None = Field(default=None, gt=0)
    poll_seconds: float = Field(default=2, gt=0)
    nice: int = Field(default=10, ge=0, le=19)


CONTAINMENT_LIMIT = (
    "A double-fork daemon reparented to PID 1 between polls can escape tracking; "
    "macOS has no reliable user-space hook without cgroups."
)


class BudgetReceipt(BaseModel):
    budget: ResourceBudget
    peak_rss_bytes: int = 0
    peak_processes: int = 0
    wall_seconds: float = 0
    cpu_user_s: float = 0
    cpu_sys_s: float = 0
    exit_reason: str = "ok"
    no_lock: bool = False
    detail: str = ""
    diagnostics: list[str] = Field(default_factory=list)
    containment_limit: str = CONTAINMENT_LIMIT


class HostProbe(Protocol):
    def available_bytes(self) -> int: ...
    def cpu_count(self) -> int: ...


class ProcessTreeProbe(Protocol):
    def sample(self, root: int, remembered: set[int]) -> dict[int, int]:
        """Live tracked PID -> RSS bytes; union PGID and remembered PPID descendants."""
        ...


class HostLock(Protocol):
    def acquire(self, command: Sequence[str]) -> str | None:
        """Return holder description on conflict, None on success."""
        ...
    def release(self) -> None: ...


class JobLease(Protocol):
    @property
    def held(self) -> bool: ...
    def acquire(self, command: Sequence[str]) -> str | None: ...
    def release(self) -> None: ...


class LockJobLease:
    def __init__(self, lock: HostLock) -> None:
        self.lock = lock
        self.held = False

    def acquire(self, command: Sequence[str]) -> str | None:
        if self.held:
            raise RuntimeError("Job lease already held")
        holder = self.lock.acquire(command)
        self.held = holder is None
        return holder

    def release(self) -> None:
        if self.held:
            self.lock.release()
            self.held = False


class JobLockHeld(RuntimeError):
    pass


class Clock(Protocol):
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class ProcessLauncher(Protocol):
    def __call__(self, args: Sequence[str], **kwargs: Any) -> Any: ...


class Signaler(Protocol):
    def group(self, pgid: int, sig: int) -> str | None: ...
    def pid(self, pid: int, sig: int) -> None: ...


ENV_FIELDS = {
    "max_rss_bytes": "PARAPET_BUDGET_MAX_RSS_GB",
    "expected_rss_bytes": "PARAPET_BUDGET_EXPECTED_RSS_GB",
    "max_processes": "PARAPET_BUDGET_MAX_PROCESSES",
    "max_workers": "PARAPET_BUDGET_MAX_WORKERS",
    "wall_seconds": "PARAPET_BUDGET_WALL",
    "min_free_bytes": "PARAPET_BUDGET_MIN_FREE_GB",
    "nice": "PARAPET_BUDGET_NICE",
    "poll_seconds": "PARAPET_BUDGET_POLL",
}
FLAGS = {"expected_rss_bytes": "--expected-rss", "max_rss_bytes": "--max-rss", "max_processes": "--max-processes",
         "max_workers": "--max-workers", "wall_seconds": "--wall",
         "min_free_bytes": "--min-free", "nice": "--nice", "poll_seconds": "--poll"}
EXIT_CODES = {"rss_exceeded": 120, "process_cap": 121, "wall_exceeded": 122,
              "refused_preflight": 123, "lock_held": 124}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT", "PARAPET_MAX_WORKERS")


def quantity(value: Any, *, duration: bool = False, bare_gb: bool = False) -> float:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]*)\s*", str(value))
    if not match:
        raise ValueError(f"invalid quantity {value!r}")
    number, unit = match.groups()
    factors = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400} if duration else {
        "": 1024**3 if bare_gb else 1, "b": 1, "k": 1024, "kb": 1024, "kib": 1024,
        "m": 1024**2, "mb": 1024**2, "mib": 1024**2,
        "g": 1024**3, "gb": 1024**3, "gib": 1024**3,
        "t": 1024**4, "tb": 1024**4, "tib": 1024**4}
    if unit.lower() not in factors:
        raise ValueError(f"invalid unit {unit!r}")
    result = float(number) * factors[unit.lower()]
    if not math.isfinite(result):
        raise ValueError("quantity must be finite")
    return result


class BudgetConfigError(ValueError):
    """Invalid budget input, suitable for a one-line CLI diagnostic."""


def resolve_budget(cli: Mapping[str, Any] | None = None, *, env: Mapping[str, str] | None = None
                   ) -> tuple[ResourceBudget, dict[str, str]]:
    cli = cli or {}
    env = os.environ if env is None else env
    values = ResourceBudget().model_dump()
    sources = dict.fromkeys(values, "built-in default")
    path = Path(env.get("PARAPET_BUDGET_CONFIG", "~/.config/parapet/budget.yaml")).expanduser()
    config: dict = {}
    if path.exists() or "PARAPET_BUDGET_CONFIG" in env:
        try:
            loaded = yaml.safe_load(path.read_text())
            config = {} if loaded is None else loaded
            if not isinstance(config, dict) or set(config) - {"defaults", "profiles"}:
                raise ValueError("expected defaults and profiles mappings")
            if not isinstance(config.get("profiles", {}), Mapping):
                raise ValueError("profiles must be a mapping")
        except Exception as exc:
            raise BudgetConfigError(f"PARAPET_BUDGET_CONFIG ({path}): invalid YAML/config; fix the file ({str(exc).splitlines()[0]})") from exc

    def apply(layer: Mapping, source: str, *, gb: bool = False) -> None:
        if not isinstance(layer, Mapping):
            raise BudgetConfigError(f"{source}: expected mapping; use budget field/value pairs")
        for key, value in layer.items():
            if key not in values:
                raise BudgetConfigError(f"{source}: unknown budget field {key}; use a supported budget field")
            try:
                if key == "expected_rss_bytes" and value is None and not gb:
                    values[key], sources[key] = None, source
                    continue
                if key in {"max_rss_bytes", "min_free_bytes", "expected_rss_bytes"}:
                    parsed = quantity(value, bare_gb=gb)
                    if parsed != int(parsed):
                        raise ValueError("size must resolve to whole bytes")
                    value = int(parsed)
                elif key in {"wall_seconds", "poll_seconds"}:
                    value = quantity(value, duration=True)
                else:
                    if isinstance(value, bool) or str(value) != str(int(value)):
                        raise ValueError("expected integer")
                    value = int(value)
                ResourceBudget.model_validate({**values, key: value})
            except (ValueError, TypeError, OverflowError) as exc:
                detail = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
                raise BudgetConfigError(f"{source} ({key}): {detail}; set {FLAGS[key]} / {ENV_FIELDS[key]} to a valid value") from exc
            values[key], sources[key] = value, source

    apply(config.get("defaults", {}), f"config {path}")
    profile = cli.get("profile") or env.get("PARAPET_BUDGET_PROFILE")
    if profile:
        profiles = config.get("profiles", {})
        if not isinstance(profiles, Mapping) or profile not in profiles:
            raise BudgetConfigError(f"--profile / PARAPET_BUDGET_PROFILE: unknown profile {profile!r} in {path}; define it under profiles or select an existing profile")
        apply(profiles[profile], f"profile {profile} ({path})")
    for key, var in ENV_FIELDS.items():
        if var in env:
            apply({key: env[var]}, var, gb=key.endswith("_bytes"))
    apply({k: v for k, v in cli.items() if k in values and v is not None}, "CLI flags")
    return ResourceBudget(**values), sources


class SystemHostProbe:
    def available_bytes(self) -> int:
        if sys.platform == "darwin":
            output = subprocess.check_output(["vm_stat"], text=True)
            page_size = int(re.search(r"page size of (\d+) bytes", output).group(1))
            pages = sum(int(re.search(rf"Pages {key}:\s*(\d+)", output).group(1))
                        for key in ("free", "inactive", "speculative", "purgeable"))
            return pages * page_size
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("Host memory probe unavailable; refusing launch")

    def cpu_count(self) -> int:
        return os.cpu_count() or 1


class PsProcessTreeProbe:
    def sample(self, root: int, remembered: set[int]) -> dict[int, int]:
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid=,rss=,stat="], text=True)
        return self.parse(output, root, remembered)

    @staticmethod
    def parse(output: str, root: int, remembered: set[int]) -> dict[int, int]:
        rows = {}
        for line in output.splitlines():
            pid, ppid, pgid, rss, state = line.split()
            if "Z" not in state:
                rows[int(pid)] = (int(ppid), int(pgid), int(rss) * 1024)
        tracked = ({pid for pid in remembered if pid in rows}
                   | {pid for pid, row in rows.items() if row[1] == root}
                   | ({root} if root in rows else set()))
        while True:
            descendants = {pid for pid, row in rows.items() if row[0] in tracked}
            expanded = tracked | descendants
            if expanded == tracked:
                break
            tracked = expanded
        return {pid: rows[pid][2] for pid in tracked}


class FileHostLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def acquire(self, command: Sequence[str]) -> str | None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            os.fchmod(fd, 0o600)
            if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                raise PermissionError("Job lock metadata requires mode 0600")
            handle = os.fdopen(fd, "r+")
        except BaseException:
            os.close(fd)
            raise
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            try:
                metadata = json.load(handle)
                pid = int(metadata["pid"])
                program = Path(str(metadata["executable"])).name[:64]
                holder = f"PID {pid} executable {program}"
            except (ValueError, KeyError, TypeError):
                holder = "holder metadata unavailable"
            handle.close()
            return holder
        self.handle = handle
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "executable": Path(command[0]).name[:64] if command else "unknown"}))
            handle.flush()
        except BaseException:
            self.release()
            raise
        return None

    def release(self) -> None:
        if self.handle is not None:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            finally:
                self.handle.close()
                self.handle = None


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class PosixSignaler:
    def group(self, pgid: int, sig: int) -> str | None:
        try:
            os.killpg(pgid, sig)
        except PermissionError:
            # macOS can deny killpg when only unreaped zombies remain.
            return "group_signal_eperm"
        except ProcessLookupError:
            pass
        return None
    def pid(self, pid: int, sig: int) -> None:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


class BudgetedExecutor:
    """CommandExecutor with receipts available after each invocation.

    run_with_receipt returns the CommandResult and its receipt explicitly.
    Dependencies are assigned here; probes, locks and child launch run only in run().
    """
    def __init__(self, budget: ResourceBudget, *, host: HostProbe, tree: ProcessTreeProbe,
                 lock: HostLock, clock: Clock, launcher: ProcessLauncher,
                 signaler: Signaler, environ: Mapping[str, str], lease: JobLease, no_lock: bool = False) -> None:
        self.budget = budget
        self.host, self.tree, self.lock, self.clock = host, tree, lock, clock
        self.launcher, self.signaler, self.environ = launcher, signaler, environ
        self.no_lock = no_lock
        self.lease = lease
        self.receipts: list[BudgetReceipt] = []

    @contextmanager
    def job(self, executable: str):
        """Hold the injected lease through caller output and receipt emission."""
        acquired = False
        receipt = BudgetReceipt(budget=self.budget, no_lock=self.no_lock)
        start = self.clock.monotonic()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            if not self.no_lock:
                holder = self.lease.acquire([executable])
                if holder is not None:
                    receipt.exit_reason = "lock_held"
                    receipt.detail = f"Parapet job lock held by {holder}"
                    raise JobLockHeld(receipt.detail)
                acquired = True
            def snapshot():
                self.update_job_receipt(receipt, start, usage)
                return receipt
            yield snapshot
        finally:
            self.update_job_receipt(receipt, start, usage)
            if acquired:
                self.lease.release()

    def update_job_receipt(self, receipt, start, usage):
        current = resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.wall_seconds = self.clock.monotonic() - start
        receipt.cpu_user_s = current.ru_utime - usage.ru_utime
        receipt.cpu_sys_s = current.ru_stime - usage.ru_stime

    def run_with_receipt(self, args: Sequence[str], *, cwd: Path | None = None):
        result = self.run(args, cwd=cwd)
        return result, self.receipts[-1]

    def run(self, args: Sequence[str], *, cwd: Path | None = None):
        from .baseline import CommandResult
        budget = self.budget
        receipt = BudgetReceipt(budget=budget, no_lock=self.no_lock)
        start = self.clock.monotonic()
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        child = None
        tracked: set[int] = set()
        acquired = False

        def sample() -> dict[int, int]:
            live = self.tree.sample(child.pid, tracked)
            tracked.clear()
            tracked.update(live)
            receipt.peak_rss_bytes = max(receipt.peak_rss_bytes, sum(live.values()))
            receipt.peak_processes = max(receipt.peak_processes, len(live))
            return live

        def signal_group(sig: int) -> None:
            try:
                diagnostic = self.signaler.group(child.pid, sig)
            except PermissionError:
                # Also support injected signalers exposing the OS error directly.
                diagnostic = "group_signal_eperm"
            if diagnostic and diagnostic not in receipt.diagnostics:
                receipt.diagnostics.append(diagnostic)

        def stop() -> None:
            # Re-enumerate immediately before both rounds. Keep known PIDs if the
            # probe itself fails so interruption still attempts containment.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    sample()
                except Exception:
                    pass
                signal_group(sig)
                for pid in sorted(tracked):
                    self.signaler.pid(pid, sig)
                if sig == signal.SIGTERM:
                    hard_remaining = start + 1.5 * budget.wall_seconds - self.clock.monotonic()
                    try:
                        self.clock.sleep(max(0, min(5, hard_remaining)))
                    except BaseException:
                        # A second interruption must not bypass the hard kill.
                        try:
                            sample()
                        except Exception:
                            pass
                        signal_group(signal.SIGKILL)
                        for pid in sorted(tracked):
                            self.signaler.pid(pid, signal.SIGKILL)
                        child.wait()
                        raise
            child.wait()

        try:
            if not self.no_lock and not self.lease.held:
                holder = self.lock.acquire(args)
                if holder is not None:
                    receipt.exit_reason = "lock_held"
                    receipt.detail = f"Parapet job lock held by {holder}; budget-run offers an operator --no-lock escape hatch"
                    return CommandResult(tuple(args), EXIT_CODES[receipt.exit_reason], "", receipt.detail)
                acquired = True
            available = self.host.available_bytes()
            required = budget.min_free_bytes + (budget.expected_rss_bytes or 0)
            if available < required:
                receipt.exit_reason = "refused_preflight"
                receipt.detail = (f"available={available} bytes, required={required}; change --expected-rss / "
                                  "PARAPET_BUDGET_EXPECTED_RSS_GB or --min-free / PARAPET_BUDGET_MIN_FREE_GB")
                return CommandResult(tuple(args), EXIT_CODES[receipt.exit_reason], "", receipt.detail)
            env = dict(self.environ)
            env.update({var: str(budget.max_workers) for var in THREAD_VARS})
            # Temporary files avoid pipe deadlock and unbounded Python buffers while polling.
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                child = self.launcher(["nice", "-n", str(budget.nice), *args], cwd=cwd,
                                      env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    while True:
                        returncode = child.poll()
                        live = sample()
                        elapsed = self.clock.monotonic() - start
                        reason = None
                        if sum(live.values()) > budget.max_rss_bytes:
                            reason = "rss_exceeded"
                        elif len(live) > budget.max_processes:
                            reason = "process_cap"
                        elif elapsed >= budget.wall_seconds:
                            reason = "wall_exceeded"
                        elif returncode is not None and not live:
                            child.wait()
                            receipt.exit_reason = "ok" if returncode == 0 else "nonzero"
                            break
                        if reason:
                            receipt.exit_reason = reason
                            key = {"rss_exceeded": "max_rss_bytes", "process_cap": "max_processes",
                                   "wall_exceeded": "wall_seconds"}[reason]
                            receipt.detail = f"{reason}; change {FLAGS[key]} / {ENV_FIELDS[key]}"
                            stop()
                            returncode = EXIT_CODES[reason]
                            break
                        remaining = budget.wall_seconds - elapsed
                        self.clock.sleep(min(budget.poll_seconds, max(0, remaining)))
                except BaseException:
                    receipt.exit_reason = "nonzero"
                    receipt.detail = "Execution interrupted or process probe failed; containment attempted"
                    stop()
                    raise
                stdout.seek(0)
                stderr.seek(0)
                return CommandResult(tuple(args), returncode, stdout.read().decode("utf-8", "replace"),
                                     stderr.read().decode("utf-8", "replace") + receipt.detail)
        except BaseException as exc:
            if not receipt.detail:
                receipt.exit_reason = "nonzero"
                receipt.detail = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.update_job_receipt(receipt, start, usage)
            self.receipts.append(receipt)
            if acquired:
                self.lock.release()


def new_budgeted_executor(budget: ResourceBudget | None = None, *, no_lock: bool = False) -> BudgetedExecutor:
    lock = FileHostLock(Path(os.environ.get("PARAPET_JOB_LOCK", "~/.cache/parapet/job.lock")).expanduser())
    return BudgetedExecutor(budget or resolve_budget()[0], host=SystemHostProbe(),
                            tree=PsProcessTreeProbe(), lock=lock, lease=LockJobLease(lock),
                            clock=SystemClock(), launcher=subprocess.Popen, signaler=PosixSignaler(),
                            environ=dict(os.environ), no_lock=no_lock)


def add_budget_flags(parser: argparse.ArgumentParser, *, allow_no_lock: bool = False) -> None:
    parser.add_argument("--profile")
    for key, flag in FLAGS.items():
        parser.add_argument(flag, dest=key)
    if allow_no_lock:
        parser.add_argument("--no-lock", action="store_true", help="Operator escape hatch; not mechanically read-only; recorded in receipt")
    parser.add_argument("--show-config", action="store_true")


def executor_from_args(args: argparse.Namespace) -> BudgetedExecutor:
    budget, sources = resolve_budget(vars(args))
    if args.show_config:
        print(json.dumps({"budget": budget.model_dump(), "sources": sources}, indent=2))
    return new_budgeted_executor(budget, no_lock=getattr(args, "no_lock", False))


def budget_run(args: argparse.Namespace) -> int:
    executor = executor_from_args(args)
    if args.show_config:
        return 0
    command = args.child_command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise BudgetConfigError("budget-run: missing command; supply a command after --")
    with executor.job(command[0]) as job_receipt:
        result, receipt = executor.run_with_receipt(command)
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        total = job_receipt()
        receipt.cpu_user_s, receipt.cpu_sys_s = total.cpu_user_s, total.cpu_sys_s
        print(receipt.model_dump_json())
        return result.returncode if result.returncode >= 0 else 128 - result.returncode
