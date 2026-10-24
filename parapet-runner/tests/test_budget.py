"""Budget behavior with deterministic host, process and clock boundaries."""
import json
import signal
import subprocess
import tempfile
from pathlib import Path

import pytest

from parapet_runner.budget import (
    BudgetedExecutor, BudgetReceipt, LockJobLease, FileHostLock, ResourceBudget,
    PsProcessTreeProbe, THREAD_VARS, quantity, resolve_budget,
)
from parapet_runner.runner import main
from parapet_runner.manifest import RuntimeIdentity


class Clock:
    def __init__(self):
        self.now = 0
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds
    def wait(self, child, seconds):
        self.sleep(seconds)


class Host:
    def __init__(self, free=10**12):
        self.free = free
    def available_bytes(self):
        return self.free
    def cpu_count(self):
        return 4


class Lock:
    def __init__(self, holder=None):
        self.holder = holder
        self.released = False
    def acquire(self, command):
        return self.holder
    def release(self):
        self.released = True


class Tree:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.last = {}
        self.remembered = []
    def sample(self, root, remembered):
        self.remembered.append(set(remembered))
        self.last = next(self.frames, self.last)
        return self.last


class Child:
    pid = 100
    def __init__(self, codes):
        self.codes = iter(codes)
        self.code = None
    def poll(self):
        self.code = next(self.codes, self.code)
        return self.code
    def wait(self):
        return -9


class Launcher:
    def __init__(self, child=None, error=None):
        self.child, self.error = child, error
        self.calls = []
    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if self.error:
            raise self.error
        kwargs['stdout'].write(b'child output')
        return self.child


class Signals:
    def __init__(self):
        self.calls = []
    def group(self, pid, sig):
        self.calls.append(('group', pid, sig))
    def pid(self, pid, sig):
        self.calls.append(('pid', pid, sig))


def executor(*, frames=({},), codes=(0,), host=None, lock=None, launcher=None, **limits):
    lock = lock or Lock()
    return BudgetedExecutor(ResourceBudget(**limits), host=host or Host(),
        tree=Tree(frames), lock=lock, lease=LockJobLease(lock), clock=Clock(),
        launcher=launcher or Launcher(Child(codes)), signaler=Signals(), environ={'EXISTING': 'yes'})


def test_immediate_child_exit_wakes_before_poll_interval():
    e = executor(frames=({100: 1}, {}), codes=(None, 0))
    waits = []
    def wait(child, seconds):
        waits.append((child, seconds))
    def unexpected_sleep(seconds):
        pytest.fail('immediately exiting child must not sleep')
    e.clock.wait = wait
    e.clock.sleep = unexpected_sleep
    result, receipt = e.run_with_receipt(['cmd'])
    assert result.returncode == 0 and receipt.exit_reason == 'ok'
    assert waits == [(e.launcher.child, e.budget.poll_seconds)]
    assert receipt.wall_seconds == 0


def test_exited_child_keeps_sampling_live_descendants():
    e = executor(frames=({101: 1}, {101: 1}, {}), codes=(0,))
    def unexpected_wait(child, seconds):
        pytest.fail('exited child must not cause a busy wait')
    e.clock.wait = unexpected_wait
    result, receipt = e.run_with_receipt(['cmd'])
    assert result.returncode == 0 and receipt.exit_reason == 'ok'
    assert receipt.wall_seconds == 2 * e.budget.poll_seconds


def test_preflight_refuses_and_releases():
    e = executor(host=Host(1))
    result, receipt = e.run_with_receipt(['cmd'])
    assert result.returncode == 123
    assert receipt.exit_reason == 'refused_preflight'
    assert '--expected-rss' in result.stderr and 'PARAPET_BUDGET_MIN_FREE_GB' in result.stderr
    assert not e.launcher.calls and e.lock.released


def test_lock_refusal_names_holder_and_override():
    e = executor(lock=Lock('PID 42 command train'))
    result = e.run(['cmd'])
    assert result.returncode == 124 and 'PID 42' in result.stderr and '--no-lock' in result.stderr
    assert not e.launcher.calls and not e.lock.released


@pytest.mark.parametrize('code,reason', [(0, 'ok'), (7, 'nonzero')])
def test_clean_exit_and_env(code, reason):
    e = executor(codes=(code,), max_workers=3)
    result, receipt = e.run_with_receipt(['cmd'])
    assert (result.returncode, receipt.exit_reason, result.stdout) == (code, reason, 'child output')
    args, kwargs = e.launcher.calls[0]
    assert args[:3] == ['nice', '-n', '10'] and kwargs['start_new_session']
    assert all(kwargs['env'][var] == '3' for var in THREAD_VARS)
    assert kwargs['env']['EXISTING'] == 'yes' and e.lock.released


@pytest.mark.parametrize('limits,frames,reason,code', [
    ({'max_rss_bytes': 64}, ({100: 150},), 'rss_exceeded', 120),
    ({'max_processes': 1}, ({100: 10, 101: 10},), 'process_cap', 121),
    ({'wall_seconds': 2, 'poll_seconds': 1}, ({100: 10},), 'wall_exceeded', 122),
])
def test_breaches_signal_group_and_descendants(limits, frames, reason, code):
    e = executor(frames=frames, codes=(None,), **limits)
    result, receipt = e.run_with_receipt(['cmd'])
    assert (result.returncode, receipt.exit_reason) == (code, reason)
    assert ('group', 100, signal.SIGTERM) in e.signaler.calls
    assert ('group', 100, signal.SIGKILL) in e.signaler.calls
    assert ('pid', 100, signal.SIGKILL) in e.signaler.calls
    assert receipt.peak_processes == len(frames[0]) and e.lock.released
    assert '--' in result.stderr


def test_remember_reparented_descendant_and_discover_before_kill_rounds():
    e = executor(frames=({100: 1}, {100: 1, 101: 10}, {101: 10, 102: 1}, {102: 1, 103: 1}),
                 codes=(None,), max_rss_bytes=5, poll_seconds=1)
    e.run(['cmd'])
    assert e.tree.remembered == [set(), {100}, {100, 101}, {101, 102}]
    assert ('pid', 102, signal.SIGTERM) in e.signaler.calls
    assert ('pid', 103, signal.SIGKILL) in e.signaler.calls


def test_ps_union_and_remembered_reparent():
    raw = '100 9 100 4 S\n101 100 101 5 S\n102 101 102 6 S\n103 1 103 7 S\n104 1 100 8 S\n105 100 100 9 Z\n106 100 100 9 SZ+\n'
    assert PsProcessTreeProbe.parse(raw, 100, {103, 999}) == {
        100: 4096, 101: 5120, 102: 6144, 103: 7168, 104: 8192}


@pytest.mark.parametrize('error', [RuntimeError('launch failed'), KeyboardInterrupt()])
def test_lock_released_on_launch_error_or_interrupt(error):
    e = executor(launcher=Launcher(error=error))
    with pytest.raises(type(error)):
        e.run(['cmd'])
    assert e.lock.released and e.receipts[-1].exit_reason == 'nonzero'


def test_interrupted_parent_kills_before_unlock():
    e = executor(frames=({100: 1, 101: 1},), codes=(None,))
    def interrupted(seconds):
        if seconds != 5:
            raise KeyboardInterrupt()
    e.clock.sleep = interrupted
    with pytest.raises(KeyboardInterrupt):
        e.run(['cmd'])
    assert ('pid', 101, signal.SIGKILL) in e.signaler.calls
    assert e.lock.released


def test_real_file_lock_reports_holder_and_can_reacquire():
    with tempfile.TemporaryDirectory() as d:
        a, b = FileHostLock(Path(d)/'job.lock'), FileHostLock(Path(d)/'job.lock')
        try:
            assert a.acquire(['train']) is None
            assert 'train' in b.acquire(['other'])
        finally:
            a.release()
        assert b.acquire(['other']) is None
        b.release()


def test_no_lock_recorded():
    e = executor()
    e.no_lock = True
    e.run(['cmd'])
    assert e.receipts[-1].no_lock and not e.lock.released


@pytest.mark.parametrize('raw,result,duration', [('12G', 12*1024**3, False), ('512M', 512*1024**2, False),
                                                ('90m', 5400, True), ('4h', 14400, True)])
def test_units(raw, result, duration):
    assert quantity(raw, duration=duration) == result


def test_config_precedence_and_sources():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'budget.yaml'
        path.write_text('defaults:\n  max_workers: 1\n  max_rss_bytes: 1G\nprofiles:\n  train:\n    max_workers: 2\n    wall_seconds: 90m\n')
        env = {'PARAPET_BUDGET_CONFIG': str(path), 'PARAPET_BUDGET_PROFILE': 'train',
               'PARAPET_BUDGET_MAX_WORKERS': '3', 'PARAPET_BUDGET_MAX_RSS_GB': '2'}
        budget, source = resolve_budget({'max_workers': '4'}, env=env)
        assert budget.max_workers == 4 and source['max_workers'] == 'CLI flags'
        assert budget.max_rss_bytes == 2*1024**3 and source['max_rss_bytes'].endswith('_GB')
        assert budget.wall_seconds == 5400 and source['wall_seconds'].startswith('profile train')
        assert source['nice'] == 'built-in default'
        assert resolve_budget(env=env)[0].max_workers == 3
        del env['PARAPET_BUDGET_MAX_WORKERS']
        assert resolve_budget(env=env)[0].max_workers == 2
        del env['PARAPET_BUDGET_PROFILE']
        assert resolve_budget(env=env)[0].max_workers == 1


@pytest.mark.parametrize('env,source', [({'PARAPET_BUDGET_MAX_WORKERS': '0'}, 'PARAPET_BUDGET_MAX_WORKERS'),
    ({'PARAPET_BUDGET_WALL': 'nan'}, 'PARAPET_BUDGET_WALL'),
    ({'PARAPET_BUDGET_MAX_RSS_GB': 'wrong'}, 'PARAPET_BUDGET_MAX_RSS_GB'),
    ({'PARAPET_BUDGET_PROFILE': 'absent'}, 'PARAPET_BUDGET_PROFILE')])
def test_bad_configuration_names_source(env, source):
    with pytest.raises(ValueError, match=source):
        resolve_budget(env=env)


def test_show_config_no_child(capsys, monkeypatch):
    monkeypatch.delenv('PARAPET_BUDGET_CONFIG', raising=False)
    assert main(['budget-run', '--max-rss', '512M', '--wall', '90m', '--show-config']) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved['budget']['max_rss_bytes'] == 512*1024**2
    assert resolved['budget']['wall_seconds'] == 5400
    assert resolved['sources']['wall_seconds'] == 'CLI flags'


def test_receipts_roundtrip_runtime_identity():
    receipt = BudgetReceipt(budget=ResourceBudget(), exit_reason='rss_exceeded', no_lock=True)
    identity = RuntimeIdentity(git_sha='a', trainer_script_hash='b', parapet_eval_hash='c',
                               pg2_model_id='d', eval_config_hash='e', env_hash='f', budget_receipts=[receipt])
    assert RuntimeIdentity.model_validate_json(identity.model_dump_json()).budget_receipts == [receipt]
    old = identity.model_dump()
    del old['budget_receipts']
    assert RuntimeIdentity(**old).budget_receipts is None


def test_no_unbounded_jobs_in_tracked_python():
    root = Path(__file__).resolve().parents[2]
    tracked = subprocess.check_output(['git', 'ls-files', 'scripts/*.py', 'parapet-runner/**/*.py'],
                                      cwd=root, text=True).splitlines()
    pattern = __import__('re').compile(r'n_jobs\s*=\s*-1\b')
    offenders = [name for name in tracked if pattern.search((root/name).read_text())]
    assert not offenders, offenders


def test_probe_failure_cleans_up_and_releases():
    e = executor(codes=(None,))
    def fail(root, remembered):
        raise PermissionError('ps denied')
    e.tree.sample = fail
    with pytest.raises(PermissionError, match='ps denied'):
        e.run(['cmd'])
    assert ('group', 100, signal.SIGKILL) in e.signaler.calls and e.lock.released


def test_launch_no_io_in_constructor():
    launch = Launcher(error=AssertionError('must not launch'))
    e = executor(launcher=launch)
    assert e.receipts == [] and launch.calls == []


def test_late_descendant_after_root_exits_is_still_enforced():
    e = executor(frames=({101: 10}, {101: 10}, {}, {}), codes=(0,), max_rss_bytes=5)
    _, receipt = e.run_with_receipt(['cmd'])
    assert receipt.exit_reason == 'rss_exceeded'
    assert ('pid', 101, signal.SIGTERM) in e.signaler.calls


def test_five_second_grace_and_wall_hard_deadline():
    e = executor(frames=({100: 10},), codes=(None,), max_rss_bytes=5)
    e.run(['cmd'])
    assert e.clock.now == 5
    e = executor(frames=({100: 10},), codes=(None,), wall_seconds=2, poll_seconds=30)
    e.run(['cmd'])
    assert e.clock.now == 3


def test_invalid_config_file_and_profile_sources():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'budget.yaml'
        env = {'PARAPET_BUDGET_CONFIG': str(path)}
        with pytest.raises(ValueError, match='PARAPET_BUDGET_CONFIG'):
            resolve_budget(env=env)
        for payload in ('defaults:\n  max_workers: null', 'defaults:\n  nope: 1', 'defaults: []'):
            path.write_text(payload)
            with pytest.raises(ValueError, match='config'):
                resolve_budget(env=env)
        path.write_text('profiles:\n  train:\n    nice: 25')
        with pytest.raises(ValueError, match='profile train'):
            resolve_budget({'profile': 'train'}, env=env)
        with pytest.raises(ValueError, match='CLI flags'):
            resolve_budget({'max_workers': '1.5'}, env=env)


def test_system_memory_probe_paths(monkeypatch):
    from parapet_runner import budget as module
    monkeypatch.setattr(module.sys, 'platform', 'darwin')
    monkeypatch.setattr(module.subprocess, 'check_output', lambda *a, **kw:
        'Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 2.\nPages inactive: 3.\nPages speculative: 4.\nPages purgeable: 5.\n')
    assert module.SystemHostProbe().available_bytes() == 14*16384
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setattr(Path, 'read_text', lambda *a, **kw: 'MemTotal: 999 kB\nMemAvailable: 123 kB\n')
    assert module.SystemHostProbe().available_bytes() == 123*1024


@pytest.mark.parametrize('interrupted', [False, True])
def test_group_permission_error_keeps_per_pid_containment(interrupted):
    class DeniedGroups(Signals):
        def group(self, pid, sig):
            super().group(pid, sig)
            raise PermissionError(1, 'group denied')

    e = executor(frames=({100: 10, 101: 10},), codes=(None,), max_rss_bytes=5)
    e.signaler = DeniedGroups()
    if interrupted:
        def interrupt(seconds):
            raise KeyboardInterrupt()
        e.clock.sleep = interrupt
        with pytest.raises(KeyboardInterrupt):
            e.run(['cmd'])
        receipt = e.receipts[-1]
    else:
        result, receipt = e.run_with_receipt(['cmd'])
        assert (result.returncode, receipt.exit_reason) == (120, 'rss_exceeded')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in (100, 101):
            assert ('pid', pid, sig) in e.signaler.calls
    assert receipt.diagnostics == ['group_signal_eperm']
    assert e.lock.released


def test_posix_group_permission_error_is_diagnostic(monkeypatch):
    from parapet_runner.budget import PosixSignaler
    def denied(*args):
        raise PermissionError(1, 'group denied')
    monkeypatch.setattr('os.killpg', denied)
    assert PosixSignaler().group(100, signal.SIGTERM) == 'group_signal_eperm'


def test_default_preflight_allows_trivial_job_at_16_gib():
    e = executor(host=Host(16*1024**3))
    assert e.budget.min_free_bytes == 4*1024**3
    assert e.budget.expected_rss_bytes is None
    assert e.run(['cmd']).returncode == 0
    assert e.launcher.calls


@pytest.mark.parametrize('available,code', [(10*1024**3-1, 123), (10*1024**3, 0)])
def test_expected_rss_preflight_boundary(available, code):
    e = executor(host=Host(available), expected_rss_bytes=6*1024**3)
    assert e.run(['cmd']).returncode == code
    assert bool(e.launcher.calls) == (code == 0)


def test_expected_rss_sources():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'budget.yaml'
        path.write_text('defaults: {expected_rss_bytes: 1G}\nprofiles: {train: {expected_rss_bytes: 2G}}')
        env = {'PARAPET_BUDGET_CONFIG': str(path)}
        assert resolve_budget(env=env)[0].expected_rss_bytes == 1024**3
        assert resolve_budget({'profile': 'train'}, env=env)[0].expected_rss_bytes == 2*1024**3
        env['PARAPET_BUDGET_EXPECTED_RSS_GB'] = '3'
        assert resolve_budget({'profile': 'train'}, env=env)[0].expected_rss_bytes == 3*1024**3
        assert resolve_budget({'expected_rss_bytes': '4G'}, env=env)[0].expected_rss_bytes == 4*1024**3


@pytest.mark.parametrize('flags,payload,env_extra,source', [
    (['--profile', 'nope'], '{}', {}, '--profile'),
    (['--wall', '3weeks'], '{}', {}, 'CLI flags'),
    ([], 'defaults: [', {}, 'PARAPET_BUDGET_CONFIG'),
    ([], 'false', {}, 'PARAPET_BUDGET_CONFIG'),
    ([], 'profiles: []', {}, 'PARAPET_BUDGET_CONFIG'),
    (['--max-workers', '-1'], '{}', {}, 'CLI flags'),
    (['--max-rss', '0'], '{}', {}, 'CLI flags'),
    (['--expected-rss', '0'], '{}', {}, 'CLI flags'),
    ([], 'defaults: {max_workers: 0}', {}, 'config'),
    ([], '{}', {'PARAPET_BUDGET_EXPECTED_RSS_GB': '-2'}, 'PARAPET_BUDGET_EXPECTED_RSS_GB'),
])
def test_cli_config_errors_are_one_line_exit_two(flags, payload, env_extra, source, monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/'budget.yaml'
        path.write_text(payload)
        monkeypatch.setenv('PARAPET_BUDGET_CONFIG', str(path))
        for key, value in env_extra.items():
            monkeypatch.setenv(key, value)
        assert main(['budget-run', *flags, '--show-config']) == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        assert source in captured.err and 'Traceback' not in captured.err
        assert len(captured.err.splitlines()) == 1 and captured.err.endswith('\n')
        assert any(word in captured.err for word in ('set ', 'fix ', 'select '))


@pytest.mark.parametrize('available,code', [(1, 123), (16*1024**3, 0)])
def test_cli_receipt_stdout_and_diagnostic_stderr(available, code, monkeypatch, capsys):
    e = executor(host=Host(available))
    monkeypatch.setattr('parapet_runner.budget.executor_from_args', lambda args: e)
    assert main(['budget-run', '--', 'cmd']) == code
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    receipt = json.loads(lines[-1])
    assert receipt['exit_reason'] == ('ok' if code == 0 else 'refused_preflight')
    assert captured.out.endswith('\n')
    if code:
        assert len(lines) == 1 and captured.err.endswith('\n')
        assert 'available=' in captured.err
    else:
        assert lines[0] == 'child output' and captured.err == ''


def test_job_instances_hold_real_lease_across_three_commands():
    from parapet_runner.budget import JobLockHeld
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'job.lock'
        first = executor(lock=FileHostLock(path))
        second = executor(lock=FileHostLock(path))
        with first.job('pipeline'):
            for _ in range(3):
                assert first.run(['phase']).returncode == 0
                with pytest.raises(JobLockHeld):
                    with second.job('other'):
                        second.run(['must-not-launch'])
                assert not second.launcher.calls
        with second.job('other'):
            assert second.run(['allowed']).returncode == 0


def test_lock_metadata_never_discloses_argv(capsys, monkeypatch):
    import stat
    from parapet_runner.budget import budget_run
    from argparse import Namespace
    secret = 'SENTINEL_PRIVATE_ARGUMENT_87325'
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'job.lock'
        first = executor(lock=FileHostLock(path))
        second = executor(lock=FileHostLock(path))
        with first.job('/bin/holder'):
            assert first.run(['phase', secret]).returncode == 0
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert json.loads(path.read_text()) == {'pid': __import__('os').getpid(), 'executable': 'holder'}
            result, receipt = second.run_with_receipt(['other', secret])
            assert result.returncode == 124
            monkeypatch.setattr('parapet_runner.budget.executor_from_args', lambda args: second)
            from parapet_runner.budget import JobLockHeld
            with pytest.raises(JobLockHeld) as error:
                budget_run(Namespace(show_config=False, child_command=['other', secret]))
            assert secret not in str(error.value)
            print(result.stderr, file=__import__('sys').stderr)
            assert secret not in receipt.model_dump_json()
            assert secret not in path.read_text()
            assert secret not in capsys.readouterr().err


@pytest.mark.parametrize('command', ['run', 'eval'])
def test_mutating_commands_reject_no_lock(command, capsys):
    from parapet_runner.runner import _build_parser
    parser = _build_parser()
    # Supply required arguments so refusal specifically proves the bypass is absent.
    args = ([ '--curation-manifest', 'c', '--train-config', 't', '--output-dir', 'o']
            if command == 'run' else ['--dataset', 'd', '--output-dir', 'o'])
    with pytest.raises(SystemExit) as error:
        parser.parse_args([command, *args, '--no-lock'])
    assert error.value.code == 2
    assert 'unrecognized arguments: --no-lock' in capsys.readouterr().err


def test_short_cpu_child_accounted_in_job_receipt():
    import sys
    from parapet_runner.budget import SystemClock
    e = executor(poll_seconds=0.01)
    e.clock = SystemClock()
    e.launcher = subprocess.Popen
    with e.job('cpu-test') as snapshot:
        result = e.run([sys.executable, '-c',
                        'import time; end=time.process_time()+0.06\nwhile time.process_time()<end: pass'])
        receipt = snapshot()
        assert result.returncode == 0
        assert receipt.cpu_user_s + receipt.cpu_sys_s >= 0.04
        assert receipt.cpu_user_s >= 0 and receipt.cpu_sys_s >= 0

@pytest.mark.parametrize('subcmd', ['budget-run', 'run', 'eval'])
def test_show_config_across_all_budgeted_commands(subcmd, capsys, monkeypatch):
    monkeypatch.delenv('PARAPET_BUDGET_CONFIG', raising=False)
    assert main([subcmd, '--max-rss', '512M', '--max-workers', '3', '--show-config']) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved['budget']['max_rss_bytes'] == 512 * 1024**2
    assert resolved['budget']['max_workers'] == 3
    assert resolved['sources']['max_workers'] == 'CLI flags'


@pytest.mark.parametrize('subcmd,flag', [('run', '--curation-manifest'), ('eval', '--dataset')])
def test_missing_required_execution_flags_exit_two_without_acquiring_lock(subcmd, flag, capsys, monkeypatch):
    e = executor()
    monkeypatch.setattr('parapet_runner.runner.executor_from_args', lambda args: e)
    assert main([subcmd]) == 2
    err = capsys.readouterr().err
    assert flag in err and '--show-config' in err
    assert not e.lease.held and not e.lock.released


def test_budget_run_signal_exit_preserves_128_plus_signal(monkeypatch, capsys):
    e = executor(codes=(-9,))
    monkeypatch.setattr('parapet_runner.budget.executor_from_args', lambda args: e)
    assert main(['budget-run', '--', 'cmd']) == 137
    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt['exit_reason'] == 'nonzero'


def test_cli_eval_writes_budget_receipt_and_holds_lease(monkeypatch):
    import yaml
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        dataset = base / 'dataset'
        dataset.mkdir()
        (dataset / 'split.yaml').write_text(
            yaml.safe_dump([{'id': '1', 'label': 'malicious', 'content': 'attack text'}]),
            encoding='utf-8',
        )
        eval_cfg = base / 'eval_config.yaml'
        eval_cfg.write_text(yaml.safe_dump({'layers': {'L1': {'mode': 'block', 'threshold': 0.0}}}), encoding='utf-8')
        out_dir = base / 'out'
        lock_path = base / 'job.lock'
        e = executor(lock=FileHostLock(lock_path))
        other = executor(lock=FileHostLock(lock_path))

        def run_eval(args, *, cwd=None):
            # Verify lease is held while child runs and blocks second job
            assert e.lease.held
            assert other.run(['contending']).returncode == 124
            out_idx = list(args).index('--output') + 1
            Path(args[out_idx]).write_text(json.dumps({
                'layers': [{'layer': 'l1', 'tp': 1, 'fp': 0, 'fn_count': 0, 'tn': 0,
                            'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'total': 1}]
            }), encoding='utf-8')
            e.receipts.append(BudgetReceipt(budget=e.budget, peak_rss_bytes=4096, peak_processes=1))
            from parapet_runner.baseline import CommandResult
            return CommandResult(tuple(args), 0, '', '')

        e.run = run_eval
        monkeypatch.setattr('parapet_runner.runner.executor_from_args', lambda args: e)
        rc = main([
            'eval',
            '--dataset', str(dataset),
            '--output-dir', str(out_dir),
            '--workspace-root', str(base),
            '--eval-config', str(eval_cfg),
            '--parapet-eval-bin', str(base / 'parapet-eval'),
        ])
        assert rc == 0
        assert not e.lease.held
        receipt_path = out_dir / 'budget_receipt.json'
        assert receipt_path.exists()
        receipt = BudgetReceipt.model_validate_json(receipt_path.read_text(encoding='utf-8'))
        assert receipt.exit_reason == 'ok'
