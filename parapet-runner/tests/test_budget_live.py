"""Small real children; serial tests requiring POSIX ps inspection."""
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import pytest
from parapet_runner.budget import CONTAINMENT_LIMIT, ResourceBudget, new_budgeted_executor
pytestmark = pytest.mark.live_process

@pytest.fixture(autouse=True)
def require_ps():
    try:
        subprocess.check_output(['ps', '-axo', 'pid=,ppid=,pgid=,rss=,stat='], stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f'ps cannot be executed in this environment: {exc}')

def run_child(script, **limits):
    with tempfile.TemporaryDirectory() as d:
        pidfile = Path(d)/'pids'
        values = dict(max_rss_bytes=190*1024**2, min_free_bytes=0, poll_seconds=.1, wall_seconds=4)
        values.update(limits)
        executor = new_budgeted_executor(ResourceBudget(**values), no_lock=True)
        try:
            result, receipt = executor.run_with_receipt([sys.executable, '-c', script, str(pidfile)])
            pids = list(map(int, pidfile.read_text().split())) if pidfile.exists() else []
            output = subprocess.check_output(['ps', '-axo', 'pid=,stat='], text=True)
            living = {int(row.split()[0]) for row in output.splitlines() if 'Z' not in row.split()[1]}
            assert not living.intersection(pids), f'surviving children: {living.intersection(pids)}'
            return result, receipt
        finally:
            if pidfile.exists():
                for pid in map(int, pidfile.read_text().split()):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

def test_live_rss_exceeded():
    result, receipt = run_child('import time; data=bytearray(150*1024**2); time.sleep(10)', max_rss_bytes=64*1024**2)
    assert receipt.exit_reason == 'rss_exceeded' and result.returncode == 120

def test_live_process_cap():
    _, receipt = run_child('''import os, sys, time
from pathlib import Path
pids=[]
for _ in range(5):
    pid=os.fork()
    if pid == 0:
        time.sleep(10)
        os._exit(0)
    pids.append(pid)
Path(sys.argv[1]).write_text(' '.join(map(str,pids)))
time.sleep(10)
''', max_processes=3)
    assert receipt.exit_reason == 'process_cap'

def test_live_wall_exceeded():
    _, receipt = run_child('import time; time.sleep(10)', wall_seconds=2)
    assert receipt.exit_reason == 'wall_exceeded'

def test_live_setsid_descendant_accounted_and_killed():
    _, receipt = run_child('''import os, sys, time
from pathlib import Path
pid=os.fork()
if pid == 0:
    os.setsid()
    time.sleep(10)
    os._exit(0)
Path(sys.argv[1]).write_text(str(pid))
time.sleep(10)
''', max_processes=1)
    assert receipt.exit_reason == 'process_cap' and receipt.peak_processes >= 2

@pytest.mark.xfail(strict=False, reason=CONTAINMENT_LIMIT)
def test_live_double_fork_reparented_between_polls():
    _, receipt = run_child('''import os, sys, time
from pathlib import Path
pid=os.fork()
if pid == 0:
    os.setsid()
    child=os.fork()
    if child:
        os._exit(0)
    Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(10)
    os._exit(0)
os.waitpid(pid,0)
time.sleep(10)
''')
    assert receipt.exit_reason == 'wall_exceeded'

@pytest.mark.parametrize('code', [0, 7])
def test_live_exact_child_exit_and_environment(code):
    result, receipt = run_child(f'import os; assert os.environ["PARAPET_MAX_WORKERS"] == "2"; raise SystemExit({code})')
    assert result.returncode == code and receipt.exit_reason == ('ok' if code == 0 else 'nonzero')


def test_live_echo_finishes_before_default_poll_interval():
    executor = new_budgeted_executor(ResourceBudget(), no_lock=True)
    start = time.monotonic()
    result, receipt = executor.run_with_receipt(['/bin/echo', 'hi'])
    elapsed = time.monotonic() - start
    assert result.returncode == 0 and result.stdout == 'hi\n'
    assert receipt.exit_reason == 'ok'
    assert elapsed < 0.5
