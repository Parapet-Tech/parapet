from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_publisher_context", ROOT / "scripts" / "check_publisher_context.py"
)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publisher
SPEC.loader.exec_module(publisher)


class FakeRunner:
    def __init__(self, greeting: str):
        self.greeting = greeting
        self.calls: list[list[str]] = []

    def run(self, args):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args, 1, "", self.greeting)


def good_context():
    return publisher.PublisherContext(
        hooks_path=".githooks",
        identity="some one <someone@example.com>",
        remote_url="git@github.com:Parapet-Tech/parapet.git",
        ssh_command="ssh -i ~/.ssh/parapet_publisher -o IdentitiesOnly=yes",
    )


class PublisherContextTests(unittest.TestCase):
    def test_valid_context_and_actor_pass(self):
        problems, probe = publisher.validate_context(good_context())
        self.assertEqual(problems, [])
        self.assertIsNotNone(probe)
        runner = FakeRunner("Hi parapet-worker! You've successfully authenticated")
        self.assertIsNone(publisher.verify_ssh_actor(probe, runner))
        self.assertEqual(Path(runner.calls[0][3]).name, "parapet_publisher")

    def test_personal_or_wrong_key_context_fails_closed(self):
        context = publisher.PublisherContext(
            hooks_path="",
            identity="Other <other@example.com>",
            remote_url="https://github.com/Parapet-Tech/parapet.git",
            ssh_command="ssh -i ~/.ssh/id_ed25519",
        )
        problems, probe = publisher.validate_context(context)
        self.assertIsNone(probe)
        self.assertGreaterEqual(len(problems), 5)

    def test_wrong_authenticated_actor_fails(self):
        problems, probe = publisher.validate_context(good_context())
        self.assertEqual(problems, [])
        self.assertEqual(
            publisher.verify_ssh_actor(probe, FakeRunner("Hi a-personal-account!")),
            "publisher SSH key did not authenticate as parapet-worker",
        )

    def test_transport_and_command_config_overrides_fail_closed(self):
        for name in publisher.FORBIDDEN_GIT_OVERRIDES:
            with self.subTest(name=name):
                context = publisher.PublisherContext(
                    **{**good_context().__dict__, "active_overrides": (name,)}
                )
                runner = FakeRunner(
                    "Hi parapet-worker! You've successfully authenticated"
                )
                problems = publisher.check_context(context, runner)
                self.assertTrue(
                    any(name in problem for problem in problems), problems
                )
                self.assertEqual(runner.calls, [])

    def test_ssh_command_rejects_extra_options(self):
        context = publisher.PublisherContext(
            **{
                **good_context().__dict__,
                "ssh_command": (
                    "ssh -i ~/.ssh/parapet_publisher -o IdentitiesOnly=yes "
                    "-o ProxyCommand=unexpected"
                ),
            }
        )
        problems, _ = publisher.validate_context(context)
        self.assertIn(
            "core.sshCommand contains unsupported publisher options", problems
        )


class LandingScriptTests(unittest.TestCase):
    def test_local_lander_uses_exact_fast_forward_and_publisher_check(self):
        script = (ROOT / "scripts" / "merge_pr_local.sh").read_text()
        commands = "\n".join(
            line for line in script.splitlines() if not line.lstrip().startswith("#")
        )
        self.assertNotIn("gh pr merge", commands)
        self.assertNotIn("git merge --no-ff", commands)
        self.assertIn("git merge --ff-only", commands)
        self.assertIn("check_publisher_context.py", script)
        self.assertIn("gh pr checks", commands)

    def test_pre_push_hook_verifies_publisher_context(self):
        hook = (ROOT / ".githooks" / "pre-push").read_text()
        self.assertIn("check_publisher_context.py", hook)

    def test_identity_workflow_has_targeted_and_full_audit_paths(self):
        workflow = (ROOT / ".github" / "workflows" / "identity-guard.yml").read_text()
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("schedule:", workflow)
        self.assertIn("--full-blob-scan", workflow)
        self.assertIn("+refs/heads/*:refs/remotes/origin/*", workflow)
        self.assertIn("Test publication controls", workflow)


class HygieneRangeTests(unittest.TestCase):
    def run_git(
        self, repo: Path, *args: str, env=None
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def commit(self, repo: Path, message: str, name: str, email: str) -> str:
        marker = repo / "marker.txt"
        marker.write_text(message + "\n")
        self.run_git(repo, "add", "marker.txt")
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
        self.run_git(
            repo,
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            message,
            env=env,
        )
        return self.run_git(repo, "rev-parse", "HEAD").stdout.strip()

    def hygiene(self, repo: Path, revision_range: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_public_hygiene.py"),
                revision_range,
            ],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_new_branch_and_push_ranges_accept_clean_and_reject_bad_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            self.run_git(repo, "init", "-b", "main")
            base = self.commit(
                repo, "base", "some one", "someone@example.com"
            )
            self.run_git(repo, "switch", "-c", "feature")
            clean = self.commit(
                repo, "clean feature", "some one", "someone@example.com"
            )

            merge_base = self.run_git(
                repo, "merge-base", "main", clean
            ).stdout.strip()
            new_branch_clean = f"{merge_base}..{clean}"
            push_clean = f"{base}..{clean}"
            self.assertEqual(self.hygiene(repo, new_branch_clean).returncode, 0)
            self.assertEqual(self.hygiene(repo, push_clean).returncode, 0)

            dirty = self.commit(repo, "bad identity", "Other", "other@example.com")
            new_branch_dirty = f"{merge_base}..{dirty}"
            push_dirty = f"{clean}..{dirty}"
            for revision_range in (new_branch_dirty, push_dirty):
                result = self.hygiene(repo, revision_range)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("author is 'Other <other@example.com>'", result.stdout)


if __name__ == "__main__":
    unittest.main()
