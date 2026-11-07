from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
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
        self.assertIn("required_status_checks", commands)
        self.assertIn("waiting for required checks to register", script)
        self.assertNotIn("gh api --method DELETE", commands)
        self.assertNotIn("gh api --method PUT", commands)
        self.assertIn("configure the controlled publisher bypass", script)

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


class ReleaseWorkflowTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (ROOT / ".github" / "workflows" / name).read_text()

    def trigger_block(self, workflow: str) -> str:
        return workflow.split("\npermissions:", 1)[0]

    def python_blocks(self, workflow: str) -> list[str]:
        lines = workflow.splitlines()
        blocks: list[str] = []
        index = 0
        while index < len(lines):
            if "python3 - <<'PY'" not in lines[index]:
                index += 1
                continue
            indent = len(lines[index]) - len(lines[index].lstrip())
            index += 1
            block: list[str] = []
            delimiter = " " * indent + "PY"
            while index < len(lines) and lines[index] != delimiter:
                line = lines[index]
                self.assertTrue(not line or line.startswith(" " * indent))
                block.append(line[indent:] if line else "")
                index += 1
            self.assertLess(index, len(lines), "unterminated Python heredoc")
            blocks.append("\n".join(block) + "\n")
            index += 1
        return blocks

    def run_python_block(
        self,
        block: str,
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", block],
            cwd=cwd,
            env={**os.environ, **(env or {})},
            capture_output=True,
            text=True,
            check=False,
        )

    def write_wheel(self, path: Path, platform_tag: str) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(
                "parapet-0.1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: parapet\nVersion: 0.1.0\n",
            )
            archive.writestr(
                "parapet-0.1.0.dist-info/WHEEL",
                f"Wheel-Version: 1.0\nTag: py3-none-{platform_tag}\n",
            )

    def test_tag_workflow_builds_only_and_cannot_publish(self):
        workflow = self.workflow("py-sdk-wheels.yml")
        self.assertIn('tags: ["v*"]', workflow)
        self.assertIn("actions/upload-artifact@v4", workflow)
        self.assertEqual(workflow.count("args: --release --locked --out dist"), 3)
        self.assertNotIn("PYPI_API_TOKEN", workflow)
        self.assertNotIn("MATURIN_PYPI_TOKEN", workflow)
        self.assertNotIn("command: upload", workflow)
        self.assertNotIn("gh release", workflow)
        self.assertNotIn("--draft=false", workflow)
        self.assertNotIn("environment:", workflow)

    def test_pypi_publish_is_manual_protected_and_bound_to_tag_build(self):
        workflow = self.workflow("py-sdk-publish.yml")
        trigger = self.trigger_block(workflow)
        self.assertIn("workflow_dispatch:", trigger)
        self.assertNotIn("push:", trigger)
        self.assertNotIn("pull_request:", trigger)
        self.assertIn("name: pypi", workflow)
        self.assertIn('test "$GITHUB_REF" = "refs/heads/main"', workflow)
        self.assertIn("PUBLISH PYPI v0.1.0", workflow)
        self.assertIn("actions/workflows/py-sdk-wheels.yml", workflow)
        self.assertIn(".workflow_id", workflow)
        self.assertIn(".head_sha", workflow)
        self.assertIn('= "completed"', workflow)
        self.assertIn('= "success"', workflow)
        self.assertIn("run-id: ${{ inputs.build_run_id }}", workflow)
        self.assertIn("github-token: ${{ github.token }}", workflow)
        self.assertIn("expected 5 wheels", workflow)
        self.assertIn("wheel metadata identity mismatch", workflow)
        self.assertIn("unexpected non-wheel artifact entry", workflow)
        self.assertIn("command: upload", workflow)
        self.assertNotIn("--skip-existing", workflow)
        self.assertNotIn("npm", workflow.lower())
        self.assertNotIn("contents: write", workflow)

    def test_github_release_publish_is_manual_protected_and_draft_only(self):
        workflow = self.workflow("github-release-publish.yml")
        trigger = self.trigger_block(workflow)
        self.assertIn("workflow_dispatch:", trigger)
        self.assertNotIn("push:", trigger)
        self.assertNotIn("pull_request:", trigger)
        self.assertIn("name: github-release", workflow)
        self.assertIn('test "$GITHUB_REF" = "refs/heads/main"', workflow)
        self.assertIn("PUBLISH GITHUB RELEASE v0.1.0", workflow)
        self.assertIn("contents: write", workflow)
        self.assertIn(".targetCommitish", workflow)
        self.assertIn(".isDraft", workflow)
        self.assertIn('= "true"', workflow)
        self.assertIn("parapet-v0.1.0-RELEASE_SHA256SUMS", workflow)
        self.assertIn("parapet-v0.1.0-macos-arm64-evaluator.tar.gz", workflow)
        self.assertIn("parapet-v0.1.0-release-evidence.tar.gz", workflow)
        self.assertIn("checksum manifest does not cover the exact downloadable asset set", workflow)
        self.assertIn("release asset set mismatch", workflow)
        self.assertIn("wheel metadata identity mismatch", workflow)
        self.assertIn('sort_by(.name)', workflow)
        self.assertIn('test "$published_assets" = "$expected_assets"', workflow)
        self.assertIn("--draft=false --latest=false --verify-tag", workflow)
        self.assertNotIn("npm", workflow.lower())

    def test_pypi_artifact_and_wheel_validators_fail_closed(self):
        blocks = self.python_blocks(self.workflow("py-sdk-publish.yml"))
        self.assertEqual(len(blocks), 2)
        artifact_names = [
            "parapet-manylinux_2_17_x86_64-wheels",
            "parapet-manylinux_2_17_aarch64-wheels",
            "parapet-macosx_11_0_arm64-wheels",
            "parapet-macosx_10_12_x86_64-wheels",
            "parapet-win_amd64-wheels",
        ]
        payload = {
            "artifacts": [
                {"name": name, "expired": False, "size_in_bytes": 1}
                for name in artifact_names
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            good = self.run_python_block(
                blocks[0], cwd=root, env={"ARTIFACTS_JSON": json.dumps(payload)}
            )
            self.assertEqual(good.returncode, 0, good.stdout + good.stderr)
            payload["artifacts"].pop()
            missing = self.run_python_block(
                blocks[0], cwd=root, env={"ARTIFACTS_JSON": json.dumps(payload)}
            )
            self.assertNotEqual(missing.returncode, 0)

            dist = root / "dist"
            dist.mkdir()
            platform_tags = [
                "manylinux_2_17_x86_64",
                "manylinux_2_17_aarch64",
                "macosx_11_0_arm64",
                "macosx_10_12_x86_64",
                "win_amd64",
            ]
            for platform_tag in platform_tags:
                wheel = dist / f"parapet-0.1.0-py3-none-{platform_tag}.whl"
                self.write_wheel(wheel, platform_tag)
            valid_wheels = self.run_python_block(blocks[1], cwd=root)
            self.assertEqual(
                valid_wheels.returncode,
                0,
                valid_wheels.stdout + valid_wheels.stderr,
            )
            unexpected_file = dist / "unexpected.txt"
            unexpected_file.write_text("unexpected")
            unexpected = self.run_python_block(blocks[1], cwd=root)
            self.assertNotEqual(unexpected.returncode, 0)
            unexpected_file.unlink()
            unexpected_dir = dist / "unexpected"
            unexpected_dir.mkdir()
            unexpected = self.run_python_block(blocks[1], cwd=root)
            self.assertNotEqual(unexpected.returncode, 0)
            unexpected_dir.rmdir()
            (dist / "parapet-0.1.0-py3-none-win_amd64.whl").rename(
                dist / "parapet-0.2.0-py3-none-win_amd64.whl"
            )
            wrong_version = self.run_python_block(blocks[1], cwd=root)
            self.assertNotEqual(wrong_version.returncode, 0)

    def test_github_release_asset_validators_fail_closed(self):
        blocks = self.python_blocks(self.workflow("github-release-publish.yml"))
        self.assertEqual(len(blocks), 2)
        base_assets = [
            {"name": "one", "size": 1},
            {"name": "two", "size": 2},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = self.run_python_block(
                blocks[0],
                cwd=root,
                env={"RELEASE_JSON": json.dumps({"assets": base_assets})},
            )
            self.assertEqual(metadata.returncode, 0, metadata.stdout + metadata.stderr)
            duplicate = self.run_python_block(
                blocks[0],
                cwd=root,
                env={
                    "RELEASE_JSON": json.dumps(
                        {"assets": [base_assets[0], base_assets[0]]}
                    )
                },
            )
            self.assertNotEqual(duplicate.returncode, 0)

            assets_dir = root / "release-assets"
            assets_dir.mkdir()
            required_names = [
                "parapet-v0.1.0-macos-arm64-evaluator.tar.gz",
                "parapet-v0.1.0-release-evidence.tar.gz",
            ]
            for index, name in enumerate(required_names):
                (assets_dir / name).write_bytes(f"asset-{index}".encode())
            platform_tags = [
                "manylinux_2_17_x86_64",
                "manylinux_2_17_aarch64",
                "macosx_11_0_arm64",
                "macosx_10_12_x86_64",
                "win_amd64",
            ]
            wheel_names = [
                f"parapet-0.1.0-py3-none-{platform_tag}.whl"
                for platform_tag in platform_tags
            ]
            for name, platform_tag in zip(wheel_names, platform_tags):
                self.write_wheel(assets_dir / name, platform_tag)
            names = required_names + wheel_names
            manifest_name = "parapet-v0.1.0-RELEASE_SHA256SUMS"
            manifest = assets_dir / manifest_name

            def write_manifest(asset_names: list[str]) -> None:
                manifest.write_text(
                    "".join(
                        f"{hashlib.sha256((assets_dir / name).read_bytes()).hexdigest()}  {name}\n"
                        for name in asset_names
                    )
                )

            def release_assets() -> list[dict[str, object]]:
                return [
                    {"name": path.name, "size": path.stat().st_size}
                    for path in sorted(assets_dir.iterdir())
                ]

            write_manifest(names)
            current_release_assets = release_assets()
            valid = self.run_python_block(
                blocks[1],
                cwd=root,
                env={"RELEASE_JSON": json.dumps({"assets": current_release_assets})},
            )
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)

            extra_name = "unreviewed-debug.tar.gz"
            (assets_dir / extra_name).write_bytes(b"unexpected")
            write_manifest(names + [extra_name])
            unexpected = self.run_python_block(
                blocks[1],
                cwd=root,
                env={"RELEASE_JSON": json.dumps({"assets": release_assets()})},
            )
            self.assertNotEqual(unexpected.returncode, 0)
            (assets_dir / extra_name).unlink()
            write_manifest(names)
            current_release_assets = release_assets()

            (assets_dir / names[0]).write_bytes(b"tampered")
            tampered = self.run_python_block(
                blocks[1],
                cwd=root,
                env={"RELEASE_JSON": json.dumps({"assets": current_release_assets})},
            )
            self.assertNotEqual(tampered.returncode, 0)


class LanderHarnessTests(unittest.TestCase):
    def write_executable(self, path: Path, body: str) -> None:
        path.write_text(f"#!{sys.executable}\n{body}")
        path.chmod(0o755)

    def run_lander(
        self,
        *,
        push_fails: bool = False,
        locked: bool = False,
        required_api_fails: bool = False,
        required_missing: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log = root / "events.log"
            check_count = root / "check-count"

            self.write_executable(
                fake_bin / "git",
                """import os
import sys
from pathlib import Path

args = sys.argv[1:]
log = Path(os.environ["HARNESS_LOG"])
with log.open("a") as handle:
    handle.write("GIT " + " ".join(args) + "\\n")

if args == ["rev-parse", "--show-toplevel"]:
    print(os.environ["HARNESS_ROOT"])
elif args[:1] == ["config"]:
    values = {
        "core.hooksPath": ".githooks",
        "user.name": "some one",
        "user.email": "someone@example.com",
    }
    print(values.get(args[-1], ""))
elif args == ["status", "--porcelain"]:
    pass
elif args == ["rev-parse", "pr-26"]:
    print("a" * 40)
elif args[:1] == ["push"] and os.environ.get("PUSH_FAIL") == "1":
    raise SystemExit(1)
raise SystemExit(0)
""",
            )
            self.write_executable(
                fake_bin / "gh",
                """import os
import sys
from pathlib import Path

args = sys.argv[1:]
log = Path(os.environ["HARNESS_LOG"])
with log.open("a") as handle:
    handle.write("GH " + " ".join(args) + "\\n")

if args[:2] == ["pr", "view"]:
    query = args[args.index("-q") + 1]
    values = {
        ".baseRefName": "main",
        ".title": "publication controls",
        ".headRefOid": "a" * 40,
    }
    print(values[query])
elif args[:2] == ["repo", "view"]:
    print("Parapet-Tech/parapet")
elif args[:2] == ["pr", "checks"] and "--json" in args:
    counter = Path(os.environ["CHECK_COUNT"])
    count = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(count + 1))
    print("hygiene")
    if count:
        print("Rust tests passed")
elif args[:2] == ["pr", "checks"]:
    pass
elif args[:1] == ["api"]:
    endpoint = next(arg for arg in args[1:] if arg.startswith("repos/"))
    if "required_status_checks" in endpoint:
        if os.environ.get("REQUIRED_API_FAIL") == "1":
            raise SystemExit(1)
        print("hygiene")
        if os.environ.get("REQUIRED_MISSING") != "1":
            print("Rust tests passed")
    elif endpoint.endswith("/protection"):
        print(os.environ["GH_LOCKED"])
raise SystemExit(0)
""",
            )
            self.write_executable(fake_bin / "python3", "raise SystemExit(0)\n")
            self.write_executable(fake_bin / "sleep", "raise SystemExit(0)\n")

            env = {
                **os.environ,
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "HARNESS_LOG": str(log),
                "HARNESS_ROOT": str(root),
                "CHECK_COUNT": str(check_count),
                "CHECK_WAIT_TIMEOUT_SECONDS": "2",
                "PUSH_FAIL": "1" if push_fails else "0",
                "GH_LOCKED": "true" if locked else "false",
                "REQUIRED_API_FAIL": "1" if required_api_fails else "0",
                "REQUIRED_MISSING": "1" if required_missing else "0",
            }
            result = subprocess.run(
                ["/bin/sh", str(ROOT / "scripts" / "merge_pr_local.sh"), "26"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            return result, log.read_text().splitlines()

    def test_lander_waits_for_all_contexts_before_push(self):
        result, events = self.run_lander()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "waiting for required checks to register: Rust tests passed",
            result.stdout,
        )
        self.assertIn("GIT push origin main", events)

    def test_lander_refuses_a_locked_branch_without_mutating_settings(self):
        result, events = self.run_lander(locked=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("GIT push origin main", events)
        self.assertFalse(any("--method" in event for event in events))

    def test_lander_fails_closed_when_required_check_api_fails(self):
        result, events = self.run_lander(required_api_fails=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("GIT push origin main", events)

    def test_lander_fails_closed_when_contract_check_is_not_configured(self):
        result, events = self.run_lander(required_missing=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("GIT push origin main", events)


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
