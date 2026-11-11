from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import re
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
    WORKFLOWS = (
        "py-sdk-wheels.yml",
        "py-sdk-publish.yml",
        "github-release-publish.yml",
    )

    def workflow(self, name: str) -> str:
        return (ROOT / ".github" / "workflows" / name).read_text()

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
                "parapet-0.2.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: parapet\nVersion: 0.2.0\n",
            )
            archive.writestr(
                "parapet-0.2.0.dist-info/WHEEL",
                f"Wheel-Version: 1.0\nTag: py3-none-{platform_tag}\n",
            )

    def wheel_set_sha256(self, paths: list[Path]) -> str:
        lines = [
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            for path in paths
        ]
        return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode()).hexdigest()

    def test_release_versions_are_coherent(self):
        import tomllib
        pyproject = tomllib.loads((ROOT / "parapet-py/pyproject.toml").read_text())
        cargo_toml = tomllib.loads((ROOT / "parapet/Cargo.toml").read_text())
        cargo_lock = tomllib.loads((ROOT / "parapet/Cargo.lock").read_text())
        package = json.loads((ROOT / "parapet-ts/package.json").read_text())
        package_lock = json.loads((ROOT / "parapet-ts/package-lock.json").read_text())
        self.assertEqual(pyproject["project"]["version"], "0.2.0")
        self.assertEqual(cargo_toml["package"]["version"], "0.2.0")
        roots = [p for p in cargo_lock["package"] if p["name"] == "parapet"]
        self.assertEqual([p["version"] for p in roots], ["0.2.0"])
        self.assertEqual(package["version"], "0.2.0")
        self.assertEqual(package_lock["version"], "0.2.0")
        self.assertEqual(package_lock["packages"][""]["version"], "0.2.0")
        metadata = subprocess.run(
            ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
            cwd=ROOT / "parapet",
            check=True,
            capture_output=True,
            text=True,
        )
        packages = json.loads(metadata.stdout)["packages"]
        self.assertEqual(
            [p["version"] for p in packages if p["name"] == "parapet"],
            ["0.2.0"],
        )

    def test_workflows_are_pinned_attempt_one_and_secret_free(self):
        for name in self.WORKFLOWS:
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                uses = re.findall(
                    r"^\s*(?:-\s*)?uses:\s*(\S+)", workflow, re.MULTILINE
                )
                self.assertTrue(uses)
                for action in uses:
                    self.assertRegex(action, r"^[^@]+@[0-9a-f]{40}$")
                self.assertNotIn("secrets.", workflow)
                self.assertNotIn("--skip-existing", workflow)
                self.assertTrue(
                    "GITHUB_RUN_ATTEMPT" in workflow or "RUN_ATTEMPT:" in workflow
                )
                self.assertIn('= "1"', workflow)

    def test_publisher_workflow_content_rereads_execute_with_gh_jq(self):
        payload = b"reviewed workflow bytes\n"
        encoded = base64.b64encode(payload).decode()
        expected = hashlib.sha256(payload).hexdigest()
        for name in ("py-sdk-publish.yml", "github-release-publish.yml"):
            with (
                self.subTest(workflow=name),
                tempfile.TemporaryDirectory() as temp_dir,
            ):
                workflow = self.workflow(name)
                self.assertNotIn("--jq -r .content", workflow)
                line = next(
                    line.strip()
                    for line in workflow.splitlines()
                    if "/contents/.github/workflows/" in line
                )
                root = Path(temp_dir)
                fake_bin = root / "bin"
                fake_bin.mkdir()
                gh = fake_bin / "gh"
                gh.write_text(
                    f"#!{sys.executable}\n"
                    "import os, sys\n"
                    "args = sys.argv[1:]\n"
                    "if len(args) != 4 or args[0] != 'api' or args[2:] != ['--jq', '.content']:\n"
                    "    raise SystemExit(f'unexpected gh arguments: {args!r}')\n"
                    "print(os.environ['CONTENT_B64'])\n"
                )
                gh.chmod(0o755)
                result = subprocess.run(
                    ["bash", "-c", "set -euo pipefail\n" + line],
                    cwd=root,
                    env={
                        **os.environ,
                        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                        "CONTENT_B64": encoded,
                        "CONTROL_COMMIT": "a" * 40,
                        "GITHUB_REPOSITORY": "Parapet-Tech/parapet",
                        "EXPECTED_WORKFLOW_SHA256": expected,
                    },
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_publishers_use_reader_independent_config_projections(self):
        for name in ("py-sdk-publish.yml", "github-release-publish.yml"):
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                self.assertIn(
                    "{id,name,target,enforcement,conditions,rules,updated_at}",
                    workflow,
                )
                self.assertIn("reviewer_id:.reviewer.id", workflow)
                self.assertIn("prevent_self_review", workflow)
                self.assertIn("deployment_branch_policy", workflow)
                self.assertNotIn("current_user_can_bypass", workflow)
                self.assertNotIn("bypass_actors", workflow)
                self.assertNotIn("/secrets", workflow)

                expressions = re.findall(
                    r"jq -S -c '([^']+)' \| sha256sum",
                    workflow,
                )
                environment_expression = next(
                    expression
                    for expression in expressions
                    if "protection_rules" in expression
                )
                admin_view = {
                    "name": "pypi",
                    "can_admins_bypass": False,
                    "protection_rules": [
                        {
                            "id": 1,
                            "type": "required_reviewers",
                            "reviewers": [
                                {
                                    "type": "User",
                                    "reviewer": {"id": 4242, "login": "independent-reviewer"},
                                }
                            ],
                            "prevent_self_review": True,
                        }
                    ],
                    "deployment_branch_policy": {
                        "protected_branches": False,
                        "custom_branch_policies": True,
                    },
                    "url": "admin-relative-url",
                }
                token_view = {
                    **admin_view,
                    "url": "token-relative-url",
                    "html_url": "reader-only-field",
                }
                outputs = []
                for payload in (admin_view, token_view):
                    result = subprocess.run(
                        ["jq", "-S", "-c", environment_expression],
                        input=json.dumps(payload),
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    outputs.append(result.stdout)
                self.assertEqual(outputs[0], outputs[1])

    def test_all_workflow_jq_filters_compile(self):
        token = r"(?:'[^']*'|\"[^\"\n]*\"|[^\s|;&<>\)]+)"
        direct = re.compile(
            rf"(?<![-\w])jq(?:\s+-[A-Za-z]+)+\s+(?P<filter>{token})"
        )
        gh = re.compile(rf"--jq\s+(?P<filter>{token})")
        single_quoted = 0
        for name in self.WORKFLOWS:
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                filters: list[tuple[int, str]] = []
                expected = 0
                for line_number, line in enumerate(workflow.splitlines(), 1):
                    expected += len(re.findall(r"(?<![-\w])jq\s+-", line))
                    expected += line.count("--jq")
                    for pattern in (direct, gh):
                        for match in pattern.finditer(line):
                            expression = match.group("filter")
                            if expression.startswith("'"):
                                single_quoted += 1
                            if expression[:1] in {"'", '"'}:
                                self.assertEqual(expression[-1], expression[0])
                                expression = expression[1:-1]
                            filters.append((line_number, expression))
                self.assertGreater(expected, 0)
                self.assertEqual(len(filters), expected)
                for line_number, expression in filters:
                    with self.subTest(line=line_number, jq=expression):
                        result = subprocess.run(
                            ["jq", "-n", f"try ({expression}) catch empty"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(
                            result.returncode,
                            0,
                            f"{name}:{line_number}: {result.stderr}",
                        )
        self.assertEqual(single_quoted, 23)

    def test_all_workflow_run_blocks_parse_as_bash(self):
        expression = re.compile(r"\$\{\{.*?\}\}")
        for name in self.WORKFLOWS:
            with self.subTest(workflow=name):
                workflow = self.workflow(name)
                lines = workflow.splitlines()
                blocks: list[tuple[int, str]] = []
                index = 0
                while index < len(lines):
                    match = re.match(
                        r"^(?P<indent>\s*)run:\s*(?P<body>.*)$", lines[index]
                    )
                    if match is None:
                        index += 1
                        continue
                    line_number = index + 1
                    body = match.group("body")
                    key_indent = len(match.group("indent"))
                    if re.fullmatch(r"[|>][+-]?", body):
                        index += 1
                        block_lines: list[str] = []
                        content_indent: int | None = None
                        while index < len(lines):
                            line = lines[index]
                            indent = len(line) - len(line.lstrip())
                            if line and indent <= key_indent:
                                break
                            if line and content_indent is None:
                                content_indent = indent
                            if line:
                                self.assertIsNotNone(content_indent)
                                self.assertGreaterEqual(indent, content_indent)
                                block_lines.append(line[content_indent:])
                            else:
                                block_lines.append("")
                            index += 1
                        self.assertIsNotNone(
                            content_indent, f"{name}:{line_number}: empty run block"
                        )
                        body = "\n".join(block_lines) + "\n"
                    else:
                        self.assertTrue(body, f"{name}:{line_number}: empty inline run")
                        index += 1
                    blocks.append(
                        (line_number, expression.sub("GITHUB_EXPRESSION", body))
                    )

                self.assertEqual(
                    len(blocks),
                    len(re.findall(r"^\s*run:", workflow, re.MULTILINE)),
                )
                self.assertGreater(len(blocks), 0)
                for line_number, block in blocks:
                    with self.subTest(line=line_number):
                        result = subprocess.run(
                            ["bash", "-n", "-c", block],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        self.assertEqual(
                            result.returncode,
                            0,
                            f"{name}:{line_number}: {result.stderr}",
                        )

    def test_dispatch_input_counts_fit_github_limit(self):
        for name in (
            "py-sdk-wheels.yml",
            "py-sdk-publish.yml",
            "github-release-publish.yml",
        ):
            with self.subTest(workflow=name):
                workflow = self.workflow(name).split("\npermissions:", 1)[0]
                inputs = re.findall(
                    r"^      ([a-z0-9_]+):(?:\n| )", workflow, re.MULTILINE
                )
                self.assertLessEqual(len(inputs), 25)
        self.assertEqual(
            len(
                re.findall(
                    r"^      ([a-z0-9_]+):(?:\n| )",
                    self.workflow("github-release-publish.yml").split(
                        "\npermissions:", 1
                    )[0],
                    re.MULTILINE,
                )
            ),
            18,
        )

    def test_wheel_workflow_is_single_run_five_platform_and_pretag(self):
        workflow = self.workflow("py-sdk-wheels.yml")
        self.assertIn("BUILD PRETAG v0.2.0", workflow)
        self.assertIn('tags: ["v0.2.0"]', workflow)
        self.assertIn("os: macos-15", workflow)
        self.assertIn("macos-15-intel", workflow)
        self.assertIn("toolchain: 1.99.1", workflow)
        self.assertEqual(workflow.count("rust-toolchain: 1.99.1"), 3)
        self.assertIn("run: rustc -V", workflow)
        self.assertIn("maturin-version: v1.16.0", workflow)
        self.assertIn("@sha256:", workflow)
        self.assertEqual(workflow.count("args: --release --locked --out dist"), 3)
        self.assertIn("expected 5 wheels", workflow)
        self.assertNotIn("command: upload", workflow)
        self.assertNotIn("environment:", workflow)
        self.assertNotIn("ubuntu-latest", workflow)
        self.assertNotIn("windows-latest", workflow)
        self.assertIn("ubuntu-24.04", workflow)
        self.assertIn("windows-2025", workflow)
        self.assertIn("image_version=${ImageVersion:-unavailable}", workflow)

    def test_publishers_bind_control_numeric_actors_and_config(self):
        pypi = self.workflow("py-sdk-publish.yml")
        github = self.workflow("github-release-publish.yml")
        for workflow, environment in ((pypi, "pypi"), (github, "github-release")):
            self.assertIn("parapet-worker:319884068", workflow)
            self.assertIn("reviewer=protected-environment", workflow)
            self.assertIn("GITHUB_WORKFLOW_SHA", workflow)
            self.assertIn("control_commit", workflow)
            self.assertIn("workflow_sha256", workflow)
            self.assertIn("main_ruleset_sha256", workflow)
            self.assertIn("release_ruleset_sha256", workflow)
            self.assertIn(f"name: {environment}", workflow)
            self.assertNotIn("/secrets", workflow)
            self.assertNotIn(
                "${{ github.event.sender",
                "\n".join(
                    line
                    for line in workflow.splitlines()
                    if line.lstrip().startswith("test ")
                ),
            )
        self.assertIn("id-token: write", pypi)
        self.assertIn("ACTIONS_ID_TOKEN_REQUEST_TOKEN", pypi)
        self.assertIn("--non-interactive dist/*.whl", pypi)
        self.assertIn("https://pypi.org/pypi/parapet/0.2.0/json", pypi)
        self.assertNotIn("contents: write", pypi)
        self.assertIn("contents: write", github)
        self.assertIn("--draft=false --latest --verify-tag", github)
        self.assertIn("release_text_sha256", github)
        self.assertIn("{name,body}", github)
        self.assertNotIn("downloadCount", github)
        self.assertIn("{id,name,size,digest,state,contentType,updatedAt}", github)
        self.assertIn("parapet-v0.2.0-RELEASE_SHA256SUMS", github)
        self.assertIn("parapet-v0.2.0-macos-arm64-evaluator.tar.gz", github)
        self.assertIn("parapet-v0.2.0-release-evidence.tar.gz", github)

    def test_pypi_python_validators_accept_only_exact_sets(self):
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
            valid_artifacts = self.run_python_block(
                blocks[0],
                cwd=root,
                env={"ARTIFACTS_JSON": json.dumps(payload)},
            )
            self.assertEqual(
                valid_artifacts.returncode,
                0,
                valid_artifacts.stdout + valid_artifacts.stderr,
            )
            missing = self.run_python_block(
                blocks[0],
                cwd=root,
                env={"ARTIFACTS_JSON": json.dumps({"artifacts": payload["artifacts"][:-1]})},
            )
            self.assertNotEqual(missing.returncode, 0)
            dist = root / "dist"
            dist.mkdir()
            tags = [
                "manylinux_2_17_x86_64",
                "manylinux_2_17_aarch64",
                "macosx_11_0_arm64",
                "macosx_10_12_x86_64",
                "win_amd64",
            ]
            wheels = []
            for tag in tags:
                wheel = dist / f"parapet-0.2.0-py3-none-{tag}.whl"
                self.write_wheel(wheel, tag)
                wheels.append(wheel)
            env = {"EXPECTED_WHEEL_SET_SHA256": self.wheel_set_sha256(wheels)}
            valid_wheels = self.run_python_block(blocks[1], cwd=root, env=env)
            self.assertEqual(
                valid_wheels.returncode, 0, valid_wheels.stdout + valid_wheels.stderr
            )
            (dist / "unexpected.txt").write_text("x")
            invalid = self.run_python_block(blocks[1], cwd=root, env=env)
            self.assertNotEqual(invalid.returncode, 0)

    def test_github_asset_validator_accepts_exact_manifest(self):
        blocks = self.python_blocks(self.workflow("github-release-publish.yml"))
        self.assertEqual(len(blocks), 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assets = root / "release-assets"
            assets.mkdir()
            payloads = [
                "parapet-v0.2.0-macos-arm64-evaluator.tar.gz",
                "parapet-v0.2.0-release-evidence.tar.gz",
            ]
            tags = [
                "manylinux_2_17_x86_64.manylinux2014_x86_64",
                "manylinux_2_17_aarch64.manylinux2014_aarch64",
                "macosx_11_0_arm64",
                "macosx_10_12_x86_64",
                "win_amd64",
            ]
            wheels = []
            for tag in tags:
                wheel = assets / f"parapet-0.2.0-py3-none-{tag}.whl"
                self.write_wheel(wheel, tag)
                wheels.append(wheel)
            for index, name in enumerate(payloads):
                (assets / name).write_bytes(f"asset-{index}".encode())
            manifest = assets / "parapet-v0.2.0-RELEASE_SHA256SUMS"
            covered = wheels + [assets / name for name in payloads]
            manifest.write_text(
                "".join(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
                    for path in covered
                )
            )
            env = {
                "EXPECTED_RELEASE_MANIFEST_SHA256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "EXPECTED_WHEEL_SET_SHA256": self.wheel_set_sha256(wheels),
            }
            valid = self.run_python_block(blocks[0], cwd=root, env=env)
            self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
            env["EXPECTED_RELEASE_MANIFEST_SHA256"] = "0" * 64
            invalid = self.run_python_block(blocks[0], cwd=root, env=env)
            self.assertNotEqual(invalid.returncode, 0)


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
