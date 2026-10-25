#!/usr/bin/env python3
"""Fail closed unless a push uses Parapet's controlled publisher context."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


ALLOWED_IDENTITY = "some one <someone@example.com>"
EXPECTED_REMOTE = "git@github.com:Parapet-Tech/parapet.git"
EXPECTED_SSH_ACTOR = "parapet-worker"
EXPECTED_KEY_BASENAME = "parapet_publisher"
FORBIDDEN_GIT_OVERRIDES = (
    "GIT_SSH",
    "GIT_SSH_COMMAND",
    "GIT_SSH_VARIANT",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
)


class CommandRunner(Protocol):
    """Execution boundary used by the live checker and deterministic tests."""

    def run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]: ...


class SubprocessRunner:
    def run(self, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, capture_output=True, text=True, check=False)


@dataclass(frozen=True)
class PublisherContext:
    hooks_path: str
    identity: str
    remote_url: str
    ssh_command: str
    active_overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class SshProbe:
    executable: str
    identity_file: str

    def command(self) -> list[str]:
        return [
            self.executable,
            "-T",
            "-i",
            os.path.expanduser(self.identity_file),
            "-o",
            "IdentitiesOnly=yes",
            "git@github.com",
        ]


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def read_context(remote: str) -> PublisherContext:
    name = _git_value("config", "--local", "--get", "user.name")
    email = _git_value("config", "--local", "--get", "user.email")
    identity = f"{name} <{email}>" if name or email else ""
    return PublisherContext(
        hooks_path=_git_value("config", "--local", "--get", "core.hooksPath"),
        identity=identity,
        remote_url=_git_value("remote", "get-url", "--push", remote),
        ssh_command=_git_value("config", "--local", "--get", "core.sshCommand"),
        active_overrides=tuple(
            name for name in FORBIDDEN_GIT_OVERRIDES if os.environ.get(name)
        ),
    )


def validate_context(context: PublisherContext) -> tuple[list[str], SshProbe | None]:
    problems: list[str] = []
    if context.active_overrides:
        problems.append(
            "publisher transport/config overrides are forbidden: "
            + ", ".join(context.active_overrides)
        )
    if context.hooks_path != ".githooks":
        problems.append("core.hooksPath must be .githooks")
    if context.identity != ALLOWED_IDENTITY:
        problems.append("repo identity is not the anonymous identity")
    if context.remote_url != EXPECTED_REMOTE:
        problems.append("push remote is not the canonical SSH remote")

    try:
        tokens = shlex.split(context.ssh_command)
    except ValueError:
        tokens = []
    probe: SshProbe | None = None
    if not tokens or Path(tokens[0]).name != "ssh":
        problems.append("core.sshCommand must invoke ssh")
    else:
        identity_files: list[str] = []
        identities_only = False
        index = 1
        valid = True
        while index < len(tokens):
            token = tokens[index]
            if token == "-i" and index + 1 < len(tokens):
                identity_files.append(tokens[index + 1])
                index += 2
            elif token == "-o" and index + 1 < len(tokens):
                identities_only = tokens[index + 1].lower() == "identitiesonly=yes"
                valid = valid and identities_only
                index += 2
            else:
                valid = False
                break
        if not valid:
            problems.append("core.sshCommand contains unsupported publisher options")
        if len(identity_files) != 1 or Path(identity_files[0]).name != EXPECTED_KEY_BASENAME:
            problems.append("core.sshCommand must select the dedicated publisher key")
        if not identities_only:
            problems.append("core.sshCommand must set IdentitiesOnly=yes")
        if valid and len(identity_files) == 1 and identities_only:
            probe = SshProbe(tokens[0], identity_files[0])
    return problems, probe


def verify_ssh_actor(probe: SshProbe, runner: CommandRunner) -> str | None:
    result = runner.run(probe.command())
    output = f"{result.stdout}\n{result.stderr}"
    expected = f"Hi {EXPECTED_SSH_ACTOR}!"
    if expected not in output:
        return "publisher SSH key did not authenticate as parapet-worker"
    return None


def check_context(
    context: PublisherContext, runner: CommandRunner
) -> list[str]:
    """Validate static context before making the live actor probe."""
    problems, probe = validate_context(context)
    if problems:
        return problems
    if probe is None:
        return ["publisher SSH probe could not be constructed"]
    actor_problem = verify_ssh_actor(probe, runner)
    if actor_problem:
        problems.append(actor_problem)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="origin")
    args = parser.parse_args()

    context = read_context(args.remote)
    problems = check_context(context, SubprocessRunner())

    for problem in problems:
        print(f"publisher context: FAIL: {problem}", file=sys.stderr)
    if problems:
        print("push refused; use the controlled parapet-worker publisher context", file=sys.stderr)
        return 1
    print("publisher context: clean (parapet-worker)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
