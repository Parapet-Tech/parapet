#!/usr/bin/env python3
"""Fail closed on commits, tags, or blobs that leak identity or tooling metadata.

Every commit in the checked range must satisfy:

1. Author and committer are exactly the repo's anonymous identity.
2. The raw commit object carries no embedded signature (gpgsig header).
3. The message carries no tool trailers, session URLs, or noreply identities,
   and no absolute home-directory path.
4. Added diff lines carry no blocked content pattern and no home-directory
   path.
5. Blobs introduced by the commit (text or binary, including PDFs) carry no
   blocked byte pattern. Diff-line checks cannot see binary content; this
   blob-level pass can.

In --all mode EVERY ref is enumerated (rev-list --all: branches, tags, and
any PR refs present in a mirror), and every annotated tag object is also
checked: tagger identity, tag message, and absence of PGP signature blocks.

--full-blob-scan additionally scans every blob reachable from any ref
(prepublication control; slower, use before a first push or after a
history rewrite).

Path rules use exact path-component boundaries: /Users/<name>, /home/<name>,
and <drive>:/Users/<name> are blocked with or without a trailing slash, and
only the exact fixture components listed in USERS_FIXTURES / HOME_FIXTURES
are exempt (a prefix such as user-prod is NOT exempted by the user fixture).

Real identifying strings must never appear in this public script. Put them,
one literal per line ('#' starts a comment), in a gitignored file named
.hygiene-blocklist.local at the repo root (override with the
HYGIENE_BLOCKLIST env var). When present, those literals are blocked in
messages, identities, diffs, tags, and blob bytes.

Usage:
    check_public_hygiene.py <rev-range>          # e.g. origin/main..HEAD
    check_public_hygiene.py --all                # every branch + tag object
    check_public_hygiene.py --all --full-blob-scan

Exit 0 when clean, 1 with a report otherwise.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ALLOWED_IDENTITY = "some one <someone@example.com>"

MESSAGE_BLOCKLIST = [
    re.compile(r"^Claude-Session:", re.M),
    re.compile(r"^Co-Authored-By:", re.M | re.I),
    re.compile(r"claude\.ai", re.I),
    re.compile(r"@users\.noreply\.github\.com"),
    re.compile(r"noreply@anthropic\.com"),
    re.compile(r"^🤖 Generated with", re.M),
]

# High-signal patterns blocked in added diff lines and in blob bytes.
# Generic forms only; real identifying literals belong in the local blocklist.
CONTENT_BLOCKLIST = [
    r"claude\.ai/code/session_",
    r"Claude-Session:",
    r"Co-Authored-By: Claude",
    r"@users\.noreply\.github\.com",
    r"MacBook-Pro",
]

# Exact fixture components exempt from the path rules (measured legitimate
# content: C:/Users/example/... test fixtures, /Users/Documents dataset text).
USERS_FIXTURES = ["example", "Documents"]
HOME_FIXTURES = ["user", "runner", "test", "example", "fixture"]

# This script necessarily contains the literal patterns it blocks, so its
# own blob/diff content is exempt from the CONTENT pattern scan (identity,
# message, and path checks still fully apply to commits touching it).
SCAN_EXEMPT_PATHS = frozenset({"scripts/check_public_hygiene.py"})

_COMP = r"[A-Za-z0-9_.-]+"
_END = r"(?![A-Za-z0-9_.-])"


def _leak_rule(prefix: str, allowed: list[str]) -> str:
    allow = "|".join(re.escape(a) for a in allowed)
    return rf"{prefix}(?!(?:{allow}){_END}){_COMP}{_END}"


PATH_LEAK = re.compile(
    "|".join(
        [
            _leak_rule(r"(?<![\w/])/Users/", USERS_FIXTURES),
            _leak_rule(r"(?<![\w/])/home/", HOME_FIXTURES),
            _leak_rule(r"(?<![\w/])[A-Za-z]:[/\\]+Users[/\\]+", USERS_FIXTURES),
            # MSYS / Git-Bash drive form: /c/Users/<name>
            _leak_rule(r"(?<![\w/])/[A-Za-z]/Users/", USERS_FIXTURES),
        ]
    )
)

PATH_LEAK_B = re.compile(PATH_LEAK.pattern.encode())
CONTENT_BLOCKLIST_RE = [re.compile(p) for p in CONTENT_BLOCKLIST]
CONTENT_BLOCKLIST_B = [re.compile(p.encode()) for p in CONTENT_BLOCKLIST]

MAX_BLOB_BYTES = 100 * 1024 * 1024


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True
    ).stdout


def load_local_blocklist() -> list[bytes]:
    path = os.environ.get("HYGIENE_BLOCKLIST")
    if not path:
        try:
            top = git("rev-parse", "--show-toplevel").strip()
            path = os.path.join(top, ".hygiene-blocklist.local")
        except subprocess.CalledProcessError:
            path = ".hygiene-blocklist.local"
    if not os.path.isfile(path):
        return []
    literals: list[bytes] = []
    with open(path, "rb") as fh:
        for raw in fh:
            entry = raw.strip()
            if entry and not entry.startswith(b"#"):
                literals.append(entry)
    return literals


LOCAL_BLOCKLIST = load_local_blocklist()


def scan_text(text: str, where: str) -> list[str]:
    problems: list[str] = []
    for pattern in CONTENT_BLOCKLIST_RE:
        if pattern.search(text):
            problems.append(f"{where} matches blocked pattern {pattern.pattern!r}")
    if PATH_LEAK.search(text):
        problems.append(f"{where} contains an absolute home-directory path")
    data = text.encode("utf-8", "surrogateescape")
    for literal in LOCAL_BLOCKLIST:
        if literal in data:
            problems.append(f"{where} contains a local-blocklist literal")
            break
    return problems


def scan_blob(sha: str, path: str) -> list[str]:
    size = int(git("cat-file", "-s", sha).strip())
    if size > MAX_BLOB_BYTES:
        return [f"blob {sha[:12]} ({path}) exceeds scan size limit; refusing"]
    data = git_bytes("cat-file", "blob", sha)
    problems: list[str] = []
    if PATH_LEAK_B.search(data):
        problems.append(f"blob {sha[:12]} ({path}) contains a home-directory path")
    for pattern in CONTENT_BLOCKLIST_B:
        if pattern.search(data):
            problems.append(
                f"blob {sha[:12]} ({path}) matches blocked pattern"
                f" {pattern.pattern.decode()!r}"
            )
    for literal in LOCAL_BLOCKLIST:
        if literal in data:
            problems.append(f"blob {sha[:12]} ({path}) contains a local-blocklist literal")
            break
    return problems


def new_blobs(cid: str) -> list[tuple[str, str]]:
    out = git(
        "diff-tree", "-r", "-m", "--root", "--no-commit-id",
        "--diff-filter=AM", cid,
    )
    blobs: dict[str, str] = {}
    for line in out.splitlines():
        if not line.startswith(":"):
            continue
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) >= 4:
            sha = parts[3]
            if sha != "0" * len(sha):
                blobs.setdefault(sha, path)
    return list(blobs.items())


def check_commit(cid: str, seen_blobs: set[str]) -> list[str]:
    problems: list[str] = []
    author, committer, message = git(
        "log", "-1", "--format=%an <%ae>%x00%cn <%ce>%x00%B", cid
    ).split("\x00", 2)
    if author != ALLOWED_IDENTITY:
        problems.append(f"author is {author!r}, must be {ALLOWED_IDENTITY!r}")
    if committer != ALLOWED_IDENTITY:
        problems.append(f"committer is {committer!r}, must be {ALLOWED_IDENTITY!r}")
    raw = git("cat-file", "commit", cid)
    if re.search(r"^gpgsig", raw, re.M):
        problems.append("commit object carries an embedded signature (gpgsig)")
    for pattern in MESSAGE_BLOCKLIST:
        if pattern.search(message):
            problems.append(f"message matches blocked pattern {pattern.pattern!r}")
    problems.extend(scan_text(message, "message"))
    diff = git("show", "--format=", "--no-color", "-m", cid)
    current_path = ""
    for line in diff.splitlines():
        if line.startswith("+++ "):
            current_path = line[4:].removeprefix("b/")
        elif line.startswith("+"):
            if current_path in SCAN_EXEMPT_PATHS:
                continue
            line_problems = scan_text(line, "added diff line")
            if line_problems:
                problems.append(f"{line_problems[0]}: {line[:120]}")
                break
    for sha, path in new_blobs(cid):
        if sha in seen_blobs or path in SCAN_EXEMPT_PATHS:
            continue
        seen_blobs.add(sha)
        problems.extend(scan_blob(sha, path))
    return problems


def check_tag(refname: str) -> list[str]:
    raw = git("cat-file", "tag", refname)
    header, _, message = raw.partition("\n\n")
    problems: list[str] = []
    tagger = None
    for line in header.splitlines():
        if line.startswith("tagger "):
            tagger = re.sub(r" \d+ [+-]\d{4}$", "", line[len("tagger "):])
    if tagger is None:
        problems.append("tag object has no tagger header")
    elif tagger != ALLOWED_IDENTITY:
        problems.append(f"tagger is {tagger!r}, must be {ALLOWED_IDENTITY!r}")
    if "-----BEGIN PGP SIGNATURE-----" in raw:
        problems.append("tag object carries a PGP signature")
    for pattern in MESSAGE_BLOCKLIST:
        if pattern.search(message):
            problems.append(f"tag message matches blocked pattern {pattern.pattern!r}")
    problems.extend(scan_text(message, "tag message"))
    return problems


def full_blob_scan() -> int:
    listing = git("rev-list", "--objects", "--all")
    entries: dict[str, str] = {}
    for line in listing.splitlines():
        sha, _, path = line.partition(" ")
        entries.setdefault(sha, path)
    batch = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
        input="\n".join(entries),
        capture_output=True, text=True, check=True,
    ).stdout
    dirty = 0
    for line in batch.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "blob":
            if entries.get(parts[0]) in SCAN_EXEMPT_PATHS:
                continue
            problems = scan_blob(parts[0], entries.get(parts[0], "?"))
            for problem in problems:
                dirty += 1
                print(f"LEAK {problem}")
    return dirty


def main() -> int:
    args = sys.argv[1:]
    do_full_scan = "--full-blob-scan" in args
    args = [a for a in args if a != "--full-blob-scan"]
    if len(args) == 2 and args[0] == "--tag-object":
        ref = args[1]
        if git("cat-file", "-t", ref).strip() != "tag":
            print("public hygiene: clean (lightweight tag, commit checks apply)")
            return 0
        problems = check_tag(ref)
        for problem in problems:
            print(f"LEAK tag {ref}\n  - {problem}")
        if problems:
            print("\ntag failed public hygiene; nothing may be pushed.")
            return 1
        print("public hygiene: clean")
        return 0
    if not args and not do_full_scan:
        print(__doc__)
        return 2
    all_mode = args == ["--all"]
    range_spec = ["--all"] if all_mode else args
    dirty = 0
    seen_blobs: set[str] = set()
    if range_spec:
        for cid in git("rev-list", *range_spec).split():
            problems = check_commit(cid, seen_blobs)
            if problems:
                dirty += 1
                subject = git("log", "-1", "--format=%h %s", cid).strip()
                print(f"LEAK {subject}")
                for problem in problems:
                    print(f"  - {problem}")
    if all_mode:
        for line in git(
            "for-each-ref", "--format=%(refname) %(objecttype)", "refs/tags"
        ).splitlines():
            refname, objecttype = line.split()
            if objecttype != "tag":
                continue
            problems = check_tag(refname)
            if problems:
                dirty += 1
                print(f"LEAK tag {refname}")
                for problem in problems:
                    print(f"  - {problem}")
    if do_full_scan:
        dirty += full_blob_scan()
    if dirty:
        print(f"\n{dirty} object(s) failed public hygiene; nothing may be pushed.")
        return 1
    print("public hygiene: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
