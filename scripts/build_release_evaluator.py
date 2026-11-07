#!/usr/bin/env python3
"""Build and verify the deterministic Parapet v0.1.0 evaluator archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile


EVALUATOR_NAME = "parapet-v0.1.0-macos-arm64-evaluator"
ARCHIVE_NAME = f"{EVALUATOR_NAME}.tar.gz"
WHEEL_NAME = "parapet-0.1.0-py3-none-macosx_11_0_arm64.whl"
CHECKSUM_NAME = "EVALUATOR_SHA256SUMS"
DEMO_FILES = (
    "README.md",
    "config_demo_v0.yaml",
    "doc_benign_write.txt",
    "doc_injection_benign.txt",
    "doc_mismatch.txt",
    "doc_unauthorized.txt",
    "manifest.json",
    "request_benign_write.json",
    "request_injection_benign.json",
    "request_mismatch.json",
    "request_unauthorized.json",
)
STATIC_FILES = (
    ("release/v0.1.0/START_HERE.md", "START_HERE.md"),
    ("README.md", "README.md"),
    ("docs/local-first-use.md", "docs/local-first-use.md"),
    ("release/v0.1.0/RESULT_TEMPLATE.md", "RESULT_TEMPLATE.md"),
)
EXPECTED_DIRS = (
    "dist",
    "docs",
    "parapet",
    "parapet/demo",
    "parapet/demo/v0",
)
CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class BuildError(RuntimeError):
    """Closed evaluator build or verification failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(raw: str) -> PurePosixPath:
    if not raw or "\\" in raw:
        raise BuildError(f"invalid relative path: {raw!r}")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise BuildError(f"invalid relative path: {raw!r}")
    return path


def require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"{label} must be a regular non-symlink file")


def source_file(source_root: Path, relative: PurePosixPath, label: str) -> Path:
    current = source_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise BuildError(f"{label} must not contain symlink components")
    require_regular_file(current, label)
    return current


def validate_wheel(path: Path) -> None:
    if path.name != WHEEL_NAME:
        raise BuildError(f"wheel must be named exactly {WHEEL_NAME}")
    require_regular_file(path, "wheel")
    if not zipfile.is_zipfile(path):
        raise BuildError("wheel is not a valid ZIP archive")

    metadata_name = "parapet-0.1.0.dist-info/METADATA"
    wheel_name = "parapet-0.1.0.dist-info/WHEEL"
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise BuildError("wheel contains duplicate entries")
        for info in infos:
            name = info.filename.rstrip("/")
            normalized = safe_relative_path(name)
            if normalized.as_posix() != name:
                raise BuildError(f"wheel contains a non-canonical path: {name!r}")
            unix_type = (info.external_attr >> 16) & 0o170000
            if unix_type == stat.S_IFLNK:
                raise BuildError(f"wheel contains a symlink: {name!r}")
        if metadata_name not in names or wheel_name not in names:
            raise BuildError("wheel is missing v0.1.0 dist-info metadata")
        metadata = archive.read(metadata_name).decode("utf-8", "strict")
        wheel_metadata = archive.read(wheel_name).decode("utf-8", "strict")
    if not re.search(r"(?m)^Name: parapet$", metadata):
        raise BuildError("wheel metadata has the wrong distribution name")
    if not re.search(r"(?m)^Version: 0\.1\.0$", metadata):
        raise BuildError("wheel metadata has the wrong version")
    if not re.search(r"(?m)^Tag: py3-none-macosx_11_0_arm64$", wheel_metadata):
        raise BuildError("wheel metadata has the wrong platform tag")


def source_file_map(source_root: Path, wheel: Path) -> dict[PurePosixPath, Path]:
    files: dict[PurePosixPath, Path] = {}
    for source_name, dest_name in STATIC_FILES:
        relative = safe_relative_path(source_name)
        source = source_file(source_root, relative, source_name)
        files[safe_relative_path(dest_name)] = source

    demo_relative = safe_relative_path("parapet/demo/v0")
    demo_root = source_root
    for part in demo_relative.parts:
        demo_root /= part
        if demo_root.is_symlink():
            raise BuildError("parapet/demo/v0 must not contain symlink components")
    if not demo_root.is_dir():
        raise BuildError("parapet/demo/v0 must be a regular directory")
    observed = {entry.name for entry in demo_root.iterdir()}
    expected = set(DEMO_FILES)
    if observed != expected:
        raise BuildError(
            "demo asset set mismatch: "
            f"missing={sorted(expected - observed)!r} "
            f"extra={sorted(observed - expected)!r}"
        )
    for name in DEMO_FILES:
        relative = safe_relative_path(f"parapet/demo/v0/{name}")
        source = source_file(source_root, relative, f"demo asset {name}")
        files[safe_relative_path(f"parapet/demo/v0/{name}")] = source

    validate_wheel(wheel)
    files[safe_relative_path(f"dist/{WHEEL_NAME}")] = wheel
    return files


def expected_files() -> set[str]:
    names = {dest for _, dest in STATIC_FILES}
    names.update(f"parapet/demo/v0/{name}" for name in DEMO_FILES)
    names.add(f"dist/{WHEEL_NAME}")
    names.add(CHECKSUM_NAME)
    return names


def tree_entries(root: Path) -> tuple[set[str], set[str]]:
    if root.is_symlink() or not root.is_dir():
        raise BuildError("evaluator root must be a regular directory")
    directories: set[str] = set()
    files: set[str] = set()
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(dirnames):
            path = current_path / name
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                raise BuildError(f"symlink is forbidden: {relative}")
            directories.add(path.relative_to(root).as_posix())
        for name in filenames:
            path = current_path / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise BuildError(f"non-regular file is forbidden: {rel}")
            files.add(rel)
    return directories, files


def write_checksums(root: Path) -> None:
    directories, files = tree_entries(root)
    if directories != set(EXPECTED_DIRS):
        raise BuildError("staged directory set is not exact")
    expected_without_manifest = expected_files() - {CHECKSUM_NAME}
    if files != expected_without_manifest:
        raise BuildError("staged file set is not exact before checksum generation")
    lines = [f"{sha256_file(root / name)}  {name}\n" for name in sorted(files)]
    checksum_path = root / CHECKSUM_NAME
    with checksum_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(lines))
    os.chmod(checksum_path, 0o644)


def verify_evaluator_dir(root: Path) -> None:
    directories, files = tree_entries(root)
    if directories != set(EXPECTED_DIRS):
        raise BuildError(
            f"evaluator directory set mismatch: {sorted(directories)!r}"
        )
    if files != expected_files():
        raise BuildError(f"evaluator file set mismatch: {sorted(files)!r}")

    checksum_path = root / CHECKSUM_NAME
    require_regular_file(checksum_path, CHECKSUM_NAME)
    entries = parse_checksums(checksum_path.read_text(encoding="utf-8"))
    for name, expected_digest in entries.items():
        if sha256_file(root / name) != expected_digest:
            raise BuildError(f"checksum mismatch: {name}")


def parse_checksums(contents: str) -> dict[str, str]:
    raw_lines = contents.splitlines()
    entries: dict[str, str] = {}
    for line in raw_lines:
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            raise BuildError(f"invalid checksum line: {line!r}")
        digest, raw_name = match.groups()
        name = safe_relative_path(raw_name).as_posix()
        if name == CHECKSUM_NAME or name in entries:
            raise BuildError(f"invalid checksum entry: {name!r}")
        entries[name] = digest
    covered = expected_files() - {CHECKSUM_NAME}
    if list(entries) != sorted(entries) or set(entries) != covered:
        raise BuildError("checksum manifest does not cover the exact sorted file set")
    return entries


def normalized_tar_info(
    name: str, *, directory: bool, size: int = 0
) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if directory else 0o644
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = 0 if directory else size
    return info


def create_archive(root: Path, archive_path: Path) -> None:
    verify_evaluator_dir(root)
    with archive_path.open("xb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9
        ) as zipped:
            with tarfile.open(
                fileobj=zipped, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                archive.addfile(normalized_tar_info(EVALUATOR_NAME, directory=True))
                for directory in sorted(EXPECTED_DIRS):
                    archive.addfile(
                        normalized_tar_info(
                            f"{EVALUATOR_NAME}/{directory}", directory=True
                        )
                    )
                for name in sorted(expected_files()):
                    data = (root / name).read_bytes()
                    archive.addfile(
                        normalized_tar_info(
                            f"{EVALUATOR_NAME}/{name}", directory=False, size=len(data)
                        ),
                        io.BytesIO(data),
                    )


def verify_archive(archive_path: Path) -> None:
    require_regular_file(archive_path, "evaluator archive")
    expected_names = {EVALUATOR_NAME}
    expected_names.update(f"{EVALUATOR_NAME}/{name}" for name in EXPECTED_DIRS)
    expected_names.update(f"{EVALUATOR_NAME}/{name}" for name in expected_files())
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        expected_order = [EVALUATOR_NAME]
        expected_order.extend(
            f"{EVALUATOR_NAME}/{name}" for name in sorted(EXPECTED_DIRS)
        )
        expected_order.extend(
            f"{EVALUATOR_NAME}/{name}" for name in sorted(expected_files())
        )
        if names != expected_order:
            raise BuildError("archive member ordering or set is not exact")
        archived_files: dict[str, bytes] = {}
        for member in members:
            if member.name not in expected_names or not (
                member.isdir() or member.isfile()
            ):
                raise BuildError(f"invalid archive member: {member.name!r}")
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mtime != 0
                or member.mode != (0o755 if member.isdir() else 0o644)
            ):
                raise BuildError(f"archive metadata is not normalized: {member.name}")
            if member.isfile():
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise BuildError(f"archive file cannot be read: {member.name}")
                relative = member.name.removeprefix(f"{EVALUATOR_NAME}/")
                archived_files[relative] = extracted.read()

    try:
        checksum_text = archived_files[CHECKSUM_NAME].decode("utf-8", "strict")
    except (KeyError, UnicodeError) as error:
        raise BuildError("archive checksum manifest cannot be read") from error
    entries = parse_checksums(checksum_text)
    for name, expected_digest in entries.items():
        observed_digest = hashlib.sha256(archived_files[name]).hexdigest()
        if observed_digest != expected_digest:
            raise BuildError(f"archive checksum mismatch: {name}")


def build_release_evaluator(
    source_root: Path, wheel: Path, out: Path
) -> tuple[Path, Path]:
    source_root = source_root.absolute()
    wheel = wheel.absolute()
    out = out.absolute()
    if source_root.is_symlink() or not source_root.is_dir():
        raise BuildError("source root must be a regular non-symlink directory")
    if out.exists() or out.is_symlink():
        raise BuildError("output path must not already exist")
    if out.parent.is_symlink() or not out.parent.is_dir():
        raise BuildError("output parent must be an existing regular directory")

    sources = source_file_map(source_root, wheel)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        evaluator = temporary / EVALUATOR_NAME
        evaluator.mkdir(mode=0o755)
        for directory in EXPECTED_DIRS:
            (evaluator / directory).mkdir(mode=0o755, exist_ok=True)
        for relative, source in sorted(
            sources.items(), key=lambda item: item[0].as_posix()
        ):
            destination = evaluator / relative.as_posix()
            destination.write_bytes(source.read_bytes())
            os.chmod(destination, 0o644)
        write_checksums(evaluator)
        verify_evaluator_dir(evaluator)
        archive = temporary / ARCHIVE_NAME
        create_archive(evaluator, archive)
        verify_archive(archive)
        os.rename(temporary, out)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return out / EVALUATOR_NAME, out / ARCHIVE_NAME


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    source_root = Path(__file__).resolve().parent.parent
    try:
        evaluator, archive = build_release_evaluator(source_root, args.wheel, args.out)
    except (
        BuildError,
        OSError,
        UnicodeError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as error:
        print(f"release evaluator build refused: {error}", file=sys.stderr)
        return 2
    print(f"evaluator_dir={evaluator}")
    print(f"archive={archive}")
    print(f"archive_sha256={sha256_file(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
