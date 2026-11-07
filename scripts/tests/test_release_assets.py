from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_release_evaluator.py"
SPEC = importlib.util.spec_from_file_location("build_release_evaluator", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class ReleaseAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "source"
        self._copy_source_fixture()
        self.wheel = self.base / builder.WHEEL_NAME
        self._write_wheel(self.wheel)

    def _copy_source_fixture(self) -> None:
        for source_name, _ in builder.STATIC_FILES:
            source = ROOT / source_name
            destination = self.source / source_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        demo_source = ROOT / "parapet/demo/v0"
        demo_destination = self.source / "parapet/demo/v0"
        demo_destination.mkdir(parents=True)
        for name in builder.DEMO_FILES:
            shutil.copyfile(demo_source / name, demo_destination / name)

    def _write_wheel(
        self,
        path: Path,
        *,
        distribution: str = "parapet",
        version: str = "0.1.0",
        tag: str = "py3-none-macosx_11_0_arm64",
        extra_tags: tuple[str, ...] = (),
        include_scripts: bool = True,
    ) -> None:
        dist_info = "parapet-0.1.0.dist-info"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("parapet/__init__.py", "__version__ = '0.1.0'\n")
            archive.writestr(
                f"{dist_info}/METADATA",
                f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
            )
            archive.writestr(
                f"{dist_info}/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n"
                + "".join(f"Tag: {item}\n" for item in (tag, *extra_tags)),
            )
            if include_scripts:
                for script_name in builder.WHEEL_SCRIPTS:
                    archive.writestr(script_name, b"fixture-binary\n")

    def _build(self, name: str = "out") -> tuple[Path, Path]:
        return builder.build_release_evaluator(
            self.source, self.wheel, self.base / name
        )

    def test_positive_layout_checksum_archive_and_repeat_determinism(self) -> None:
        evaluator_a, archive_a = self._build("out-a")
        evaluator_b, archive_b = self._build("out-b")
        builder.verify_evaluator_dir(evaluator_a)
        builder.verify_evaluator_dir(evaluator_b)
        builder.verify_archive(archive_a)
        builder.verify_archive(archive_b)
        self.assertEqual(archive_a.read_bytes(), archive_b.read_bytes())
        self.assertEqual(
            builder.sha256_file(archive_a), builder.sha256_file(archive_b)
        )

        directories, files = builder.tree_entries(evaluator_a)
        self.assertEqual(directories, set(builder.EXPECTED_DIRS))
        self.assertEqual(files, builder.expected_files())
        checksum_lines = (
            (evaluator_a / builder.CHECKSUM_NAME).read_text().splitlines()
        )
        checksum_names = [line.split("  ", 1)[1] for line in checksum_lines]
        self.assertEqual(checksum_names, sorted(checksum_names))
        self.assertNotIn(builder.CHECKSUM_NAME, checksum_names)
        shasum = shutil.which("shasum")
        if shasum is not None:
            completed = subprocess.run(
                [shasum, "-a", "256", "-c", builder.CHECKSUM_NAME],
                cwd=evaluator_a,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

        with tarfile.open(archive_a, "r:gz") as archive:
            members = archive.getmembers()
        self.assertTrue(
            all(member.uid == member.gid == member.mtime == 0 for member in members)
        )
        self.assertTrue(
            all(not member.issym() and not member.islnk() for member in members)
        )

    def test_cli_builds_from_repository_assets(self) -> None:
        output = self.base / "cli-out"
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--wheel",
                str(self.wheel),
                "--out",
                str(output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(f"archive={output / builder.ARCHIVE_NAME}", completed.stdout)
        builder.verify_evaluator_dir(output / builder.EVALUATOR_NAME)

    def test_missing_referenced_path_fails_closed(self) -> None:
        (self.source / "README.md").unlink()
        with self.assertRaisesRegex(builder.BuildError, "regular non-symlink"):
            self._build()
        self.assertFalse((self.base / "out").exists())

    def test_extra_demo_path_fails_closed(self) -> None:
        (self.source / "parapet/demo/v0/extra.txt").write_text("extra")
        with self.assertRaisesRegex(builder.BuildError, "demo asset set mismatch"):
            self._build()

    def test_wrong_wheel_filename_and_internal_tag_fail_closed(self) -> None:
        wrong_name = self.base / "parapet-0.1.0-py3-none-win_amd64.whl"
        self._write_wheel(wrong_name)
        with self.assertRaisesRegex(builder.BuildError, "named exactly"):
            builder.build_release_evaluator(
                self.source, wrong_name, self.base / "bad-name"
            )

        self.wheel.unlink()
        self._write_wheel(self.wheel, tag="py3-none-macosx_10_9_x86_64")
        with self.assertRaisesRegex(builder.BuildError, "must contain only Tag"):
            self._build("bad-tag")

        self.wheel.unlink()
        self._write_wheel(self.wheel, version="0.1.1")
        with self.assertRaisesRegex(builder.BuildError, "wrong version"):
            self._build("bad-version")

    def test_extra_wheel_tag_and_missing_scripts_fail_closed(self) -> None:
        self.wheel.unlink()
        self._write_wheel(self.wheel, extra_tags=("py3-none-win_amd64",))
        with self.assertRaisesRegex(builder.BuildError, "must contain only Tag"):
            self._build("extra-tag")

        self.wheel.unlink()
        self._write_wheel(self.wheel, include_scripts=False)
        with self.assertRaisesRegex(builder.BuildError, "missing required script"):
            self._build("missing-scripts")

    def test_changed_byte_and_stale_checksum_fail_closed(self) -> None:
        evaluator, _ = self._build()
        readme = evaluator / "README.md"
        readme.write_bytes(readme.read_bytes() + b"changed\n")
        with self.assertRaisesRegex(builder.BuildError, "checksum mismatch: README.md"):
            builder.verify_evaluator_dir(evaluator)

    def test_symlink_source_fails_closed(self) -> None:
        target = self.source / "README.target"
        source = self.source / "README.md"
        source.rename(target)
        try:
            source.symlink_to(target.name)
        except OSError as error:
            self.skipTest(f"symlink unavailable: {error}")
        with self.assertRaisesRegex(builder.BuildError, "symlink"):
            self._build()

    def test_symlinked_source_component_and_wheel_fail_closed(self) -> None:
        real_docs = self.source / "real-docs"
        (self.source / "docs").rename(real_docs)
        try:
            (self.source / "docs").symlink_to(real_docs.name, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symlink unavailable: {error}")
        with self.assertRaisesRegex(builder.BuildError, "symlink components"):
            self._build("bad-source-component")

        (self.source / "docs").unlink()
        real_docs.rename(self.source / "docs")
        real_wheel = self.base / "real" / builder.WHEEL_NAME
        real_wheel.parent.mkdir()
        self.wheel.rename(real_wheel)
        self.wheel.symlink_to(real_wheel)
        with self.assertRaisesRegex(builder.BuildError, "regular non-symlink"):
            self._build("bad-wheel-symlink")

    def test_checksum_traversal_fails_closed(self) -> None:
        evaluator, _ = self._build()
        checksum = evaluator / builder.CHECKSUM_NAME
        lines = checksum.read_text().splitlines()
        digest, _ = lines[0].split("  ", 1)
        lines[0] = f"{digest}  ../escape"
        checksum.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(builder.BuildError, "invalid relative path"):
            builder.verify_evaluator_dir(evaluator)

        absolute_evaluator, _ = self._build("absolute-out")
        absolute_checksum = absolute_evaluator / builder.CHECKSUM_NAME
        absolute_lines = absolute_checksum.read_text().splitlines()
        digest, _ = absolute_lines[0].split("  ", 1)
        absolute_lines[0] = f"{digest}  /escape"
        absolute_checksum.write_text("\n".join(absolute_lines) + "\n")
        with self.assertRaisesRegex(builder.BuildError, "invalid relative path"):
            builder.verify_evaluator_dir(absolute_evaluator)

    def test_extra_staged_path_fails_closed(self) -> None:
        evaluator, _ = self._build()
        (evaluator / "unexpected.txt").write_text("unexpected")
        with self.assertRaisesRegex(builder.BuildError, "file set mismatch"):
            builder.verify_evaluator_dir(evaluator)

    def test_existing_output_refused_without_changes(self) -> None:
        output = self.base / "existing"
        output.mkdir()
        marker = output / "caller.marker"
        marker.write_bytes(b"caller-owned\n")
        with self.assertRaisesRegex(builder.BuildError, "must not already exist"):
            builder.build_release_evaluator(self.source, self.wheel, output)
        self.assertEqual(marker.read_bytes(), b"caller-owned\n")
        self.assertEqual([entry.name for entry in output.iterdir()], ["caller.marker"])

    def test_public_docs_name_scope_and_limitations(self) -> None:
        start = (ROOT / "release/v0.1.0/START_HERE.md").read_text()
        result = (ROOT / "release/v0.1.0/RESULT_TEMPLATE.md").read_text()
        notes = (ROOT / "release/v0.1.0/RELEASE_NOTES.md").read_text()
        self.assertIn(builder.WHEEL_NAME, start)
        self.assertIn("no real model", start.lower())
        self.assertIn("not an independent usability validation", start)
        self.assertIn("tagged\n`v0.1.0` source archive", start)
        self.assertIn("not part of this 15-minute evaluator\njourney", start)
        self.assertIn("not stable or\nproduction-ready", notes)
        self.assertIn(builder.ARCHIVE_NAME, notes)
        self.assertIn("Known issues and limitations", notes)
        for field in (
            "Result:",
            "Elapsed time:",
            "Wheel SHA-256:",
            "Deviations",
            "Cleanup result",
        ):
            self.assertIn(field, result)


if __name__ == "__main__":
    unittest.main()
