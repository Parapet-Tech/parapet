"""Frozen A.5 reference / held-out split at code-repo grain, plus shared derivations.

The split assigns each benign carrier to REFERENCE-IN (calibrates the reference
CDF) or HELD-OUT (validation, falsifier and attack hosts) by a salted sha256
bucket of its code-repo key. Because the assignment is a pure function of the
key, no repo can straddle the two sides; ``straddle`` checks the coarser
groupings (cell, task) that a key-grain split does not control by construction.

Everything here is key-level: no trajectory text is read.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

BUCKET_MODULUS = 1000

A5_SALT = "p3-slowburn-benign-split/1"
A5_HELD_OUT_MILLI = 300
A5_SPLIT_FIELD = "split_key"
# Regression pin carried by every P3 construction step: this key must land in this bucket.
A5_REGRESSION_KEY = "agentdojo|banking|user_task_1"
A5_REGRESSION_BUCKET = 854

FLOOR_SAMPLE_SALT = "p3-reference-floor/1"
SWE_SCAFFOLDS = frozenset({"openhands", "swe-agent"})


class SideAssigner(Protocol):
    """Assigns a split key to one side of the reference / held-out split."""

    def bucket(self, key: str) -> int: ...

    def is_held_out(self, key: str) -> bool: ...


@dataclass(frozen=True)
class SaltedBucketSplit:
    """HELD-OUT iff int(sha256(salt + '|' + key)[:8 hex], 16) % 1000 < held_out_milli."""

    salt: str
    held_out_milli: int

    def __post_init__(self) -> None:
        if not self.salt:
            raise ValueError("salt must be non-empty")
        if isinstance(self.held_out_milli, bool) or not isinstance(self.held_out_milli, int):
            raise TypeError("held_out_milli must be an int")
        if not 0 < self.held_out_milli < BUCKET_MODULUS:
            raise ValueError(f"held_out_milli must be in (0, {BUCKET_MODULUS})")

    def bucket(self, key: str) -> int:
        digest = hashlib.sha256((self.salt + "|" + key).encode()).hexdigest()
        return int(digest[:8], 16) % BUCKET_MODULUS

    def is_held_out(self, key: str) -> bool:
        return self.bucket(key) < self.held_out_milli


A5_SPLIT = SaltedBucketSplit(salt=A5_SALT, held_out_milli=A5_HELD_OUT_MILLI)


def assert_regression_pin(split: SideAssigner = A5_SPLIT) -> None:
    """Fail loud unless the A.5 regression key still lands in its pinned bucket."""
    got = split.bucket(A5_REGRESSION_KEY)
    if got != A5_REGRESSION_BUCKET:
        raise ValueError(f"A.5 regression pin: bucket({A5_REGRESSION_KEY!r}) = {got}, "
                         f"expected {A5_REGRESSION_BUCKET}")


def is_benign_carrier(row: Mapping[str, Any]) -> bool:
    """Usable and not an attack run. Strict identity checks: a string 'True' is not usable."""
    return row.get("usable_carrier") is True and row.get("attack_type") is None


def is_benign_swe_carrier(row: Mapping[str, Any]) -> bool:
    """The P3 benign universe: benign carriers from the SWE scaffolds."""
    return is_benign_carrier(row) and row.get("scaffold") in SWE_SCAFFOLDS


@dataclass(frozen=True)
class Partition:
    reference_in: tuple[Mapping[str, Any], ...]
    held_out: tuple[Mapping[str, Any], ...]


def partition(
    rows: Iterable[Mapping[str, Any]],
    split: SideAssigner = A5_SPLIT,
    key_field: str = A5_SPLIT_FIELD,
) -> Partition:
    """Split rows by ``split`` on ``key_field``. A row without the key field raises KeyError."""
    reference_in: list[Mapping[str, Any]] = []
    held_out: list[Mapping[str, Any]] = []
    for row in rows:
        (held_out if split.is_held_out(row[key_field]) else reference_in).append(row)
    return Partition(reference_in=tuple(reference_in), held_out=tuple(held_out))


def straddle(parts: Partition, field: str) -> set[Any]:
    """Values of ``field`` that occur on both sides of the partition."""
    return {r[field] for r in parts.reference_in} & {r[field] for r in parts.held_out}


def inventory_sha256(out_paths: Iterable[str]) -> str:
    """sha256 over the newline-joined sorted carrier paths. Duplicate paths are a data error."""
    paths = sorted(out_paths)
    for a, b in zip(paths, paths[1:]):
        if a == b:
            raise ValueError(f"duplicate out_path in inventory: {a!r}")
    return hashlib.sha256("\n".join(paths).encode()).hexdigest()


def salted_path_order_key(out_path: str, salt: str = FLOOR_SAMPLE_SALT) -> str:
    """Deterministic sample order: sort ascending by this key and take a prefix."""
    return hashlib.sha256((salt + "|" + out_path.casefold()).encode()).hexdigest()


def pick_carrier(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The carrier that represents a cell: least sha256(out_path), ties broken by the path bytes."""
    if not rows:
        raise ValueError("no carrier to pick from")
    paths = [r["out_path"] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate out_path among cell carriers")
    return min(rows, key=lambda r: (hashlib.sha256(r["out_path"].encode()).digest(), r["out_path"].encode()))
