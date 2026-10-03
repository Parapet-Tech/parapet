"""R2 / R5 backstop for the frozen reference: name-twin exclusion and trajectory near-duplicates.

A repo-grain split cannot see one codebase living under two owner names, and it cannot
see a held-out trajectory that is nearly a copy of a reference one. Two checks cover that:

B1  NAME-TWIN EXCLUSION. A reference-in repo whose casefolded name component (after the
    last ``/``) equals that of any held-out repo leaves the build set. Keys only, no
    threshold, no judgement about which twins are the same project.
B2  TRAJECTORY NEAR-DUPLICATE. For each cited held-out carrier, the nearest build-set
    carrier from another repo by Jaccard over ordered 3-gram shingles of per-event content
    hashes. A null sample drawn from the build set says whether the threshold means anything.

Everything here is corpus-free. Trajectories reach the scan only as event hashes, through
a loader the caller supplies; no event text is held, returned or logged.
"""
from __future__ import annotations

import collections
import hashlib
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from parapet_data.p3.reference.split import A5_SPLIT_FIELD, NATIVE_EVENTS_FIELD, salted_path_order_key

# B2 thresholds. BLOCKER at THRESHOLD; the other two are reported as sensitivity.
THRESHOLD = 0.50
SENSITIVITY = (0.30, 0.70)
SHINGLE_WIDTH = 3

NULL_SAMPLE_SIZE = 2000
# Above this share of the null at or over THRESHOLD, the backstop cannot gate.
NULL_SATURATED_SHARE = 0.01
NULL_QUANTILES = (0.50, 0.90, 0.99, 0.999)

FLAG_SATURATED = "SATURATED/NON-GATING"
FLAG_UNTESTED = "UNTESTED-BY-NULL"

VERDICT_NON_GATING = "NON-GATING"
VERDICT_BLOCKER = "BLOCKER"
VERDICT_PASS = "PASS"


class EventHashLoader(Protocol):
    """Returns the per-event content hashes of one carrier, in trace order.

    An unreadable carrier must raise: a carrier that cannot be read cannot be declared clean.
    """

    def __call__(self, out_path: str) -> Sequence[bytes]: ...


# ---------------------------------------------------------------- B1

def name_component(split_key: str) -> str:
    """The repo name without its owner, casefolded. Casefold only: this is not ``canonical_key``."""
    return split_key.rsplit("/", 1)[-1].casefold()


def name_twin_exclusion(
    reference_rows: Iterable[Mapping[str, Any]], held_out_keys: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Reference-in repos that share a name component with any held-out repo.

    Returns ``{repo: {"repo", "held_out_twins", "carriers", "events"}}`` in sorted repo
    order, counting the carriers and native events the exclusion removes. A row without
    a split key raises KeyError; an excluded row without an int event count raises.
    """
    held_by_name: dict[str, set[str]] = collections.defaultdict(set)
    for key in held_out_keys:
        held_by_name[name_component(key)].add(key)
    per_repo: dict[str, dict[str, int]] = collections.defaultdict(lambda: {"carriers": 0, "events": 0})
    for row in reference_rows:
        key = row[A5_SPLIT_FIELD]
        if name_component(key) in held_by_name:
            events = row[NATIVE_EVENTS_FIELD]
            if isinstance(events, bool) or not isinstance(events, int):
                raise TypeError(f"{NATIVE_EVENTS_FIELD} must be an int, got {events!r} for {key!r}")
            per_repo[key]["carriers"] += 1
            per_repo[key]["events"] += events
    return {key: {"repo": key, "held_out_twins": sorted(held_by_name[name_component(key)]), **counts}
            for key, counts in sorted(per_repo.items())}


# ---------------------------------------------------------------- hashing

def event_hash(event: Mapping[str, Any]) -> bytes:
    """sha256(str(function) + NUL + str(arguments)), 32 raw bytes. The text is not retained.

    A missing field hashes as the string ``None``, as the frozen recipe does.
    """
    if not isinstance(event, Mapping):
        raise TypeError(f"event must be a mapping, got {type(event).__name__}")
    return hashlib.sha256((str(event.get("function")) + "\x00" + str(event.get("arguments"))).encode()).digest()


def shingles(event_hashes: Sequence[bytes]) -> set[int]:
    """Ordered 3-gram shingles over the event-hash sequence, as 64-bit ints.

    A trajectory shorter than three events has no shingles and cannot be scored.
    """
    return {
        int.from_bytes(hashlib.sha256(b"".join(event_hashes[i:i + SHINGLE_WIDTH])).digest()[:8], "big")
        for i in range(len(event_hashes) - SHINGLE_WIDTH + 1)
    }


def jaccard(shared: int, size_a: int, size_b: int) -> float:
    """Jaccard of two sets from their sizes and the size of their intersection. 0.0 for two empty sets."""
    union = size_a + size_b - shared
    return shared / union if union else 0.0


# ---------------------------------------------------------------- B2 scan

@dataclass(frozen=True)
class Nearest:
    """The build carrier closest to one query."""

    jaccard: float
    out_path: str
    split_key: str
    n_events: int
    shared_shingles: int

    def beats(self, other: Nearest | None) -> bool:
        """Higher Jaccard wins; a tie goes to the smaller out_path, so the result is order-independent."""
        return (other is None or self.jaccard > other.jaccard
                or (self.jaccard == other.jaccard and self.out_path < other.out_path))


@dataclass(frozen=True)
class QueryIndex:
    """Inverted index from shingle to the queries that contain it. Build with ``build_query_index``."""

    sizes: tuple[int, ...]
    repos: tuple[str, ...]
    postings: Mapping[int, tuple[int, ...]]


def build_query_index(queries: Iterable[tuple[Collection[int], str]]) -> QueryIndex:
    """Index ``(shingle set, split_key)`` queries. Query ids are positions in ``queries``."""
    sizes: list[int] = []
    repos: list[str] = []
    postings: dict[int, list[int]] = collections.defaultdict(list)
    for query_id, (shingle_set, repo) in enumerate(queries):
        if not isinstance(shingle_set, (set, frozenset)):
            raise TypeError("a query's shingles must be a set: repeated shingles would be double counted")
        sizes.append(len(shingle_set))
        repos.append(repo)
        for shingle in shingle_set:
            postings[shingle].append(query_id)
    return QueryIndex(sizes=tuple(sizes), repos=tuple(repos),
                      postings={shingle: tuple(ids) for shingle, ids in postings.items()})


def scan_chunk(
    index: QueryIndex,
    chunk: Iterable[tuple[str, str]],
    load_event_hashes: EventHashLoader,
    exact_events: Collection[bytes] = frozenset(),
) -> tuple[dict[int, Nearest], set[bytes]]:
    """Scan ``(out_path, split_key)`` build carriers against every query.

    Returns the nearest build carrier per query id, and which of ``exact_events`` occur
    verbatim in the chunk. A carrier in the query's own repo is never a candidate. A query
    that shares no shingle with any candidate has no entry: its Jaccard against every
    candidate is 0.0 and no carrier is nearer than any other, so none is named (read it
    with ``nearest_jaccard``). Chunks can be scanned in any order or in parallel and
    folded with ``merge_nearest``.
    """
    best: dict[int, Nearest] = {}
    hits: set[bytes] = set()
    for out_path, split_key in chunk:
        event_hashes = load_event_hashes(out_path)
        if exact_events:
            hits.update(h for h in event_hashes if h in exact_events)
        carrier_shingles = shingles(event_hashes)
        if not carrier_shingles:
            continue
        shared: collections.Counter[int] = collections.Counter()
        for shingle in carrier_shingles:
            query_ids = index.postings.get(shingle)
            if query_ids:
                shared.update(query_ids)
        for query_id, n_shared in shared.items():
            if index.repos[query_id] == split_key:
                continue
            candidate = Nearest(
                jaccard=jaccard(n_shared, index.sizes[query_id], len(carrier_shingles)),
                out_path=out_path, split_key=split_key,
                n_events=len(event_hashes), shared_shingles=n_shared,
            )
            if candidate.beats(best.get(query_id)):
                best[query_id] = candidate
    return best, hits


def merge_nearest(best: dict[int, Nearest], part: Mapping[int, Nearest]) -> None:
    """Fold one chunk's result into ``best`` in place."""
    for query_id, candidate in part.items():
        if candidate.beats(best.get(query_id)):
            best[query_id] = candidate


def nearest_scan(
    index: QueryIndex,
    build: Iterable[tuple[str, str]],
    load_event_hashes: EventHashLoader,
    exact_events: Collection[bytes] = frozenset(),
) -> tuple[dict[int, Nearest], set[bytes]]:
    """Single-process scan of the whole build set. Same result as chunked scans merged."""
    return scan_chunk(index, build, load_event_hashes, exact_events)


def nearest_jaccard(best: Mapping[int, Nearest], query_id: int) -> float:
    """A query's nearest-carrier Jaccard after a scan: 0.0 when it has no entry.

    No entry means no eligible carrier shared a shingle, so the statistic is 0.0 and the
    nearest path, repo and event count are null in the reported row. A query with no
    shingles of its own also has no entry; that case is undefined, not 0.0, and the
    caller tells the two apart from the query's own shingle set before asking here.
    """
    nearest = best.get(query_id)
    return nearest.jaccard if nearest is not None else 0.0


# ---------------------------------------------------------------- verdicts and null

def _thresholds() -> tuple[float, ...]:
    return tuple(sorted(SENSITIVITY + (THRESHOLD,)))


def _label(threshold: float) -> str:
    return f"{threshold:.2f}"


def fires(value: float) -> dict[str, bool]:
    """Whether a Jaccard reaches each reported threshold, keyed '0.30', '0.50', '0.70'."""
    return {_label(t): bool(value >= t) for t in _thresholds()}


def nearest_rank_quantile(sorted_values: Sequence[float], p: float) -> float | None:
    """Nearest-rank quantile of an ascending sequence. None when there is nothing to rank."""
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(p * len(sorted_values)) - 1)]


def null_sample(build_rows: Iterable[Mapping[str, Any]], n: int = NULL_SAMPLE_SIZE) -> list[Mapping[str, Any]]:
    """The first ``n`` build carriers in floor-sample order."""
    return sorted(build_rows, key=lambda r: salted_path_order_key(r["out_path"]))[:n]


@dataclass(frozen=True)
class NullSummary:
    """What the null sample says about the threshold. ``share`` values are None for an empty null."""

    share: Mapping[str, float | None]
    flags: tuple[str, ...]


def summarize_null(scored: Sequence[float]) -> NullSummary:
    """Share of scored null Jaccards at or above each threshold, and the flags that follow.

    SATURATED/NON-GATING: more than 1% of the null reaches the blocking threshold, so a
    cited carrier that fires is not distinguishable from an ordinary one.
    UNTESTED-BY-NULL: no null carrier reaches even the lowest threshold, so the null never
    exercised the region the gate acts on. An empty null raises neither flag.
    """
    share: dict[str, float | None] = {
        _label(t): (sum(v >= t for v in scored) / len(scored) if scored else None) for t in _thresholds()
    }
    flags = []
    if scored and share[_label(THRESHOLD)] > NULL_SATURATED_SHARE:
        flags.append(FLAG_SATURATED)
    if scored and share[_label(min(SENSITIVITY))] == 0:
        flags.append(FLAG_UNTESTED)
    return NullSummary(share=share, flags=tuple(flags))


def b2_verdict(any_cited_fired: bool, null_flags: Collection[str]) -> str:
    """NON-GATING when the null is saturated, else BLOCKER if any cited carrier fired, else PASS."""
    if FLAG_SATURATED in null_flags:
        return VERDICT_NON_GATING
    return VERDICT_BLOCKER if any_cited_fired else VERDICT_PASS
