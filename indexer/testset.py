"""The measurement test set (docs/spec.md section 7.1; T26).

    python -m indexer eval-draw [--seed 2026] [--force]

100 positives drawn at random from the catalogue, each with a random start
between 20 % and 80 % of its length, plus 20 negatives: chart songs that are
NOT in the catalogue (same era, same style: a fair test of false positives).
The 120 trials are shuffled so the operator cannot tell which are which.

The list is versioned in docs/eval/testset.csv (+ testset.json: seed, date,
catalogue size). The chart import skips the negatives, so they stay absent.
Redrawing (--force, e.g. once the catalogue reaches 1 000 songs) draws new
positives and keeps the existing negatives.
"""

import csv
import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from indexer.fetch import MAX_DURATION_S, MIN_DURATION_S
from indexer.titles import normalize

EVAL_DIR = Path("docs") / "eval"
TESTSET_CSV = EVAL_DIR / "testset.csv"
TESTSET_META = EVAL_DIR / "testset.json"
N_POSITIVES, N_NEGATIVES = 100, 20
EXCERPT_S = 10
START_RANGE = (0.2, 0.8)
DEFAULT_SEED = 2026


@dataclass(frozen=True)
class Trial:
    trial: int
    kind: str  # positive | negative
    song_id: int | None  # catalogue id (positives only)
    youtube_id: str
    artist: str
    title: str
    start_s: int
    duration_s: float

    @property
    def url(self) -> str:
        """The video, starting at the excerpt."""
        return f"https://www.youtube.com/watch?v={self.youtube_id}&t={self.start_s}s"

    @property
    def start(self) -> str:
        return f"{self.start_s // 60}:{self.start_s % 60:02d}"


def _start(duration_s: float, rng: np.random.Generator) -> int:
    lo, hi = START_RANGE
    start = int(rng.uniform(lo, hi) * duration_s)
    return max(0, min(start, int(duration_s) - EXCERPT_S))


def _key(artist: str, title: str) -> tuple[str, str]:
    return normalize(artist), normalize(title)


def negative_pool(songs: list[dict], resolved: dict) -> list[dict]:
    """Chart songs with a video, absent from the catalogue (by video AND by name:
    another video of a catalogue song would really be a positive)."""
    ids = {s["youtube_id"] for s in songs}
    names = {_key(s["artist"], s["title"]) for s in songs}
    pool, seen = [], set()
    for r in sorted(resolved.values(), key=lambda r: (r["chart"], int(r["year"]), int(r["rank"]))):
        vid, dur = r.get("youtube_id"), float(r.get("duration_s") or 0)
        if not vid or vid in ids or vid in seen or _key(r["artist"], r["title"]) in names:
            continue
        if not MIN_DURATION_S + EXCERPT_S <= dur <= MAX_DURATION_S:
            continue
        seen.add(vid)
        pool.append(r)
    return pool


def draw(
    songs: list[dict],
    resolved: dict,
    seed: int = DEFAULT_SEED,
    keep_negatives: list[Trial] | None = None,
    n_pos: int = N_POSITIVES,
    n_neg: int = N_NEGATIVES,
) -> list[Trial]:
    rng = np.random.default_rng(seed)
    usable = sorted(
        (s for s in songs if (s.get("duration_s") or 0) >= MIN_DURATION_S),
        key=lambda s: s["id"],
    )
    if len(usable) < n_pos:
        raise ValueError(f"only {len(usable)} songs in the catalogue, {n_pos} needed")
    chosen = [usable[i] for i in sorted(rng.choice(len(usable), n_pos, replace=False))]
    rows = [
        dict(kind="positive", song_id=s["id"], youtube_id=s["youtube_id"], artist=s["artist"],
             title=s["title"], start_s=_start(float(s["duration_s"]), rng),
             duration_s=float(s["duration_s"]))
        for s in chosen
    ]  # fmt: skip
    in_catalogue = {s["youtube_id"] for s in songs}
    kept = [t for t in keep_negatives or [] if t.youtube_id not in in_catalogue][:n_neg]
    rows += [{k: v for k, v in asdict(t).items() if k != "trial"} for t in kept]
    if len(kept) < n_neg:
        taken = {t.youtube_id for t in kept}
        pool = [r for r in negative_pool(songs, resolved) if r["youtube_id"] not in taken]
        if len(pool) < n_neg - len(kept):
            raise ValueError(f"only {len(pool)} chart songs left outside the catalogue")
        for i in sorted(rng.choice(len(pool), n_neg - len(kept), replace=False)):
            r = pool[i]
            dur = float(r["duration_s"])
            rows.append(
                dict(kind="negative", song_id=None, youtube_id=r["youtube_id"], artist=r["artist"],
                     title=r["title"], start_s=_start(dur, rng), duration_s=dur)
            )  # fmt: skip
    order = rng.permutation(len(rows))
    return [Trial(trial=n, **rows[i]) for n, i in enumerate(order, 1)]


def save(trials: list[Trial], meta: dict, path: Path = TESTSET_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in fields(Trial)], lineterminator="\n")
        w.writeheader()
        for t in trials:
            w.writerow({**asdict(t), "song_id": t.song_id if t.song_id is not None else ""})
    path.with_suffix(".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def load(path: Path = TESTSET_CSV) -> list[Trial]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [
            Trial(
                trial=int(r["trial"]), kind=r["kind"],
                song_id=int(r["song_id"]) if r["song_id"] else None,
                youtube_id=r["youtube_id"], artist=r["artist"], title=r["title"],
                start_s=int(r["start_s"]), duration_s=float(r["duration_s"]),
            )
            for r in csv.DictReader(f)
        ]  # fmt: skip


def negative_ids(path: Path = TESTSET_CSV) -> set[str]:
    """Videos the catalogue must never contain (read by the chart import)."""
    return {t.youtube_id for t in load(path) if t.kind == "negative"}


def meta_for(songs: list[dict], seed: int, params_version: int) -> dict:
    return {
        "seed": seed,
        "drawn_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "catalogue_songs": len(songs),
        "params_version": params_version,
        "positives": N_POSITIVES,
        "negatives": N_NEGATIVES,
        "start_range": list(START_RANGE),
    }
