"""In-memory fingerprint index and offset-histogram matching.

Every hash of every song is kept in three parallel arrays sorted by hash
(hash, song_id, anchor frame). A query looks each of its hashes up with a
binary search, and for every hit computes  db_frame - query_frame : for the
right song, the hits pile up on a single value (where the excerpt starts in
the song), while chance hits spread out. The score of a song is the height of
that pile, with +-1 frame tolerance for excerpts not aligned on the frame grid.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earworm.fingerprint import PARAMS_VERSION, frame_to_seconds

# Decision thresholds: a best match counts as "found" only if both hold.
# Set on the T07 bench (docs/results.md): the best of 999 absent songs reaches
# a score of 12 by chance less than 1 % of the time (extrapolated); x2 margin.
# Tied to PARAMS_VERSION: re-tune if the fingerprint parameters change.
MIN_SCORE = 24
MIN_RATIO = 2.0

_SHIFT = 1 << 20  # makes (db_frame - query_frame) non-negative
_STRIDE = 1 << 21  # key = song_id * _STRIDE + diff + _SHIFT


@dataclass(frozen=True)
class Match:
    song_id: int
    score: int  # aligned hashes (with +-1 frame tolerance)
    ratio: float  # score / best score among the *other* songs
    offset: int  # frame of the song where the excerpt starts

    @property
    def offset_s(self) -> float:
        return float(frame_to_seconds(self.offset))

    @property
    def is_confident(self) -> bool:
        return self.score >= MIN_SCORE and self.ratio >= MIN_RATIO


class Index:
    def __init__(self) -> None:
        self._parts: list[tuple[int, np.ndarray]] = []
        self._song_ids: set[int] = set()
        self._titles: dict[int, str | None] = {}
        self._hash = np.zeros(0, dtype=np.uint32)
        self._song = np.zeros(0, dtype=np.uint32)
        self._frame = np.zeros(0, dtype=np.uint32)
        self._frozen = False

    def add(self, song_id: int, hashes: np.ndarray, title: str | None = None) -> None:
        """Add a song's (hash, anchor_frame) array. Call freeze() afterwards."""
        if song_id in self._song_ids:
            raise ValueError(f"song {song_id} already in index")
        if not 0 <= song_id < (1 << 32):  # stored as uint32; keys are int64
            raise ValueError("song_id out of range")
        self._song_ids.add(song_id)
        self._titles[song_id] = title
        self._parts.append((song_id, np.asarray(hashes, dtype=np.uint32)))
        self._frozen = False

    def freeze(self) -> None:
        """Build the sorted arrays. Needed before query()."""
        if self._parts:
            new_hash = np.concatenate([h[:, 0] for _, h in self._parts])
            new_frame = np.concatenate([h[:, 1] for _, h in self._parts])
            new_song = np.concatenate(
                [np.full(len(h), sid, dtype=np.uint32) for sid, h in self._parts]
            )
            hash_ = np.concatenate([self._hash, new_hash])
            song = np.concatenate([self._song, new_song])
            frame = np.concatenate([self._frame, new_frame])
            order = np.argsort(hash_, kind="stable")
            self._hash, self._song, self._frame = hash_[order], song[order], frame[order]
            self._parts = []
        self._frozen = True

    def title(self, song_id: int) -> str | None:
        return self._titles.get(song_id)

    def save(self, path: str | Path) -> None:
        """Save a frozen index to a .npz file (local use; Supabase comes in T13)."""
        if not self._frozen:
            raise RuntimeError("call freeze() before save()")
        meta = {
            "params_version": PARAMS_VERSION,
            "titles": {str(k): v for k, v in self._titles.items()},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez(
                f,
                hash=self._hash,
                song=self._song,
                frame=self._frame,
                meta=np.array(json.dumps(meta)),
            )

    @classmethod
    def load(cls, path: str | Path) -> "Index":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta["params_version"] != PARAMS_VERSION:
                raise ValueError(
                    f"index built with params_version {meta['params_version']}, "
                    f"code is at {PARAMS_VERSION}: re-index the songs"
                )
            idx = cls()
            idx._hash, idx._song, idx._frame = data["hash"], data["song"], data["frame"]
        idx._titles = {int(k): v for k, v in meta["titles"].items()}
        idx._song_ids = set(idx._titles)
        idx._frozen = True
        return idx

    @property
    def n_songs(self) -> int:
        return len(self._song_ids)

    @property
    def n_hashes(self) -> int:
        return len(self._hash) + sum(len(h) for _, h in self._parts)

    def query(self, hashes: np.ndarray, top_k: int = 5) -> list[Match]:
        """Best matching songs for a query's (hash, frame) array, best first."""
        if not self._frozen:
            raise RuntimeError("call freeze() before query()")
        q = np.asarray(hashes, dtype=np.uint32)
        if len(q) == 0 or len(self._hash) == 0:
            return []

        # 1. all index rows whose hash appears in the query
        lo = np.searchsorted(self._hash, q[:, 0], side="left")
        hi = np.searchsorted(self._hash, q[:, 0], side="right")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            return []
        q_idx = np.repeat(np.arange(len(q)), counts)
        starts = np.repeat(lo - (np.cumsum(counts) - counts), counts)
        rows = starts + np.arange(total)

        # 2. histogram of (song, db_frame - query_frame)
        diff = self._frame[rows].astype(np.int64) - q[q_idx, 1].astype(np.int64)
        keys = self._song[rows].astype(np.int64) * _STRIDE + diff + _SHIFT
        uk, c = np.unique(keys, return_counts=True)

        # 3. +-1 frame tolerance: add the neighbouring bins of the same song
        smoothed = c.copy()
        for step in (-1, 1):
            pos = np.searchsorted(uk, uk + step)
            pos_ok = np.minimum(pos, len(uk) - 1)
            hit = (pos < len(uk)) & (uk[pos_ok] == uk + step)
            smoothed[hit] += c[pos_ok[hit]]

        # 4. best bin per song (highest smoothed, then highest raw count)
        song_of = uk // _STRIDE
        order = np.lexsort((-c, -smoothed, song_of))
        first = np.ones(len(order), dtype=bool)
        first[1:] = song_of[order][1:] != song_of[order][:-1]
        best = order[first]
        best = best[np.argsort(-smoothed[best], kind="stable")]

        scores = smoothed[best]
        top_score = int(scores[0])
        second = int(scores[1]) if len(scores) > 1 else 0
        results = []
        for rank, i in enumerate(best[:top_k]):
            other = second if rank == 0 else top_score
            results.append(
                Match(
                    song_id=int(song_of[i]),
                    score=int(smoothed[i]),
                    ratio=float(smoothed[i] / max(1, other)),
                    offset=int(uk[i] % _STRIDE - _SHIFT),
                )
            )
        return results
