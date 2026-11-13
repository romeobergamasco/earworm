"""In-memory fingerprint index and offset-histogram matching.

Every hash of every song is one uint64 row  hash << 32 | song << 16 | frame
(song = position of the song in the index, frame = anchor frame), and the rows
are kept sorted, i.e. grouped by hash: 8 bytes per hash, and the server can
fill and sort the array in place, without the copies that made the first
version (three uint32 arrays) peak above Render's 512 MB (T17b).

A query looks each of its hashes up with a binary search, and for every hit
computes  db_frame - query_frame : for the right song, the hits pile up on a
single value (where the excerpt starts in the song), while chance hits spread
out. The score of a song is the height of that pile, with +-1 frame tolerance
for excerpts not aligned on the frame grid.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earworm.fingerprint import PARAMS_VERSION, frame_to_seconds

# Decision thresholds: a best match counts as "found" only if both hold.
# T08 (docs/results.md): simulating catalogues of 10 to 5000 absent songs from the
# measured chance scores, score >= 30 with ratio >= 2 gives <= 0.01 % false
# positives at every size. Tied to PARAMS_VERSION: re-tune if fingerprints change.
MIN_SCORE = 30
MIN_RATIO = 2.0

_SHIFT = 1 << 20  # makes (db_frame - query_frame) non-negative
_STRIDE = 1 << 21  # key = song_id * _STRIDE + diff + _SHIFT

# row layout: hash (26 bits used) << 32 | song position (16 bits) << 16 | frame (16 bits)
MAX_SONGS = 1 << 16
MAX_FRAME = (1 << 16) - 1  # = the blob format's uint16 frames, ~50 min of audio
_LOW16 = np.uint64(0xFFFF)


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
        self._parts: list[np.ndarray] = []  # packed rows of songs added since freeze()
        self._ids: list[int] = []  # song position -> song_id
        self._song_ids: set[int] = set()
        self._titles: dict[int, str | None] = {}
        self._rows = np.zeros(0, dtype=np.uint64)
        self._id_of = np.zeros(0, dtype=np.int64)
        self._frozen = False

    def add(self, song_id: int, hashes: np.ndarray, title: str | None = None) -> None:
        """Add a song's (hash, anchor_frame) array. Call freeze() afterwards."""
        if song_id in self._song_ids:
            raise ValueError(f"song {song_id} already in index")
        if not 0 <= song_id < (1 << 32):  # keys of the histogram are int64
            raise ValueError("song_id out of range")
        if len(self._ids) >= MAX_SONGS:
            raise ValueError(f"index full ({MAX_SONGS} songs)")
        h = np.asarray(hashes).reshape(-1, 2)
        if len(h) and (h[:, 1].max() > MAX_FRAME or h.min() < 0 or h[:, 0].max() >= 1 << 32):
            raise ValueError("hash or frame out of range")
        pos = np.uint64(len(self._ids))
        self._parts.append(
            (h[:, 0].astype(np.uint64) << np.uint64(32))
            | (pos << np.uint64(16))
            | h[:, 1].astype(np.uint64)
        )
        self._ids.append(song_id)
        self._song_ids.add(song_id)
        self._titles[song_id] = title
        self._frozen = False

    def freeze(self) -> None:
        """Sort the rows. Needed before query().

        The parts are moved one by one into a single array (each freed once
        copied) and sorted in place: peak memory stays close to the final size.
        """
        if self._parts:
            rows = np.empty(len(self._rows) + sum(len(p) for p in self._parts), dtype=np.uint64)
            rows[: len(self._rows)] = self._rows
            at = len(self._rows)
            self._rows = rows
            while self._parts:
                part = self._parts.pop(0)
                rows[at : at + len(part)] = part
                at += len(part)
                del part
            rows.sort()  # in place
        self._id_of = np.array(self._ids, dtype=np.int64)
        self._frozen = True

    def title(self, song_id: int) -> str | None:
        return self._titles.get(song_id)

    def save(self, path: str | Path) -> None:
        """Save a frozen index to a .npz file (local use)."""
        if not self._frozen:
            raise RuntimeError("call freeze() before save()")
        meta = {
            "params_version": PARAMS_VERSION,
            "titles": {str(k): v for k, v in self._titles.items()},
        }
        arrays = {"rows": self._rows, "ids": self._id_of, "meta": np.array(json.dumps(meta))}
        if hasattr(path, "write"):  # an open binary file
            np.savez(path, **arrays)
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez(f, **arrays)

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
            if "rows" in data:
                idx._rows, idx._id_of = data["rows"], data["ids"]
            else:  # files saved before T17b: three uint32 arrays
                ids, pos = np.unique(data["song"], return_inverse=True)
                idx._rows = (
                    (data["hash"].astype(np.uint64) << np.uint64(32))
                    | (pos.astype(np.uint64) << np.uint64(16))
                    | data["frame"].astype(np.uint64)
                )
                idx._rows.sort()
                idx._id_of = ids.astype(np.int64)
        idx._ids = [int(i) for i in idx._id_of]
        idx._titles = {int(k): v for k, v in meta["titles"].items()}
        idx._song_ids = set(idx._titles)
        idx._frozen = True
        return idx

    @property
    def n_songs(self) -> int:
        return len(self._song_ids)

    @property
    def n_hashes(self) -> int:
        return len(self._rows) + sum(len(p) for p in self._parts)

    @property
    def nbytes(self) -> int:
        return int(self._rows.nbytes)

    def query(self, hashes: np.ndarray, top_k: int = 5) -> list[Match]:
        """Best matching songs for a query's (hash, frame) array, best first."""
        if not self._frozen:
            raise RuntimeError("call freeze() before query()")
        q = np.asarray(hashes, dtype=np.uint32)
        if len(q) == 0 or len(self._rows) == 0:
            return []

        # 1. all index rows whose hash appears in the query
        first_row = q[:, 0].astype(np.uint64) << np.uint64(32)
        lo = np.searchsorted(self._rows, first_row, side="left")
        hi = np.searchsorted(self._rows, first_row + np.uint64(1 << 32), side="left")
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            return []
        q_idx = np.repeat(np.arange(len(q)), counts)
        starts = np.repeat(lo - (np.cumsum(counts) - counts), counts)
        rows = starts + np.arange(total)

        # 2. histogram of (song, db_frame - query_frame)
        hit = self._rows[rows]
        frame = (hit & _LOW16).astype(np.int64)
        song = self._id_of[((hit >> np.uint64(16)) & _LOW16).astype(np.intp)]
        diff = frame - q[q_idx, 1].astype(np.int64)
        keys = song * _STRIDE + diff + _SHIFT
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
