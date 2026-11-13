"""The server's in-memory catalogue: index + songs, loaded in the background
and swapped in one go when Supabase reports a new catalogue version."""

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from earworm.index import Index

log = logging.getLogger("earworm.store")

Loader = Callable[[], tuple[Index, dict[int, dict], int]]


# Memory the process may use (Render free: 512 MB). Python, numpy and FastAPI
# take ~120 MB; the index is 8 bytes per hash (T17b).
BASE_MB = 120


class CatalogStore:
    def __init__(
        self, loader: Loader, version_fn: Callable[[], int], budget_mb: float | None = None
    ) -> None:
        self._loader = loader
        self.budget_mb = budget_mb  # None = no limit
        self._version_fn = version_fn
        self._lock = threading.Lock()  # one load at a time
        # replaced as a whole: readers always see a consistent (index, songs) pair
        self._state: tuple[Index, dict[int, dict]] | None = None
        self.version: int | None = None
        self.loaded_at: datetime | None = None
        self.load_seconds: float | None = None

    @property
    def ready(self) -> bool:
        return self._state is not None

    @property
    def index(self) -> Index:
        return self._state[0]

    @property
    def songs(self) -> dict[int, dict]:
        return self._state[1]

    def snapshot(self) -> tuple[Index, dict[int, dict]] | None:
        return self._state

    def _fits_twice(self) -> bool:
        """Can the old index keep serving while the new one (about as big) is built?"""
        if self.budget_mb is None or self._state is None:
            return True
        return BASE_MB + 2 * self._state[0].nbytes / 2**20 <= self.budget_mb

    def load(self) -> None:
        with self._lock:
            if not self._fits_twice():
                # better a short "loading" (the app retries) than the process killed
                log.warning("catalogue too big to reload side by side: unloading the old one")
                self._state = None
            t0 = time.perf_counter()
            index, songs, version = self._loader()
            self._state = (index, songs)
            self.version = version
            self.loaded_at = datetime.now(UTC)
            self.load_seconds = time.perf_counter() - t0
        log.info(
            "catalogue v%s loaded: %d songs, %d hashes in %.1f s",
            version, index.n_songs, index.n_hashes, self.load_seconds,
        )  # fmt: skip

    def maybe_reload(self) -> bool:
        """Reload if the catalogue version changed. True if it did."""
        if self.version is not None and self._version_fn() == self.version:
            return False
        self.load()
        return True

    def start(self, every_s: float = 600) -> threading.Thread:
        """Load now, then check for a new version every `every_s` seconds (daemon thread)."""

        def run() -> None:
            while True:
                try:
                    self.maybe_reload()
                except Exception:  # keep serving the old index; retry next round
                    log.exception("catalogue load failed")
                time.sleep(every_s if self.ready else 30)

        t = threading.Thread(target=run, name="catalogue-loader", daemon=True)
        t.start()
        return t


def attach_similar(songs: dict[int, dict], ids: list[int], vectors) -> None:
    """Store each song's similar songs (ids, best first) in songs[id]["similar_ids"]."""
    from earworm.similar import neighbours

    keep = [i for i, sid in enumerate(ids) if sid in songs]  # embeddings of deleted songs
    near = neighbours([ids[i] for i in keep], vectors[keep]) if keep else {}
    for sid, song in songs.items():
        song["similar_ids"] = near.get(sid, [])


def attach_genres(songs: dict[int, dict], rows: list[dict]) -> None:
    """Store each song's genre prediction (API format) in songs[id]["genre"]."""
    for r in rows:
        if r["song_id"] in songs:
            top3 = r["top3"] or []
            songs[r["song_id"]]["genre"] = {
                "label": r["label"],
                "name": top3[0].get("name", r["label"]) if top3 else r["label"],
                "proba": r["proba"],
                "top3": top3,
            }


def supabase_loader(db) -> tuple[Loader, Callable[[], int]]:
    """Loader and version check reading the catalogue from Supabase."""
    import numpy as np

    from earworm.remote import build_index, fetch_songs
    from earworm.similar import parse_vector

    def version() -> int:
        return db.select("catalog_meta", "version", id="eq.1")[0]["version"]

    def load():
        v = version()  # read first: a change during the load is caught next round
        songs = {s["id"]: s for s in fetch_songs(db)}
        index = build_index(db, list(songs.values()))
        rows = db.select_all("embeddings", "song_id,embedding", order="song_id")
        vectors = np.stack([parse_vector(r["embedding"]) for r in rows]) if rows else None
        attach_similar(songs, [r["song_id"] for r in rows], vectors)
        attach_genres(songs, db.select_all("genre_predictions", "song_id,label,proba,top3",
                                           order="song_id"))  # fmt: skip
        return index, songs, v

    return load, version
