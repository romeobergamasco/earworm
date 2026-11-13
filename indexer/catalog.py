"""Local catalogue: song records (catalog.json) + their fingerprints (index.npz).

Temporary home until Supabase (T13). Both files are rewritten atomically, so
an interrupted run never leaves a half-written catalogue.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from earworm.index import Index

DEFAULT_ROOT = Path("data") / "catalog"


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


class LocalCatalog:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.json_path = self.root / "catalog.json"
        self.index_path = self.root / "index.npz"
        self.embeddings_path = self.root / "embeddings.npz"
        self.songs: list[dict] = []
        if self.json_path.exists():
            self.songs = json.loads(self.json_path.read_text(encoding="utf-8"))["songs"]
        self._index: Index | None = None

    @property
    def index(self) -> Index:
        if self._index is None:
            if self.index_path.exists():
                self._index = Index.load(self.index_path)
            else:
                self._index = Index()
                self._index.freeze()
        return self._index

    def by_youtube_id(self, youtube_id: str) -> dict | None:
        return next((s for s in self.songs if s["youtube_id"] == youtube_id), None)

    def by_id(self, song_id: int) -> dict | None:
        return next((s for s in self.songs if s["id"] == song_id), None)

    def embeddings(self) -> tuple[list[int], np.ndarray, str | None]:
        """(song ids, vectors, model) of the songs that have an embedding."""
        if not self.embeddings_path.exists():
            return [], np.zeros((0, 0), dtype=np.float32), None
        with np.load(self.embeddings_path, allow_pickle=False) as z:
            return [int(i) for i in z["ids"]], z["vectors"], str(z["model"])

    def add(self, record: dict, hashes: np.ndarray, embedding=None) -> dict:
        """Give the record an id, store it, its fingerprints and its embedding
        ((model, vector) or None), save to disk."""
        song = {
            "id": max((s["id"] for s in self.songs), default=0) + 1,
            **record,
            "added_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        idx = self.index
        idx.add(song["id"], hashes, title=f"{song['artist']} - {song['title']}")
        idx.freeze()
        self.root.mkdir(parents=True, exist_ok=True)
        self.songs.append(song)
        _atomic_write(self.index_path, idx.save)
        if embedding is not None:
            model, vector = embedding
            ids, vectors, _ = self.embeddings()
            vectors = np.vstack([vectors, vector[None]]) if ids else vector[None]
            arrays = {"ids": np.array([*ids, song["id"]]), "vectors": vectors,
                      "model": np.array(model)}  # fmt: skip
            _atomic_write(self.embeddings_path, lambda f: np.savez(f, **arrays))
        payload = json.dumps({"songs": self.songs}, ensure_ascii=False, indent=1)
        _atomic_write(self.json_path, lambda f: f.write(payload.encode("utf-8")))
        return song


class SupabaseCatalog:
    """Same interface as LocalCatalog, backed by Supabase (spec section 4).

    Songs are read once at start; the index is downloaded only when needed (for
    the acoustic duplicate check) and then kept up to date in memory.
    """

    def __init__(self, db) -> None:
        from earworm.remote import fetch_songs

        self.db = db
        self.songs: list[dict] = fetch_songs(db)
        self._index: Index | None = None

    @property
    def index(self) -> Index:
        if self._index is None:
            from earworm.remote import build_index

            self._index = build_index(self.db, self.songs)
        return self._index

    def by_youtube_id(self, youtube_id: str) -> dict | None:
        return next((s for s in self.songs if s["youtube_id"] == youtube_id), None)

    def by_id(self, song_id: int) -> dict | None:
        return next((s for s in self.songs if s["id"] == song_id), None)

    def add(self, record: dict, hashes: np.ndarray, embedding=None) -> dict:
        """Song + fingerprints (+ embedding: (model, vector) or None) in one
        database transaction (db/002_add_song.sql, db/003_embeddings.sql)."""
        from earworm.fingerprint import PARAMS_VERSION, to_bytes
        from earworm.remote import song_title
        from earworm.similar import to_pgvector

        columns = {
            "youtube_id",
            "title",
            "artist",
            "year",
            "mb_recording_id",
            "mb_genres",
            "mb_tags",
            "genre_label",
            "cover_url",
            "duration_s",
            "source",
            "youtube_title",
            "youtube_channel",
        }
        args = {
            "p_song": {k: v for k, v in record.items() if k in columns},
            "p_fingerprint_hex": to_bytes(hashes).hex(),
            "p_params_version": PARAMS_VERSION,
            "p_n_hashes": int(len(hashes)),
        }
        if embedding is not None:  # needs db/003_embeddings.sql
            args["p_model"], args["p_embedding"] = embedding[0], to_pgvector(embedding[1])
        song = self.db.rpc("add_song", args)
        self.songs.append(song)
        if self._index is not None:
            self._index.add(song["id"], hashes, title=song_title(song))
            self._index.freeze()
        return song
