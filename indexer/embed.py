"""Catch-up: give an embedding to every catalogue song that has none (T19).

    python -m indexer embed-missing [--cache data/embeddings/t18]

Songs added before the embedding step existed get theirs here. When the
segment vectors of a song are already on disk (saved when it was added, or
computed for the T18 comparison), they are reused (same recipe: 10 s segments,
mean, L2); otherwise the audio is downloaded again into a temporary folder and
deleted right after.

Segment vectors are kept locally (data/embeddings/<model>/<song_id>.npz, never
committed, no audio): the genre classifier is trained on them (T20/T21).
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from earworm.embeddings import embed_segments, song_embedding
from earworm.similar import to_pgvector
from indexer.fetch import downloaded_audio

SEGMENTS_ROOT = Path("data") / "embeddings"
T18_CACHE = SEGMENTS_ROOT / "t18"
T18_KEYS = {"clap-htsat-unfused": "clap", "mert-v1-95m": "mert"}  # model -> array in T18 files


def save_segments(root: Path, model: str, song_id: int, vectors: np.ndarray) -> None:
    folder = Path(root) / model
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{song_id}.npz", segments=np.asarray(vectors, dtype=np.float16))


def load_segments(
    root: Path | None, model: str, song_id: int, t18: Path | None = T18_CACHE
) -> np.ndarray | None:
    """Segment vectors of a song, from the segment store or the T18 cache."""
    if root is not None and (f := Path(root) / model / f"{song_id}.npz").exists():
        with np.load(f) as z:
            return z["segments"].astype(np.float32)
    key = T18_KEYS.get(model)
    if t18 is not None and key and (f := Path(t18) / f"{song_id}.npz").exists():
        with np.load(f) as z:
            return z[key].astype(np.float32)
    return None


def missing_songs(db) -> list[dict]:
    from earworm.remote import fetch_songs

    have = {r["song_id"] for r in db.select_all("embeddings", "song_id", order="song_id")}
    return [s for s in fetch_songs(db) if s["id"] not in have]


def embed_missing(
    db,
    embedder,
    root: Path | None = SEGMENTS_ROOT,
    cache: Path | None = T18_CACHE,
    downloader=None,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    counts = {"cache": 0, "download": 0, "error": 0}
    todo = missing_songs(db)
    out(f"{len(todo)} songs without an embedding")
    for i, song in enumerate(todo, 1):
        label = f"[{i}/{len(todo)}] #{song['id']} {song['artist']} - {song['title']}"
        try:
            segs = load_segments(root, embedder.name, song["id"], t18=cache)
            how = "cache"
            if segs is None:
                url = f"https://www.youtube.com/watch?v={song['youtube_id']}"
                with downloaded_audio(url, downloader=downloader) as audio:
                    segs = embed_segments(audio.path, embedder)
                if root is not None:
                    save_segments(root, embedder.name, song["id"], segs)
                how = "download"
            vector = song_embedding(segs)
            db.insert(
                "embeddings",
                {"song_id": song["id"], "model": embedder.name, "embedding": to_pgvector(vector)},
            )
        except Exception as e:  # one bad video must not stop the catch-up
            counts["error"] += 1
            out(f"{label}: error {type(e).__name__}: {e}"[:300])
            continue
        counts[how] += 1
        out(f"{label}: {how}")
    return counts
