"""Catch-up: give an embedding to every catalogue song that has none (T19).

    python -m indexer embed-missing [--cache data/embeddings/t18]

Songs added before the embedding step existed get theirs here. When the T18
comparison already computed a song's segment vectors with the same model, they
are reused (same recipe: 10 s segments, mean, L2); otherwise the audio is
downloaded again into a temporary folder and deleted right after.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from earworm.embeddings import embed_file, song_embedding
from earworm.similar import to_pgvector
from indexer.fetch import downloaded_audio

T18_CACHE = Path("data") / "embeddings" / "t18"
CACHE_KEYS = {"clap-htsat-unfused": "clap"}  # model name -> array in the T18 files


def missing_songs(db) -> list[dict]:
    from earworm.remote import fetch_songs

    have = {r["song_id"] for r in db.select_all("embeddings", "song_id", order="song_id")}
    return [s for s in fetch_songs(db) if s["id"] not in have]


def cached_embedding(song_id: int, model: str, cache: Path | None) -> np.ndarray | None:
    key = CACHE_KEYS.get(model)
    f = cache / f"{song_id}.npz" if cache else None
    if key is None or f is None or not f.exists():
        return None
    with np.load(f) as z:
        return song_embedding(z[key].astype(np.float32))


def embed_missing(
    db,
    embedder,
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
            vector = cached_embedding(song["id"], embedder.name, cache)
            how = "cache"
            if vector is None:
                url = f"https://www.youtube.com/watch?v={song['youtube_id']}"
                with downloaded_audio(url, downloader=downloader) as audio:
                    vector = embed_file(audio.path, embedder)
                how = "download"
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
