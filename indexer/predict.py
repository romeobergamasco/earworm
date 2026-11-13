"""Genre predictions of the catalogue (T22).

    python -m indexer predict-genres [--model models/genre_model.npz] [--all]

Each song's prediction comes from its segment vectors kept on disk
(indexer/embed.py) and the classifier exported by T21 (earworm/genre.py). By
default only songs without a prediction from this model version are done;
--all redoes every song. Songs whose segment vectors are missing are listed:
run `embed-missing` first (it keeps the vectors of the songs it downloads).
"""

from collections.abc import Callable
from pathlib import Path

from earworm.genre import GenreModel
from indexer.embed import SEGMENTS_ROOT, T18_CACHE, load_segments


def predict_missing(
    db,
    model: GenreModel,
    redo_all: bool = False,
    root: Path | None = SEGMENTS_ROOT,
    t18: Path | None = T18_CACHE,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    from earworm.remote import fetch_songs

    done = {
        r["song_id"]
        for r in db.select_all("genre_predictions", "song_id,model_version", order="song_id")
        if r["model_version"] == model.version
    }
    songs = [s for s in fetch_songs(db) if redo_all or s["id"] not in done]
    counts = {"predicted": 0, "no_segments": 0}
    rows = []
    for s in songs:
        segs = load_segments(root, model.embed_model, s["id"], t18=t18)
        if segs is None:
            counts["no_segments"] += 1
            out(f"#{s['id']} {s['artist']} - {s['title']}: no segment vectors (embed-missing)")
            continue
        rows.append(model.predict(segs).row(s["id"], model.version))
        counts["predicted"] += 1
    for i in range(0, len(rows), 200):
        db.upsert("genre_predictions", rows[i : i + 200], on_conflict="song_id")
    out(f"model {model.version}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return counts


def coverage(db) -> dict:
    """Songs without an embedding or a genre prediction (spec section 7.3)."""
    from earworm.remote import fetch_songs

    ids = {s["id"] for s in fetch_songs(db)}
    emb = {r["song_id"] for r in db.select_all("embeddings", "song_id", order="song_id")}
    gen = {r["song_id"] for r in db.select_all("genre_predictions", "song_id", order="song_id")}
    return {
        "songs": len(ids),
        "without_embedding": sorted(ids - emb),
        "without_genre": sorted(ids - gen),
    }
