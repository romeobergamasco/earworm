"""Training data for the genre classifier (T20 -> T21 on Colab).

    python -m indexer export-genre-data [--model clap-htsat-unfused]
                                        [--out data/genre_dataset.npz]

One row per 10 s segment: its embedding, its song, the song's artist and
genre class. Songs without a class are exported too (label "") so that the
notebook can predict them; no audio, no secret. Upload the file to Colab.
"""

import json
from pathlib import Path

import numpy as np

from indexer.embed import SEGMENTS_ROOT, T18_CACHE, load_segments
from indexer.genres import label, taxonomy


def build(songs: list[dict], model: str, root=SEGMENTS_ROOT, t18=T18_CACHE) -> dict:
    tax = taxonomy()
    X, seg_song, missing, table = [], [], [], []
    for s in songs:
        segs = load_segments(root, model, s["id"], t18=t18)
        if segs is None:
            missing.append(s["id"])
            continue
        X.append(segs)
        seg_song += [s["id"]] * len(segs)
        lab = label(s.get("mb_genres") or [], s.get("mb_tags") or [], tax)
        table.append({
            "id": s["id"], "artist": s["artist"], "title": s["title"], "year": s.get("year"),
            "label": lab.label or "", "clear": lab.clear,
        })  # fmt: skip
    by_id = {t["id"]: t for t in table}
    meta = {
        "model": model,
        "taxonomy_version": tax.version,
        "classes": tax.names,
        "songs": table,
        "missing_segments": missing,
    }
    return {
        "X": np.concatenate(X).astype(np.float32) if X else np.zeros((0, 0), np.float32),
        "song_id": np.array(seg_song, dtype=np.int64),
        "label": np.array([by_id[i]["label"] for i in seg_song]),
        "artist": np.array([by_id[i]["artist"] for i in seg_song]),
        "meta": np.array(json.dumps(meta, ensure_ascii=False)),
    }


def export(songs: list[dict], model: str, out: Path, **kw) -> dict:
    data = build(songs, model, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return data
