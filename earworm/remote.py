"""The catalogue stored in Supabase: songs and the in-memory index built from it.

Used by the local indexer (duplicate checks, export) and by the server (T15).
"""

from earworm.db import Supabase, from_bytea
from earworm.fingerprint import PARAMS_VERSION, from_bytes
from earworm.index import Index

SONG_COLUMNS = (
    "id,youtube_id,title,artist,year,mb_recording_id,mb_genres,mb_tags,genre_label,"
    "cover_url,duration_s,source,youtube_title,youtube_channel,created_at"
)


def song_title(song: dict) -> str:
    return f"{song['artist']} - {song['title']}"


def fetch_songs(db: Supabase) -> list[dict]:
    return db.select_all("songs", SONG_COLUMNS)


def build_index(db: Supabase, songs: list[dict] | None = None, page: int = 50) -> Index:
    """Download every fingerprint blob and build the searchable index.

    Raises if some blobs were made with other fingerprint parameters: mixing
    versions would silently break recognition, the catalogue must be re-indexed.
    """
    songs = fetch_songs(db) if songs is None else songs
    titles = {s["id"]: song_title(s) for s in songs}
    rows = db.select_all("fingerprints", "song_id,params_version,data", order="song_id", page=page)
    stale = [r["song_id"] for r in rows if r["params_version"] != PARAMS_VERSION]
    if stale:
        raise ValueError(
            f"{len(stale)} songs were fingerprinted with other parameters than "
            f"PARAMS_VERSION {PARAMS_VERSION} (e.g. song {stale[0]}): re-index them"
        )
    idx = Index()
    for r in rows:
        idx.add(r["song_id"], from_bytes(from_bytea(r["data"])), title=titles.get(r["song_id"]))
    idx.freeze()
    return idx
