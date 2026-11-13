"""Check a Supabase project against T12's acceptance criteria.

    python -m indexer db-check

1. With the PUBLISHABLE key: no table can be read, nothing can be written.
2. With the SECRET key: a test song + fingerprint can be written, read back
   (blob intact), the catalogue version is bumped, and everything is removed.
"""

import numpy as np

from earworm.db import Supabase, from_bytea, to_bytea
from earworm.fingerprint import PARAMS_VERSION, to_bytes

TABLES = ("songs", "fingerprints", "embeddings", "genre_predictions", "proposals", "catalog_meta")
TEST_ID = "TESTdbcheck"  # 11 characters, like a YouTube id; never a real video


def _refused(status: int, payload) -> bool:
    """A refusal is an HTTP error, or an empty result (RLS hiding every row)."""
    return not 200 <= status < 300 or payload == []


def check_public_key(public: Supabase) -> list[tuple[str, bool, str]]:
    out = []
    for table in TABLES:
        status, payload = public.raw("GET", table, {"select": "*", "limit": "1"})
        out.append((f"public key cannot read {table}", _refused(status, payload), f"HTTP {status}"))
    status, payload = public.raw(
        "POST", "proposals", body={"youtube_id": TEST_ID, "client_hash": "dbcheck"}
    )
    out.append(("public key cannot write a proposal", not 200 <= status < 300, f"HTTP {status}"))
    return out


def check_secret_key(secret: Supabase) -> list[tuple[str, bool, str]]:
    out = []
    v0 = secret.select("catalog_meta", "version", id="eq.1")[0]["version"]
    out.append(("secret key reads catalog_meta", True, f"version {v0}"))
    secret.delete("songs", youtube_id=f"eq.{TEST_ID}")  # leftover of an interrupted run
    blob = to_bytes(np.array([[5, 0], [9, 3]], dtype=np.uint32))
    try:
        song = secret.insert(
            "songs",
            {
                "youtube_id": TEST_ID,
                "title": "db-check",
                "artist": "Earworm",
                "duration_s": 1.0,
                "source": "manual",
            },
        )[0]
        out.append(("secret key writes a song", True, f"id {song['id']}"))
        secret.insert(
            "fingerprints",
            {
                "song_id": song["id"],
                "params_version": PARAMS_VERSION,
                "n_hashes": 2,
                "data": to_bytea(blob),
            },
        )
        back = secret.select("fingerprints", "data", song_id=f"eq.{song['id']}")[0]["data"]
        out.append(("fingerprint blob round-trips", from_bytea(back) == blob, f"{len(blob)} bytes"))
        v1 = secret.select("catalog_meta", "version", id="eq.1")[0]["version"]
        out.append(("catalogue version bumped on insert", v1 == v0 + 1, f"{v0} -> {v1}"))
    finally:
        secret.delete("songs", youtube_id=f"eq.{TEST_ID}")
    left = secret.select("fingerprints", "song_id", song_id=f"eq.{song['id']}")
    out.append(("delete cascades to fingerprints", left == [], ""))
    v2 = secret.select("catalog_meta", "version", id="eq.1")[0]["version"]
    out.append(("catalogue version bumped on delete", v2 == v1 + 1, f"{v1} -> {v2}"))
    return out


def run(url: str, secret_key: str, public_key: str) -> list[tuple[str, bool, str]]:
    return check_public_key(Supabase(url, public_key)) + check_secret_key(Supabase(url, secret_key))
