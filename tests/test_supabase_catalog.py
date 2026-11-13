"""T13: the add pipeline writing to Supabase (in-memory fake database)."""

import pytest

from earworm.audio import SAMPLE_RATE
from earworm.db import BYTEA_PREFIX
from earworm.fingerprint import PARAMS_VERSION, fingerprint
from earworm.remote import build_index
from indexer.catalog import SupabaseCatalog
from indexer.pipeline import add_song
from tests.test_cli import song
from tests.test_pipeline import FakeDownloader, FakeMB, url


class FakeSupabase:
    """Just enough PostgREST for SupabaseCatalog: select_all and rpc('add_song')."""

    def __init__(self):
        self.tables = {"songs": [], "fingerprints": []}
        self.next_id = 2  # id 1 was used by db-check, as on the real project
        self.rpc_calls = 0

    def select_all(self, table, columns="*", order="id", page=1000, **filters):
        rows = sorted(self.tables[table], key=lambda r: r[order])
        return [dict(r) for r in rows]

    def iter_pages(self, table, columns="*", order="id", page=1000, **filters):
        rows = self.select_all(table, columns, order)
        for i in range(0, len(rows) + 1, page):
            yield rows[i : i + page]

    def rpc(self, name, args):
        assert name == "add_song"
        self.rpc_calls += 1
        song = dict(args["p_song"], id=self.next_id)
        if any(s["youtube_id"] == song["youtube_id"] for s in self.tables["songs"]):
            raise RuntimeError("duplicate key value violates unique constraint")
        self.next_id += 1
        self.tables["songs"].append(song)
        self.tables["fingerprints"].append(
            {
                "song_id": song["id"],
                "params_version": args["p_params_version"],
                "n_hashes": args["p_n_hashes"],
                "data": BYTEA_PREFIX + args["p_fingerprint_hex"],
            }
        )
        return song


@pytest.fixture
def db():
    return FakeSupabase()


def add(db_or_catalog, vid, **kw):
    catalog = (
        db_or_catalog
        if isinstance(db_or_catalog, SupabaseCatalog)
        else SupabaseCatalog(db_or_catalog)
    )
    return add_song(
        url(vid), catalog, downloader=FakeDownloader(), mb=FakeMB(), interactive=False, **kw
    )


def test_add_writes_song_and_fingerprints_in_one_call(db):
    res = add(db, "AAAAAAAAAAA", source="playlist")
    assert res.status == "added" and res.song["id"] == 2
    assert db.rpc_calls == 1
    (stored,) = db.tables["songs"]
    assert stored["youtube_id"] == "AAAAAAAAAAA" and stored["source"] == "playlist"
    assert "added_at" not in stored and "id" in stored
    (fp,) = db.tables["fingerprints"]
    assert fp["params_version"] == PARAMS_VERSION and fp["n_hashes"] > 1000


def test_index_rebuilt_from_database_recognises(db):
    add(db, "AAAAAAAAAAA")
    add(db, "BBBBBBBBBBB")
    idx = build_index(db)
    assert idx.n_songs == 2
    q = fingerprint(song(2, seconds=60)[30 * SAMPLE_RATE : 40 * SAMPLE_RATE])
    best = idx.query(q)[0]
    assert best.is_confident and idx.title(best.song_id) == "Beta - Second Song"


def test_rerun_creates_no_duplicates(db):
    add(db, "AAAAAAAAAAA")
    again = add(db, "AAAAAAAAAAA")  # fresh catalogue object, as a second run would
    assert again.status == "duplicate_link"
    same_audio = add(db, "CCCCCCCCCCC")
    assert same_audio.status == "duplicate_audio"
    assert len(db.tables["songs"]) == 1 and db.rpc_calls == 1


def test_index_kept_up_to_date_within_a_run(db):
    catalog = SupabaseCatalog(db)
    add(catalog, "AAAAAAAAAAA")
    _ = catalog.index  # downloaded once
    add(catalog, "BBBBBBBBBBB")
    assert catalog.index.n_songs == 2
    assert add(catalog, "CCCCCCCCCCC").status == "duplicate_audio"


def test_stale_fingerprints_are_refused(db):
    add(db, "AAAAAAAAAAA")
    db.tables["fingerprints"][0]["params_version"] = PARAMS_VERSION - 1
    with pytest.raises(ValueError, match="re-index"):
        build_index(db)
