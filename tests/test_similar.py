"""T19: song embeddings, similar songs, embedding step of the pipeline, catch-up."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from earworm.embeddings import segment_starts, segments, song_embedding
from earworm.index import Index
from earworm.similar import neighbours, parse_vector, to_pgvector
from indexer.catalog import LocalCatalog, SupabaseCatalog
from indexer.embed import embed_missing
from indexer.pipeline import add_song
from server.app import create_app
from server.store import CatalogStore, attach_similar
from tests.test_pipeline import FakeDownloader, FakeMB, url
from tests.test_supabase_catalog import FakeSupabase


class FakeEmbedder:
    """Deterministic 'embedding' of the audio: energies of 8 frequency bands."""

    sr = 11025
    name = "fake-model"

    def __init__(self):
        self.calls = 0

    def embed(self, segs):
        self.calls += 1
        spec = np.abs(np.fft.rfft(segs, axis=1))
        bands = np.stack([b.sum(axis=1) for b in np.array_split(spec, 8, axis=1)], axis=1)

        class Out:
            vectors = bands.astype(np.float32) + 1e-3
            seconds = 0.0

        return Out()


# ------------------------------------------------------------------ neighbours
def clustered(rng, n_groups=4, per_group=5, dim=32):
    centres = rng.normal(size=(n_groups, dim))
    X = np.concatenate([c + 0.1 * rng.normal(size=(per_group, dim)) for c in centres])
    ids = list(range(100, 100 + len(X)))
    return ids, X.astype(np.float32), np.repeat(range(n_groups), per_group)


def test_a_song_is_its_own_nearest_neighbour_before_exclusion():
    ids, X, _ = clustered(np.random.default_rng(0))
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    assert (np.argmax(Xn @ Xn.T, axis=1) == np.arange(len(ids))).all()  # itself first...
    near = neighbours(ids, X)
    assert all(sid not in near[sid] for sid in ids)  # ...then excluded


def test_neighbours_are_the_songs_of_the_same_group():
    ids, X, group = clustered(np.random.default_rng(1))
    near = neighbours(ids, X, k=4)
    for i, sid in enumerate(ids):
        assert len(near[sid]) == 4
        assert {group[ids.index(j)] for j in near[sid]} == {group[i]}


def test_tiny_catalogues():
    assert neighbours([], np.zeros((0, 3))) == {}
    assert neighbours([1], np.ones((1, 3))) == {1: []}
    assert neighbours([1, 2], np.eye(2, 3)) == {1: [2], 2: [1]}


def test_pgvector_round_trip():
    v = np.random.default_rng(2).normal(size=512).astype(np.float32)
    assert np.allclose(parse_vector(to_pgvector(v)), v, rtol=1e-5)
    assert parse_vector([1, 2]).dtype == np.float32


def test_neighbours_of_1000_songs_are_fast():
    X = np.random.default_rng(3).normal(size=(1000, 512)).astype(np.float32)
    t0 = time.perf_counter()
    near = neighbours(list(range(1000)), X)
    assert time.perf_counter() - t0 < 2.0 and len(near[0]) == 5


# ------------------------------------------------------------------ segments
def test_segments_cover_the_song_and_skip_intro_and_outro():
    starts = segment_starts(300.0)
    assert len(starts) == 6 and starts[0] == pytest.approx(30) and starts[-1] == pytest.approx(260)
    assert segment_starts(45.0) == [pytest.approx(17.5)] and segment_starts(5.0) == [0.0]
    segs = segments(np.ones(5 * 100), 100)  # 5 s song: one zero-padded 10 s segment
    assert segs.shape == (1, 1000) and segs[0, 500:].sum() == 0


def test_song_embedding_is_unit_length_and_ignores_segment_scale():
    segs = np.random.default_rng(4).normal(size=(6, 16))
    e = song_embedding(segs)
    assert np.linalg.norm(e) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(song_embedding(segs * np.arange(1, 7)[:, None]), e, atol=1e-6)


# ------------------------------------------------------------------ pipeline
def test_pipeline_stores_the_embedding_locally(tmp_path):
    catalog, emb = LocalCatalog(tmp_path / "c"), FakeEmbedder()
    kw = dict(downloader=FakeDownloader(), mb=FakeMB(), interactive=False, embedder=emb)
    a = add_song(url("AAAAAAAAAAA"), catalog, **kw).song
    b = add_song(url("BBBBBBBBBBB"), catalog, **kw).song
    dup = add_song(url("CCCCCCCCCCC"), catalog, **kw)  # same audio as A
    assert dup.status == "duplicate_audio" and emb.calls == 2  # no embedding for a duplicate
    ids, vectors, model = LocalCatalog(tmp_path / "c").embeddings()
    assert ids == [a["id"], b["id"]] and vectors.shape == (2, 8) and model == "fake-model"


def test_pipeline_sends_the_embedding_with_the_song():
    db = FakeSupabase()
    kw = dict(downloader=FakeDownloader(), mb=FakeMB(), interactive=False)
    add_song(url("AAAAAAAAAAA"), SupabaseCatalog(db), embedder=FakeEmbedder(), **kw)
    add_song(url("BBBBBBBBBBB"), SupabaseCatalog(db), **kw)  # --no-embed: old RPC arguments
    (row,) = db.tables["embeddings"]
    assert row["model"] == "fake-model" and parse_vector(row["embedding"]).shape == (8,)
    assert db.rpc_calls == 2


# ------------------------------------------------------------------ catch-up
def test_embed_missing_uses_the_cache_then_downloads(tmp_path):
    db = FakeSupabase()
    kw = dict(downloader=FakeDownloader(), mb=FakeMB(), interactive=False)
    a = add_song(url("AAAAAAAAAAA"), SupabaseCatalog(db), **kw).song
    b = add_song(url("BBBBBBBBBBB"), SupabaseCatalog(db), **kw).song
    emb = FakeEmbedder()
    emb.name = "clap-htsat-unfused"  # the model whose T18 vectors are cached
    np.savez(tmp_path / f"{a['id']}.npz", clap=np.ones((3, 8), dtype=np.float16))
    lines = []
    counts = embed_missing(db, emb, cache=tmp_path, downloader=FakeDownloader(), out=lines.append)
    assert counts == {"cache": 1, "download": 1, "error": 0} and emb.calls == 1
    assert {r["song_id"] for r in db.tables["embeddings"]} == {a["id"], b["id"]}
    again = embed_missing(db, emb, cache=tmp_path, downloader=FakeDownloader(), out=lines.append)
    assert again == {"cache": 0, "download": 0, "error": 0}


# ------------------------------------------------------------------ server
def test_api_returns_five_similar_songs():
    ids, X, group = clustered(np.random.default_rng(5), n_groups=2, per_group=6)
    songs = {
        sid: {"id": sid, "title": f"t{sid}", "artist": f"a{sid}", "year": None,
              "youtube_id": "x" * 11, "cover_url": None}
        for sid in ids
    }  # fmt: skip
    attach_similar(songs, ids + [999], np.vstack([X, X[:1]]))  # 999: embedding of a deleted song
    idx = Index()
    idx.freeze()
    store = CatalogStore(lambda: (idx, songs, 1), version_fn=lambda: 1)
    store.load()
    c = TestClient(create_app(store))
    t0 = time.perf_counter()
    body = c.get(f"/api/songs/{ids[0]}").json()
    assert time.perf_counter() - t0 < 0.1
    similar = [s["id"] for s in body["similar"]]
    assert len(similar) == 5 and ids[0] not in similar and 999 not in similar
    assert {group[ids.index(i)] for i in similar} == {group[0]}
    assert "similar_ids" not in body["song"]
