"""T22: genre prediction (numpy model from T21), pipeline step, catch-up, API."""

import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from earworm.genre import GenreModel
from earworm.index import Index
from indexer.catalog import LocalCatalog, SupabaseCatalog
from indexer.embed import save_segments
from indexer.pipeline import add_song
from indexer.predict import coverage, predict_missing
from server.app import create_app
from server.store import CatalogStore, attach_genres
from tests.test_pipeline import FakeDownloader, FakeMB, url
from tests.test_similar import FakeEmbedder
from tests.test_supabase_catalog import FakeSupabase

CLASSES = ["dance", "rock", "jazz-blues"]
NAMES = {"dance": "Électro / dance", "rock": "Rock", "jazz-blues": "Jazz / blues"}


def write_model(path, dim=8, hidden=None, embed_model="fake-model", seed=0):
    """A genre_model.npz in the T21 export format (logreg, or MLP if hidden)."""
    rng = np.random.default_rng(seed)
    sizes = [dim, hidden, len(CLASSES)] if hidden else [dim, len(CLASSES)]
    Ws = [rng.normal(size=(a, b)) for a, b in zip(sizes, sizes[1:], strict=False)]
    bs = [rng.normal(size=b) for b in sizes[1:]]
    np.savez(
        path, kind="mlp" if hidden else "logreg", mean=rng.normal(size=dim),
        scale=rng.uniform(0.5, 2, dim), classes=np.array(CLASSES), n_layers=len(Ws),
        **{f"W{i}": w for i, w in enumerate(Ws)}, **{f"b{i}": b for i, b in enumerate(bs)},
        info=json.dumps({"embed_model": embed_model, "taxonomy_version": 1, "classes": NAMES}),
    )  # fmt: skip
    return Ws, bs


def test_logreg_prediction_is_the_softmax_of_the_mean(tmp_path):
    Ws, bs = write_model(tmp_path / "m.npz")
    m = GenreModel.load(tmp_path / "m.npz")
    X = np.random.default_rng(1).normal(size=(5, 8))
    z = np.load(tmp_path / "m.npz")
    logits = ((X - z["mean"]) / z["scale"]) @ Ws[0] + bs[0]
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    assert np.allclose(m.probas(X), p, atol=1e-5)
    pred = m.predict(X)
    assert pred.label == CLASSES[int(np.argmax(p.mean(0)))]
    assert pred.proba == pytest.approx(p.mean(0).max(), abs=1e-5)
    assert [t["label"] for t in pred.top3][0] == pred.label and len(pred.top3) == 3
    assert pred.top3[0]["name"] == NAMES[pred.label]


def test_mlp_has_a_relu_between_layers(tmp_path):
    Ws, bs = write_model(tmp_path / "m.npz", hidden=4)
    m = GenreModel.load(tmp_path / "m.npz")
    X = np.random.default_rng(2).normal(size=(3, 8))
    z = np.load(tmp_path / "m.npz")
    h = np.maximum(((X - z["mean"]) / z["scale"]) @ Ws[0] + bs[0], 0) @ Ws[1] + bs[1]
    p = np.exp(h - h.max(1, keepdims=True))
    assert np.allclose(m.probas(X), p / p.sum(1, keepdims=True), atol=1e-5)


def test_model_version_traces_kind_embedding_taxonomy_and_file(tmp_path):
    write_model(tmp_path / "a.npz", seed=0)
    write_model(tmp_path / "b.npz", seed=1)
    a, b = GenreModel.load(tmp_path / "a.npz"), GenreModel.load(tmp_path / "b.npz")
    assert a.version.startswith("logreg-fake-model-tax1-") and a.version != b.version


def test_pipeline_predicts_the_genre_of_a_new_song(tmp_path):
    write_model(tmp_path / "m.npz")
    model = GenreModel.load(tmp_path / "m.npz")
    db = FakeSupabase()
    kw = dict(downloader=FakeDownloader(), mb=FakeMB(), interactive=False)
    res = add_song(url("AAAAAAAAAAA"), SupabaseCatalog(db, tmp_path / "seg"),
                   embedder=FakeEmbedder(), genre_model=model, **kw)  # fmt: skip
    (row,) = db.tables["genre_predictions"]
    assert row["song_id"] == res.song["id"] and row["model_version"] == model.version
    assert row["label"] in CLASSES and "genre" in res.detail
    local = LocalCatalog(tmp_path / "local")
    res = add_song(url("BBBBBBBBBBB"), local, embedder=FakeEmbedder(), genre_model=model, **kw)
    stored = LocalCatalog(tmp_path / "local").by_id(res.song["id"])
    assert stored["genre_prediction"]["model_version"] == model.version


def test_catch_up_and_coverage(tmp_path):
    write_model(tmp_path / "m.npz")
    model = GenreModel.load(tmp_path / "m.npz")
    db = FakeSupabase()
    kw = dict(downloader=FakeDownloader(), mb=FakeMB(), interactive=False)
    cat = SupabaseCatalog(db, tmp_path / "seg")
    a = add_song(url("AAAAAAAAAAA"), cat, **kw).song
    b = add_song(url("BBBBBBBBBBB"), cat, **kw).song
    save_segments(tmp_path / "seg", "fake-model", a["id"], np.ones((3, 8)))
    c = coverage(db)
    assert c["songs"] == 2 and c["without_genre"] == [a["id"], b["id"]]
    lines = []
    counts = predict_missing(db, model, root=tmp_path / "seg", t18=None, out=lines.append)
    assert counts == {"predicted": 1, "no_segments": 1}
    assert coverage(db)["without_genre"] == [b["id"]]
    again = predict_missing(db, model, root=tmp_path / "seg", t18=None, out=lines.append)
    assert again["predicted"] == 0  # same model version: nothing to redo
    redo = predict_missing(db, model, redo_all=True, root=tmp_path / "seg", t18=None,
                           out=lines.append)  # fmt: skip
    assert redo["predicted"] == 1 and len(db.tables["genre_predictions"]) == 1


def test_api_returns_the_stored_genre():
    songs = {1: {"id": 1, "title": "t", "artist": "a", "year": None, "youtube_id": "x" * 11,
                 "cover_url": None}}  # fmt: skip
    top3 = [{"label": "rock", "name": "Rock", "proba": 0.7}]
    attach_genres(songs, [{"song_id": 1, "label": "rock", "proba": 0.7, "top3": top3},
                          {"song_id": 99, "label": "pop", "proba": 0.5, "top3": []}])  # fmt: skip
    idx = Index()
    idx.freeze()
    store = CatalogStore(lambda: (idx, songs, 1), version_fn=lambda: 1)
    store.load()
    genre = TestClient(create_app(store)).get("/api/songs/1").json()["genre"]
    assert genre == {"label": "rock", "name": "Rock", "proba": 0.7, "top3": top3}


@pytest.mark.skipif(os.environ.get("EARWORM_LIVE") != "1", reason="live: set EARWORM_LIVE=1")
def test_live_every_song_has_an_embedding_and_a_genre():
    """Spec section 7.3, against the real Supabase project (.env)."""
    from earworm.config import load_env, require
    from earworm.db import Supabase

    load_env()
    cfg = require("SUPABASE_URL", "SUPABASE_SECRET_KEY")
    c = coverage(Supabase(cfg["SUPABASE_URL"], cfg["SUPABASE_SECRET_KEY"]))
    assert c["without_embedding"] == [] and c["without_genre"] == []
