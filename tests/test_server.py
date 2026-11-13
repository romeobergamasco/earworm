"""T15: HTTP API (FastAPI TestClient, synthetic catalogue, no network)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import fingerprint
from earworm.index import Index
from server.app import create_app
from server.store import CatalogStore
from tests.test_cli import song

SONGS = {
    7: {
        "id": 7,
        "title": "First Song",
        "artist": "Alpha",
        "year": 1999,
        "youtube_id": "AAAAAAAAAAA",
        "cover_url": None,
    },
    9: {
        "id": 9,
        "title": "Second Song",
        "artist": "Beta",
        "year": None,
        "youtube_id": "BBBBBBBBBBB",
        "cover_url": "https://coverartarchive.org/release-group/x/front-250",
    },
}
AUDIO = {7: song(1, seconds=60), 9: song(2, seconds=60)}


def loader():
    idx = Index()
    for sid, y in AUDIO.items():
        idx.add(sid, fingerprint(y), title=f"{SONGS[sid]['artist']} - {SONGS[sid]['title']}")
    idx.freeze()
    return idx, SONGS, 3


@pytest.fixture(scope="module")
def client():
    store = CatalogStore(loader, version_fn=lambda: 3)
    store.load()
    return TestClient(create_app(store))


def pcm(y: np.ndarray) -> bytes:
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


def post(client, body: bytes, rate: str = "11025"):
    return client.post(
        "/api/identify",
        content=body,
        headers={"Content-Type": "application/octet-stream", "X-Sample-Rate": rate},
    )


def test_identify_finds_the_song(client):
    clip = AUDIO[9][20 * SAMPLE_RATE : 30 * SAMPLE_RATE]
    r = post(client, pcm(clip))
    assert r.status_code == 200
    body = r.json()
    assert body["match"] is True
    assert body["song"]["id"] == 9 and body["song"]["title"] == "Second Song"
    assert body["song"]["cover_url"].endswith("front-250")
    assert body["confidence"]["score"] >= 30 and abs(body["confidence"]["offset_s"] - 20) < 0.2
    assert body["genre"] is None and body["similar"] == []  # T19 / T22
    assert set(body["timing_ms"]) == {"fingerprint", "match", "total"}


def test_identify_unknown_audio(client):
    r = post(client, pcm(song(42, seconds=10)))
    assert r.status_code == 200 and r.json()["match"] is False


@pytest.mark.parametrize(
    "body, rate, status",
    [
        (b"\x00\x00" * SAMPLE_RATE * 2, "11025", 400),  # 2 s: too short
        (b"\x00\x00" * SAMPLE_RATE * 16, "11025", 413),  # 16 s: over the size limit
        (b"\x00\x00" * SAMPLE_RATE * 5, "44100", 400),  # wrong sample rate
        (b"\x00" * (SAMPLE_RATE * 10 + 1), "11025", 400),  # odd byte count
        (b"\x00\x00" * SAMPLE_RATE * 5, "abc", 400),
    ],
)
def test_identify_rejects_bad_input(client, body, rate, status):
    assert post(client, body, rate).status_code == status


def test_song_page(client):
    r = client.get("/api/songs/7")
    assert r.status_code == 200 and r.json()["song"]["artist"] == "Alpha"
    assert client.get("/api/songs/12345").status_code == 404


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["songs"] == 2 and body["catalog_version"] == 3
    assert body["hashes"] > 0 and body["index_loaded_at"]


def test_warming_up_before_the_index_is_loaded():
    c = TestClient(create_app(CatalogStore(loader, version_fn=lambda: 3)))
    assert c.get("/api/health").json()["status"] == "loading"
    assert post(c, pcm(AUDIO[7][: 10 * SAMPLE_RATE])).status_code == 503


def test_reload_only_when_the_catalogue_version_changes():
    calls = []
    version = {"v": 3}

    def counting_loader():
        calls.append(1)
        idx, songs, _ = loader()
        return idx, songs, version["v"]

    store = CatalogStore(counting_loader, version_fn=lambda: version["v"])
    store.load()
    assert store.maybe_reload() is False and len(calls) == 1
    version["v"] = 4
    assert store.maybe_reload() is True and len(calls) == 2 and store.version == 4


def test_cors_only_for_the_allowed_origin():
    store = CatalogStore(loader, version_fn=lambda: 3)
    store.load()
    c = TestClient(create_app(store, allowed_origins=["https://someone-earworm.static.hf.space"]))
    ok = c.options(
        "/api/identify",
        headers={
            "Origin": "https://someone-earworm.static.hf.space",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-sample-rate",
        },
    )
    assert (
        ok.headers.get("access-control-allow-origin") == "https://someone-earworm.static.hf.space"
    )
    other = c.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers
