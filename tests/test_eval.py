"""T26: measurement tooling (test set, eval endpoint, sessions, report, bench)."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import fingerprint
from earworm.index import Index
from indexer import evaluation as ev
from indexer import testset
from indexer.testset import Trial
from server.app import create_app
from server.evals import MemoryEvals
from server.store import CatalogStore
from tests.test_cli import song
from tests.test_server import loader


# ------------------------------------------------------------------ test set
def catalogue(n=150):
    return [
        {"id": i, "youtube_id": f"cat{i:08d}", "artist": f"Artist {i}", "title": f"Song {i}",
         "duration_s": 120.0 + i}
        for i in range(1, n + 1)
    ]  # fmt: skip


def charts(n=60, songs=()):
    rows = {}
    for i in range(n):
        rows[("us", 1990, i + 1)] = {
            "chart": "us", "year": "1990", "rank": str(i + 1), "artist": f"Chart {i}",
            "title": f"Hit {i}", "youtube_id": f"hit{i:08d}", "duration_s": "200.0",
        }  # fmt: skip
    for k, s in enumerate(songs):  # chart songs that ARE in the catalogue
        rows[("fr", 2000, k + 1)] = {
            "chart": "fr", "year": "2000", "rank": str(k + 1), "artist": s["artist"],
            "title": s["title"], "youtube_id": s["youtube_id"], "duration_s": "200.0",
        }  # fmt: skip
    return rows


def test_draw_counts_ranges_and_determinism():
    songs = catalogue()
    a = testset.draw(songs, charts(), seed=7)
    assert len(a) == 120 and sorted(t.trial for t in a) == list(range(1, 121))
    pos = [t for t in a if t.kind == "positive"]
    neg = [t for t in a if t.kind == "negative"]
    assert len(pos) == 100 and len(neg) == 20
    assert len({t.song_id for t in pos}) == 100
    for t in pos:
        assert 0.2 * t.duration_s - 1 <= t.start_s <= 0.8 * t.duration_s
        assert t.start_s + testset.EXCERPT_S <= t.duration_s
    assert testset.draw(songs, charts(), seed=7) == a
    assert testset.draw(songs, charts(), seed=8) != a
    assert [t.kind for t in a[:20]] != ["positive"] * 20  # shuffled


def test_negatives_are_absent_by_video_and_by_name():
    songs = catalogue()
    resolved = charts(n=25, songs=songs[:30])
    resolved[("us", 1991, 1)] = {  # another video of a catalogue song: not a negative
        "chart": "us", "year": "1991", "rank": "1", "artist": "ARTIST 5", "title": "Song 5!",
        "youtube_id": "othervideo1", "duration_s": "200",
    }  # fmt: skip
    pool = testset.negative_pool(songs, resolved)
    assert {r["youtube_id"] for r in pool} == {f"hit{i:08d}" for i in range(25)}


def test_redraw_keeps_negatives(tmp_path):
    songs = catalogue()
    first = testset.draw(songs, charts(), seed=1)
    path = tmp_path / "testset.csv"
    testset.save(first, {"seed": 1}, path)
    assert testset.load(path) == first
    assert testset.negative_ids(path) == {t.youtube_id for t in first if t.kind == "negative"}
    kept = [t for t in first if t.kind == "negative"]
    again = testset.draw(songs + catalogue(200)[150:], charts(), seed=2, keep_negatives=kept)
    assert {t.youtube_id for t in again if t.kind == "negative"} == {t.youtube_id for t in kept}


def test_too_small_catalogue():
    with pytest.raises(ValueError):
        testset.draw(catalogue(50), charts())


def test_chart_import_skips_negatives(tmp_path):
    from indexer.charts import ChartEntry
    from indexer.importer import run_import

    class Cat:
        songs: list = []

        def by_youtube_id(self, vid):
            return None

    added = []

    def add(url, catalog, **kw):
        added.append(url)
        raise RuntimeError("stop")

    entries = [ChartEntry("us", 1990, i + 1, f"Chart {i}", f"Hit {i}") for i in range(3)]
    run_import(Cat(), entries, charts(3), target=10, log_path=tmp_path / "log.csv", add=add,
               out=lambda *_: None, exclude={"hit00000001"})  # fmt: skip
    assert [u[-11:] for u in added] == ["hit00000000", "hit00000002"]


# ------------------------------------------------------------------ server endpoint
@pytest.fixture
def evals():
    box = MemoryEvals()
    box.open("run-abcdefgh")
    return box


@pytest.fixture
def client(evals):
    store = CatalogStore(loader, version_fn=lambda: 3)
    store.load()
    return TestClient(create_app(store, evals=evals))


GOOD = {
    "match": True, "song_id": 7, "score": 88, "ratio": 6.5, "latency_ms": 812.4,
    "resample_ms": 30, "network_ms": 700, "server": {"fingerprint": 150, "match": 30, "total": 190},
    "capture_rate": 48000,
}  # fmt: skip


def test_eval_result_is_recorded(client, evals):
    r = client.post("/api/eval/run-abcdefgh", json=GOOD)
    assert r.status_code == 201
    (row,) = evals.rows
    assert row["run_id"] == "run-abcdefgh" and row["latency_ms"] == 812
    assert row["server_ms"] == {"fingerprint": 150, "match": 30, "total": 190}
    r = client.post("/api/eval/run-abcdefgh", json={"match": False, "latency_ms": 900})
    assert r.status_code == 201 and evals.rows[-1]["song_id"] is None


@pytest.mark.parametrize("run_id", ["nope-nope-nope", "short", "bad id with spaces"])
def test_eval_unknown_session(client, evals, run_id):
    assert client.post(f"/api/eval/{run_id}", json=GOOD).status_code == 404
    assert evals.rows == []


def test_eval_closed_or_old_session(client, evals):
    evals.open("old-session", created_at=datetime.now(UTC) - timedelta(days=3))
    assert client.post("/api/eval/old-session", json=GOOD).status_code == 404
    evals.runs["run-abcdefgh"]["closed_at"] = "2026-11-15T10:00:00+00:00"
    assert client.post("/api/eval/run-abcdefgh", json=GOOD).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"latency_ms": 100},
        {"match": "yes", "latency_ms": 100},
        {"match": True, "latency_ms": 100},  # no song_id
        {"match": False},
        {"match": False, "latency_ms": -5},
        {"match": False, "latency_ms": True},
        {"match": False, "latency_ms": 100, "ratio": "big"},
        {"match": False, "latency_ms": 100, "server": [1, 2]},
        [1, 2, 3],
    ],
)
def test_eval_invalid_body(client, evals, body):
    assert client.post("/api/eval/run-abcdefgh", json=body).status_code == 422
    assert evals.rows == []


def test_eval_too_large(client):
    r = client.post(
        "/api/eval/run-abcdefgh", content=b"{" + b" " * 5000 + b"}",
        headers={"Content-Type": "application/json"},
    )  # fmt: skip
    assert r.status_code == 413


def test_eval_not_configured():
    store = CatalogStore(loader, version_fn=lambda: 3)
    store.load()
    assert (
        TestClient(create_app(store)).post("/api/eval/run-abcdefgh", json=GOOD).status_code == 503
    )


# ------------------------------------------------------------------ statistics
def test_wilson_matches_the_spec_example():
    lo, hi = ev.wilson(90, 100)
    assert round(lo, 3) == 0.826 and round(hi, 3) == 0.945
    assert ev.wilson(0, 20)[0] == 0 and 0.16 < ev.wilson(0, 20)[1] < 0.17
    assert ev.wilson(0, 0) == (0.0, 1.0)


def test_percentiles():
    assert ev.percentiles(list(range(1, 101))) == pytest.approx((50.5, 95.05))
    assert ev.percentiles([]) is None


# ------------------------------------------------------------------ sessions
class FakeRest:
    """eval_results in memory: select (run_id, discarded, id>), update (id=eq.)."""

    def __init__(self):
        self.rows = []

    def push(self, **values):
        row = {"id": len(self.rows) + 1, "run_id": "r1", "discarded": False, "trial": None,
               "expected_kind": None, "expected_song_id": None, "score": None, "ratio": None,
               "server_ms": None, "network_ms": None, **values}  # fmt: skip
        self.rows.append(row)
        return row

    def select(self, table, columns="*", **f):
        after = int(f.get("id", "gt.0").removeprefix("gt."))
        return [
            r for r in self.rows
            if r["run_id"] == f["run_id"].removeprefix("eq.") and not r["discarded"]
            and r["id"] > after
        ]  # fmt: skip

    def update(self, table, values, **f):
        rid = int(f["id"].removeprefix("eq."))
        next(r for r in self.rows if r["id"] == rid).update(values)


def trial(n, kind="positive", song_id=None):
    return Trial(n, kind, song_id, f"vid{n:08d}", f"A{n}", f"T{n}", 61, 200.0)


def scripted(db, answers_per_trial):
    """A `wait` that makes the PWA 'send' the next scripted result."""
    it = iter(answers_per_trial)

    def wait(db_, run_id, after):
        return db.push(**next(it))

    return wait


def test_session_pairs_results_with_trials_and_redoes():
    db = FakeRest()
    trials = [trial(1, song_id=7), trial(2, "negative"), trial(3, song_id=9)]
    wait = scripted(
        db,
        [
            {"match": True, "song_id": 9, "latency_ms": 900},  # wrong song -> redone
            {"match": True, "song_id": 7, "latency_ms": 800},
            {"match": False, "song_id": None, "latency_ms": 700},
            {"match": False, "song_id": None, "latency_ms": 750},
        ],
    )
    keys = iter(["r", "", "", ""])
    counts = ev.session(db, "r1", trials, {7: "A - Seven", 9: "B - Nine"},
                        ask=lambda _: next(keys), out=lambda _: None, wait=wait)  # fmt: skip
    assert counts == {"right": 2, "wrong": 1, "redone": 1}
    assert db.rows[0]["discarded"] is True
    assert [(r["trial"], r["expected_kind"], r["expected_song_id"]) for r in db.rows[1:]] == [
        (1, "positive", 7), (2, "negative", None), (3, "positive", 9),
    ]  # fmt: skip


def test_session_resumes_after_quit():
    db = FakeRest()
    trials = [trial(1, song_id=7), trial(2, song_id=8)]
    wait = scripted(db, [{"match": True, "song_id": 7, "latency_ms": 800}])
    ev.session(db, "r1", trials, {}, ask=lambda _: "q", out=lambda _: None, wait=wait)
    shown = []
    wait = scripted(db, [{"match": False, "song_id": None, "latency_ms": 800}])
    ev.session(db, "r1", trials, {}, ask=lambda _: "", out=shown.append, wait=wait)
    assert any("n° 2" in line for line in shown) and not any("n° 1" in line for line in shown)


# ------------------------------------------------------------------ report
def row(trial_no, kind, expected, match, got, latency, total=200):
    return {"id": trial_no, "trial": trial_no, "expected_kind": kind, "expected_song_id": expected,
            "match": match, "song_id": got, "latency_ms": latency, "network_ms": latency - 100,
            "server_ms": {"total": total}, "score": 50, "ratio": 4.0}  # fmt: skip


def test_report_counts_and_markdown():
    rows = (
        [row(i, "positive", i, True, i, 800 + i) for i in range(1, 9)]  # 8 right
        + [row(9, "positive", 9, False, None, 900), row(10, "positive", 10, True, 3, 950)]
        + [row(11, "negative", None, False, None, 700), row(12, "negative", None, True, 4, 720)]
    )
    s = ev.summarize(rows)
    assert (s["positives"], s["right"], s["not_found"], s["wrong_song"]) == (10, 8, 1, 1)
    assert (s["negatives"], s["false_positives"]) == (2, 1)
    run = {"id": "r1abcdefgh", "kind": "mic", "note": "salon", "created_at": "2026-11-15T18:00"}
    text = ev.report_markdown(
        run,
        rows,
        {3: "C - Three", 10: "J - Ten"},
        {"commit": "abc1234", "params_version": 3, "catalogue_songs": 303},
    )
    assert "**8/10 = 80 %**" in text and "[49 % ; 94 %]" in text
    assert "1/2 = 50 %" in text and "abc1234" in text and "salon" in text
    assert "| 10 | J - Ten | C - Three |" in text and "non trouvé" in text


def test_save_report(tmp_path):
    p = ev.save_report("abcdefgh1234", "# x\n", folder=tmp_path)
    assert p.name.endswith("_abcdefgh.md") and p.read_text(encoding="utf-8") == "# x\n"


# ------------------------------------------------------------------ excerpts and bench
def test_cut_keeps_ten_seconds_as_int16():
    y = np.linspace(-1, 1, 60 * SAMPLE_RATE).astype(np.float32)
    clip = ev.cut(y, 20, SAMPLE_RATE)
    assert clip.dtype == np.int16 and len(clip) == 10 * SAMPLE_RATE
    with pytest.raises(ValueError):
        ev.cut(y, 55, SAMPLE_RATE)


def test_bench_on_excerpts(tmp_path):
    a, b, absent = song(1, seconds=60), song(2, seconds=60), song(3, seconds=60)
    idx = Index()
    idx.add(1, fingerprint(a))
    idx.add(2, fingerprint(b))
    idx.freeze()
    trials = [Trial(1, "positive", 1, "aaaaaaaaaaa", "A", "a", 20, 60.0),
              Trial(2, "positive", 2, "bbbbbbbbbbb", "B", "b", 30, 60.0),
              Trial(3, "negative", None, "ccccccccccc", "C", "c", 25, 60.0)]  # fmt: skip
    for t, y in zip(trials, (a, b, absent), strict=True):
        np.save(ev.clip_path(t, tmp_path), ev.cut(y, t.start_s, SAMPLE_RATE))
    res = ev.bench(trials, idx, snrs=(None, 10), folder=tmp_path)
    assert res["clips"] == 3
    clean = res["per_snr"][None]
    assert (clean["right"], clean["positives"], clean["fp"], clean["negatives"]) == (2, 2, 0, 1)
    assert "sans bruit" in ev.bench_markdown(res) and "| 10 dB |" in ev.bench_markdown(res)


def test_keep_clips_downloads_cuts_and_deletes(tmp_path):
    import soundfile as sf

    from indexer.fetch import VideoInfo

    class Downloader:
        def info(self, youtube_id):
            return VideoInfo(youtube_id, "t", "c", 60.0)

        def download(self, youtube_id, out_dir):
            p = out_dir / f"{youtube_id}.wav"
            sf.write(p, song(1, seconds=60), SAMPLE_RATE)
            return p

    t = Trial(1, "positive", 1, "aaaaaaaaaaa", "A", "a", 20, 60.0)
    folder = tmp_path / "clips"
    counts = ev.keep_clips([t], downloader=Downloader(), folder=folder, out=lambda _: None)
    assert counts == {"kept": 1, "already": 0, "error": 0}
    assert ev.load_clip(t, folder).shape == (10 * SAMPLE_RATE,)
    again = ev.keep_clips([t], downloader=Downloader(), folder=folder, out=lambda _: None)
    assert again == {"kept": 0, "already": 1, "error": 0}
    assert not list(tmp_path.glob("*.wav"))


def test_chart_import_patient_mode(tmp_path):
    from indexer.charts import ChartEntry
    from indexer.importer import run_import

    class Cat:
        songs: list = []

        def by_youtube_id(self, vid):
            return None

    def add(url, catalog, **kw):
        raise RuntimeError("Sign in to confirm you're not a bot")

    slept = []
    entries = [ChartEntry("us", 1990, i + 1, f"Chart {i}", f"Hit {i}") for i in range(10)]
    counts = run_import(Cat(), entries, charts(10), target=99, log_path=tmp_path / "log.csv",
                        add=add, out=lambda *_: None, exclude=set(), pause_s=90,
                        block_wait_s=1800, max_blocks=3, sleep=slept.append)  # fmt: skip
    assert counts == {"error": 3}  # stopped after 3 blocks in a row
    assert slept == [1800, 90, 1800, 90]  # wait after each block, pause before each download
