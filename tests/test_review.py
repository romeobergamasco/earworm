"""T25: `python -m indexer review` (spec section 2.3), no network."""

from indexer.fetch import VideoInfo
from indexer.musicbrainz import Candidate
from indexer.pipeline import AddResult
from indexer.review import Preview, hint_of, open_proposals, preview, review, watch_url


class FakeRest:
    """Just enough PostgREST: select (status=in.(...), order) and update (id=eq.N)."""

    def __init__(self, rows):
        self.rows = [dict(r) for r in rows]

    def select(self, table, columns="*", **filters):
        assert table == "proposals"
        wanted = filters["status"].removeprefix("in.(").removesuffix(")").split(",")
        rows = [r for r in self.rows if r["status"] in wanted]
        return sorted(rows, key=lambda r: r[filters.get("order", "id")])

    def update(self, table, values, **filters):
        pid = int(filters["id"].removeprefix("eq."))
        (row,) = [r for r in self.rows if r["id"] == pid]
        row.update(values)
        return [row]

    def by_id(self, pid):
        return next(r for r in self.rows if r["id"] == pid)


def proposal(pid, vid, status="pending", title=None, artist=None):
    return {
        "id": pid,
        "youtube_id": vid,
        "title_hint": title,
        "artist_hint": artist,
        "status": status,
        "created_at": f"2026-11-15T10:{pid:02d}:00+00:00",
    }


def answers(*values):
    it = iter(values)
    return lambda prompt: next(it)


def quiet_preview(p):
    return Preview(
        "Some video",
        "Some channel",
        200.0,
        "Artist - Title (2001), confidence 0.95",
        query=("Artist", "Title"),
    )


SONG = {"id": 501, "artist": "Artist", "title": "Title", "year": 2001}


def test_accept_indexes_and_records_the_song():
    db = FakeRest([proposal(1, "aaaaaaaaaaa", title="Titel", artist="Artsit")])
    calls = []

    def add(url, hint=None):
        calls.append((url, hint))
        assert db.by_id(1)["status"] == "accepted"  # marked before the long pipeline
        return AddResult("added", SONG, "MusicBrainz auto (0.95)")

    counts = review(db, add, quiet_preview, ask=answers("a"), out=lambda s: None)
    assert counts["indexed"] == 1
    assert calls == [(watch_url("aaaaaaaaaaa"), ("Artist", "Title"))]  # the previewed query
    row = db.by_id(1)
    assert row["status"] == "indexed" and row["song_id"] == 501 and row["reviewed_at"]


def test_reject_needs_a_reason():
    db = FakeRest([proposal(1, "aaaaaaaaaaa")])
    counts = review(
        db,
        add=lambda *a, **k: 1 / 0,
        show_preview=quiet_preview,
        ask=answers("x", "r", "  ", "not   a song,\nonly a podcast"),
        out=lambda s: None,
    )
    assert counts["rejected"] == 1
    row = db.by_id(1)
    assert row["status"] == "rejected" and row["reason"] == "not a song, only a podcast"


def test_failure_is_recorded_with_the_error():
    db = FakeRest([proposal(1, "aaaaaaaaaaa")])

    def add(url, hint=None):
        raise RuntimeError("video too long (95 min)")

    counts = review(db, add, quiet_preview, ask=answers("a"), out=lambda s: None)
    assert counts["failed"] == 1
    row = db.by_id(1)
    assert row["status"] == "failed" and "too long" in row["reason"]


def test_duplicate_audio_is_rejected_with_the_existing_song():
    db = FakeRest([proposal(1, "aaaaaaaaaaa")])
    add = lambda url, hint=None: AddResult("duplicate_audio", SONG, "same audio")  # noqa: E731
    review(db, add, quiet_preview, ask=answers("a"), out=lambda s: None)
    row = db.by_id(1)
    assert row["status"] == "rejected" and row["song_id"] == 501
    assert "already in the catalogue" in row["reason"]


def test_skip_and_quit_leave_proposals_pending():
    db = FakeRest(
        [proposal(1, "aaaaaaaaaaa"), proposal(2, "bbbbbbbbbbb"), proposal(3, "ccccccccccc")]
    )
    counts = review(
        db, lambda *a, **k: 1 / 0, quiet_preview, ask=answers("s", "q"), out=lambda s: None
    )
    assert counts["skipped"] == 3
    assert [r["status"] for r in db.rows] == ["pending"] * 3
    assert all("reviewed_at" not in r for r in db.rows)


def test_only_open_proposals_oldest_first():
    db = FakeRest(
        [
            proposal(3, "ccccccccccc"),
            proposal(1, "aaaaaaaaaaa", status="indexed"),
            proposal(2, "bbbbbbbbbbb", status="accepted"),  # interrupted review
            proposal(4, "ddddddddddd", status="rejected"),
        ]
    )
    assert [p["id"] for p in open_proposals(db)] == [2, 3]


def test_hint_only_when_both_are_given():
    assert hint_of({"artist_hint": "A", "title_hint": "T"}) == ("A", "T")
    assert hint_of({"artist_hint": None, "title_hint": "T"}) is None


class FakeDownloader:
    def __init__(self, fail=False):
        self.fail = fail

    def info(self, youtube_id):
        if self.fail:
            raise RuntimeError("Video unavailable")
        return VideoInfo(
            youtube_id, "Daft Punk - One More Time (Official Video)", "Daft Punk", 320.0
        )


class FakeMB:
    """Confident only for the right spelling."""

    def __init__(self):
        self.searched = []

    def search(self, artist, track):
        self.searched.append((artist, track))
        conf = 0.97 if (artist, track) == ("Daft Punk", "One More Time") else 0.6
        return [Candidate("rid", "One More Time", "Daft Punk", 2000, 100, None, confidence=conf)]


def test_preview_from_the_youtube_title():
    mb = FakeMB()
    pv = preview(proposal(1, "aaaaaaaaaaa"), FakeDownloader(), mb)
    assert pv.youtube_title.startswith("Daft Punk") and pv.duration_s == 320.0
    assert pv.match == "Daft Punk - One More Time (2000), confidence 0.97"
    assert mb.searched == [("Daft Punk", "One More Time")] and pv.query == mb.searched[0]


def test_a_typo_in_the_visitor_hint_does_not_win():
    mb = FakeMB()
    pv = preview(
        proposal(1, "aaaaaaaaaaa", title="One More Time", artist="Daft Punkk"), FakeDownloader(), mb
    )
    assert len(mb.searched) == 2
    assert pv.query == ("Daft Punk", "One More Time") and "0.97" in pv.match


def test_the_visitor_hint_wins_when_youtube_says_little():
    class Vague(FakeDownloader):
        def info(self, youtube_id):
            return VideoInfo(youtube_id, "Official Video HD", "SomeUploader", 320.0)

    pv = preview(
        proposal(1, "aaaaaaaaaaa", title="One More Time", artist="Daft Punk"), Vague(), FakeMB()
    )
    assert pv.query == ("Daft Punk", "One More Time") and "0.97" in pv.match


def test_preview_of_an_unavailable_video():
    pv = preview(proposal(1, "aaaaaaaaaaa"), FakeDownloader(fail=True), FakeMB())
    assert pv.youtube_title == "?" and "unavailable" in pv.error
