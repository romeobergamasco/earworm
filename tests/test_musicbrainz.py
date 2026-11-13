"""T10: MusicBrainz client, offline (fake HTTP)."""

import json
from urllib.parse import parse_qs, urlparse

import pytest

from indexer.musicbrainz import (
    Candidate,
    MusicBrainz,
    RateLimiter,
    confirm,
    score_candidates,
    user_agent,
)


def rec(rid, title, artist, date="", score=100, disamb="", releases=None):
    return {
        "id": rid,
        "title": title,
        "score": score,
        "disambiguation": disamb,
        "first-release-date": date,
        "artist-credit": [{"name": artist, "joinphrase": ""}],
        "releases": releases or [],
    }


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def clock(self):
        return self.now

    def sleep(self, s):
        self.slept.append(s)
        self.now += s


def test_rate_limiter_spaces_calls():
    fc = FakeClock()
    lim = RateLimiter(1.1, clock=fc.clock, sleep=fc.sleep)
    lim.wait()
    fc.now += 0.3
    lim.wait()
    assert fc.slept == [pytest.approx(0.8)]
    fc.now += 5
    lim.wait()
    assert len(fc.slept) == 1


def test_user_agent_names_app_without_personal_data(monkeypatch):
    monkeypatch.delenv("EARWORM_CONTACT", raising=False)
    ua = user_agent()
    assert ua.startswith("Earworm/") and "@" not in ua
    monkeypatch.setenv("EARWORM_CONTACT", "https://github.com/someone/earworm")
    assert "github.com/someone/earworm" in user_agent()


def test_original_beats_live_and_later_reissue():
    cands = [
        Candidate("live", "Billie Jean", "Michael Jackson", 1988, 100, None, "live, Wembley"),
        Candidate("reissue", "Billie Jean", "Michael Jackson", 2008, 100, None),
        Candidate("orig", "Billie Jean", "Michael Jackson", 1982, 100, None),
        Candidate("other", "Billie Jean", "The Bates", 1995, 100, None),
    ]
    best = score_candidates(cands, "Michael Jackson", "Billie Jean")
    assert best[0].recording_id == "orig"
    assert best[0].confidence >= 0.95
    assert {c.recording_id: c.confidence for c in best}["live"] < 0.7


def test_featured_artists_in_credit_still_match():
    cands = [Candidate("d", "Despacito", "Luis Fonsi feat. Daddy Yankee", 2017, 100, None)]
    assert score_candidates(cands, "Luis Fonsi", "Despacito")[0].confidence >= 0.95


def test_wrong_song_gets_low_confidence():
    cands = [Candidate("x", "Hello Goodbye", "The Beatles", 1967, 70, None)]
    assert score_candidates(cands, "Adele", "Hello")[0].confidence < 0.5


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes  # list of (predicate, status, payload)
        self.calls = []

    def __call__(self, url, method):
        self.calls.append((method, url))
        for pred, status, payload in self.routes:
            if pred(method, url):
                body = json.dumps(payload).encode() if payload is not None else b""
                return status, body
        return 404, b""


def no_wait_mb(http):
    fc = FakeClock()
    return MusicBrainz(http=http, limiter=RateLimiter(0, clock=fc.clock, sleep=fc.sleep)), fc


def test_search_and_metadata_end_to_end():
    releases = [
        {
            "id": "r1",
            "status": "Official",
            "release-group": {
                "id": "rg-comp",
                "primary-type": "Album",
                "secondary-types": ["Compilation"],
            },
        },
        {
            "id": "r2",
            "status": "Official",
            "release-group": {"id": "rg-album", "primary-type": "Album"},
        },
    ]
    search = {
        "recordings": [rec("rec-1", "Respect", "Aretha Franklin", "1967-04-29", releases=releases)]
    }
    routes = [
        (lambda m, u: "/ws/2/recording?" in u, 200, search),
        (
            lambda m, u: "/ws/2/recording/rec-1" in u,
            200,
            {"genres": [], "tags": [], "artist-credit": []},
        ),
        (
            lambda m, u: "/ws/2/release-group/rg-album" in u,
            200,
            {
                "genres": [{"name": "soul", "count": 3}, {"name": "rhythm and blues", "count": 1}],
                "tags": [],
            },
        ),
        (lambda m, u: m == "HEAD" and "rg-album/front-250" in u, 200, None),
    ]
    http = FakeHTTP(routes)
    mb, _ = no_wait_mb(http)
    cands = mb.search("Aretha Franklin", "Respect")
    assert cands[0].recording_id == "rec-1" and cands[0].release_group_id == "rg-album"
    meta = mb.metadata(cands[0])
    assert meta.year == 1967
    assert meta.genres == ["soul", "rhythm and blues"]
    assert meta.cover_url.endswith("/release-group/rg-album/front-250")
    query = parse_qs(urlparse(http.calls[0][1]).query)["query"][0]
    assert query.startswith(
        'recording:"Respect" AND artist:"Aretha Franklin" AND NOT comment:(live OR'
    )


def test_retries_when_musicbrainz_asks_to_slow_down():
    answers = iter([(503, b""), (200, json.dumps({"recordings": []}).encode())])
    mb, fc = no_wait_mb(lambda url, method: next(answers))
    assert mb.search("", "Anything") == []
    assert fc.slept  # backed off before retrying


def test_no_cover_when_archive_has_none():
    mb, _ = no_wait_mb(FakeHTTP([]))
    assert mb.cover_url("rg-x") is None
    assert mb.cover_url(None) is None


def test_confirm_pick_skip_and_manual():
    cands = [Candidate("a", "Hello", "Adele", 2015, 100, None, confidence=0.6)]
    mb, _ = no_wait_mb(FakeHTTP([]))
    quiet = lambda *_: None  # noqa: E731
    assert confirm("Adele", "Hello", cands, mb, ask=lambda _: "1", show=quiet).recording_id == "a"
    assert confirm("Adele", "Hello", cands, mb, ask=lambda _: "s", show=quiet) is None

    answers = iter(["m", "Adele", "Hello", "s"])
    searched = []
    mb.search = lambda artist, track: searched.append((artist, track)) or []
    assert confirm("?", "?", [], mb, ask=lambda _: next(answers), show=quiet) is None
    assert searched == [("Adele", "Hello")]


def test_song_year_is_earliest_matching_recording():
    cands = [
        Candidate("remaster", "Respect", "Aretha Franklin", 1971, 100, None),
        Candidate("orig", "Respect", "Aretha Franklin", 1967, 98, None),
        Candidate("live", "Respect", "Aretha Franklin", 1965, 100, None, "live"),
    ]
    best = score_candidates(cands, "Aretha Franklin", "Respect")[0]
    assert best.year == 1967  # not the live one from another artist's set, not the reissue


def test_several_artists_in_title_are_tried_one_by_one():
    def http(url, method):
        q = parse_qs(urlparse(url).query)["query"][0]
        if 'artist:"The Dave Brubeck Quartet"' in q:
            payload = {"recordings": [rec("tf", "Take Five", "The Dave Brubeck Quartet", "1959")]}
        elif 'artist:"Dave Brubeck"' in q:  # a later re-recording, same confidence
            payload = {"recordings": [rec("re", "Take Five", "Dave Brubeck", "1974")]}
        else:
            payload = {"recordings": [rec("x", "Take Five", "Someone Else", "2019")]}
        return 200, json.dumps(payload).encode()

    mb, _ = no_wait_mb(http)
    best = mb.search("Dave Brubeck, The Dave Brubeck Quartet", "Take Five")[0]
    assert best.recording_id == "tf" and best.year == 1959
