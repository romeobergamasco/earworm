"""T11: `indexer add` pipeline on a local catalogue (fake YouTube, fake MusicBrainz)."""

import glob
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import fingerprint
from earworm.index import Index
from indexer.catalog import LocalCatalog
from indexer.fetch import VideoInfo
from indexer.musicbrainz import Candidate, Metadata
from indexer.pipeline import add_song
from tests.test_cli import song

# youtube id -> (seed of the synthetic audio, title, channel)
VIDEOS = {
    "AAAAAAAAAAA": (1, "Alpha Band - First Song (Official Video)", "Alpha Band"),
    "BBBBBBBBBBB": (2, "Beta - Second Song", "Beta"),
    "CCCCCCCCCCC": (1, "First Song (lyrics) - Alpha Band", "Some fan channel"),  # same audio as A
}


def url(vid):
    return f"https://www.youtube.com/watch?v={vid}"


class FakeDownloader:
    def __init__(self):
        self.downloads = []

    def info(self, youtube_id):
        seed, title, channel = VIDEOS[youtube_id]
        return VideoInfo(youtube_id, title, channel, 90.0)

    def download(self, youtube_id, out_dir: Path):
        self.downloads.append(youtube_id)
        seed = VIDEOS[youtube_id][0]
        path = out_dir / f"{youtube_id}.wav"
        sf.write(path, song(seed, seconds=60), SAMPLE_RATE)
        return path


class FakeMB:
    def __init__(self, confidence=1.0, fail=False):
        self.confidence, self.fail = confidence, fail

    def search(self, artist, track):
        if self.fail:
            raise RuntimeError("MusicBrainz answered 503")
        return [
            Candidate(f"mb-{track}", track, artist, 2001, 100, "rg", confidence=self.confidence)
        ]

    def metadata(self, c):
        return Metadata(c.recording_id, c.title, c.artist, c.year, ["rock"], [], None, "rg")


@pytest.fixture
def catalog(tmp_path):
    return LocalCatalog(tmp_path / "catalog")


def temp_leftovers():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "earworm-*")))


def test_add_then_recognise(catalog):
    before = temp_leftovers()
    dl = FakeDownloader()
    res = add_song(url("AAAAAAAAAAA"), catalog, downloader=dl, mb=FakeMB(), interactive=False)
    assert res.status == "added"
    s = res.song
    assert (s["artist"], s["title"], s["year"]) == ("Alpha Band", "First Song", 2001)
    assert s["mb_recording_id"] == "mb-First Song" and s["mb_genres"] == ["rock"]
    assert s["youtube_id"] == "AAAAAAAAAAA" and s["duration_s"] == pytest.approx(60, abs=0.1)
    assert temp_leftovers() == before  # no audio left behind

    # reload from disk, recognise a 10 s excerpt
    reloaded = LocalCatalog(catalog.root)
    assert len(reloaded.songs) == 1
    idx = Index.load(reloaded.index_path)
    best = idx.query(fingerprint(song(1, seconds=60)[20 * SAMPLE_RATE : 30 * SAMPLE_RATE]))[0]
    assert best.song_id == s["id"] and best.is_confident
    assert idx.title(best.song_id) == "Alpha Band - First Song"


def test_same_link_twice_is_refused_without_download(catalog):
    dl = FakeDownloader()
    add_song(url("AAAAAAAAAAA"), catalog, downloader=dl, mb=FakeMB(), interactive=False)
    res = add_song(
        "https://youtu.be/AAAAAAAAAAA", catalog, downloader=dl, mb=FakeMB(), interactive=False
    )
    assert res.status == "duplicate_link"
    assert dl.downloads == ["AAAAAAAAAAA"]


def test_same_audio_under_another_link_is_a_duplicate(catalog):
    dl = FakeDownloader()
    first = add_song(url("AAAAAAAAAAA"), catalog, downloader=dl, mb=FakeMB(), interactive=False)
    add_song(url("BBBBBBBBBBB"), catalog, downloader=dl, mb=FakeMB(), interactive=False)
    res = add_song(url("CCCCCCCCCCC"), catalog, downloader=dl, mb=FakeMB(), interactive=False)
    assert res.status == "duplicate_audio"
    assert res.song["id"] == first.song["id"]
    assert len(LocalCatalog(catalog.root).songs) == 2


def test_unsure_musicbrainz_falls_back_to_youtube_metadata(catalog):
    res = add_song(
        url("BBBBBBBBBBB"),
        catalog,
        downloader=FakeDownloader(),
        mb=FakeMB(confidence=0.3),
        interactive=False,
    )
    assert res.status == "added"
    s = res.song
    assert (s["artist"], s["title"]) == ("Beta", "Second Song")
    assert s["mb_recording_id"] is None and s["year"] is None


def test_musicbrainz_down_does_not_lose_the_song(catalog):
    res = add_song(
        url("BBBBBBBBBBB"),
        catalog,
        downloader=FakeDownloader(),
        mb=FakeMB(fail=True),
        interactive=False,
    )
    assert res.status == "added" and res.song["mb_recording_id"] is None


def test_analysis_error_leaves_catalogue_and_disk_clean(catalog, monkeypatch):
    before = temp_leftovers()

    def boom(*a, **k):
        raise RuntimeError("decoder crashed")

    monkeypatch.setattr("indexer.pipeline.load", boom)
    with pytest.raises(RuntimeError):
        add_song(
            url("AAAAAAAAAAA"), catalog, downloader=FakeDownloader(), mb=FakeMB(), interactive=False
        )
    assert LocalCatalog(catalog.root).songs == []
    assert temp_leftovers() == before


def test_catalogue_ids_are_stable_and_increasing(catalog):
    dl = FakeDownloader()
    a = add_song(url("AAAAAAAAAAA"), catalog, downloader=dl, mb=FakeMB(), interactive=False)
    b = add_song(url("BBBBBBBBBBB"), catalog, downloader=dl, mb=FakeMB(), interactive=False)
    assert (a.song["id"], b.song["id"]) == (1, 2)
    assert np.array_equal(sorted(s["id"] for s in LocalCatalog(catalog.root).songs), [1, 2])


def test_hint_from_a_chart_beats_the_youtube_title(catalog):
    searched = []

    class RecordingMB(FakeMB):
        def search(self, artist, track):
            searched.append((artist, track))
            return []

    res = add_song(
        url("CCCCCCCCCCC"),  # YouTube title "First Song (lyrics) - Alpha Band" is misleading
        catalog,
        downloader=FakeDownloader(),
        mb=RecordingMB(),
        interactive=False,
        hint=("Alpha Band", "First Song"),
    )
    assert searched == [("Alpha Band", "First Song")]
    assert (res.song["artist"], res.song["title"]) == ("Alpha Band", "First Song")
