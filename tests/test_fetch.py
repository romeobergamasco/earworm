"""T09: download -> analyse -> the audio is gone, whatever happens."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from earworm.audio import SAMPLE_RATE, load
from indexer.fetch import DurationError, VideoInfo, downloaded_audio

VID = "dQw4w9WgXcQ"
URL = f"https://www.youtube.com/watch?v={VID}"


class FakeDownloader:
    """Writes a short WAV instead of talking to YouTube."""

    def __init__(self, duration_s=200.0):
        self.duration_s = duration_s
        self.downloads = 0

    def info(self, youtube_id: str) -> VideoInfo:
        return VideoInfo(
            youtube_id, "Some Artist - Some Song (Official Video)", "Some Artist", self.duration_s
        )

    def download(self, youtube_id: str, out_dir: Path) -> Path:
        self.downloads += 1
        path = out_dir / f"{youtube_id}.wav"
        t = np.arange(3 * SAMPLE_RATE) / SAMPLE_RATE
        sf.write(path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SAMPLE_RATE)
        return path


def test_audio_exists_during_analysis_and_is_deleted_after():
    dl = FakeDownloader()
    with downloaded_audio(URL, downloader=dl) as audio:
        assert audio.info.youtube_id == VID
        assert audio.path.is_file()
        y, _ = load(audio.path)
        assert len(y) == 3 * SAMPLE_RATE
        folder = audio.path.parent
    assert not audio.path.exists()
    assert not folder.exists()


def test_audio_is_deleted_even_if_analysis_fails():
    dl = FakeDownloader()
    with pytest.raises(RuntimeError, match="analysis crashed"):
        with downloaded_audio(URL, downloader=dl) as audio:
            folder = audio.path.parent
            raise RuntimeError("analysis crashed")
    assert not folder.exists()


def test_audio_is_deleted_if_download_fails(monkeypatch, tmp_path):
    created = []

    class Broken(FakeDownloader):
        def download(self, youtube_id, out_dir):
            created.append(out_dir)
            (out_dir / "partial.webm.part").write_bytes(b"half a file")
            raise OSError("connection reset")

    with pytest.raises(OSError):
        with downloaded_audio(URL, downloader=Broken()):
            pass
    assert created and not created[0].exists()


@pytest.mark.parametrize("duration", [30.0, 59.0, 12 * 60 + 1, 3600.0])
def test_refuses_too_short_or_too_long_without_downloading(duration):
    dl = FakeDownloader(duration_s=duration)
    with pytest.raises(DurationError):
        with downloaded_audio(URL, downloader=dl):
            pass
    assert dl.downloads == 0


def test_rejects_invalid_link_without_any_call():
    dl = FakeDownloader()
    with pytest.raises(ValueError):
        with downloaded_audio("https://example.com/video", downloader=dl):
            pass
    assert dl.downloads == 0
