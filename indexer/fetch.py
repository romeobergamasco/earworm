"""Download a YouTube video's audio, only for as long as it is being analysed.

    with downloaded_audio(url) as audio:
        y, sr = load(audio.path)      # analyse here
        ...
    # here the audio file and its folder no longer exist, even after an error

Runs on the local machine only (YouTube blocks cloud IPs). No ffmpeg needed:
the audio stream is saved as YouTube serves it (m4a / webm) and decoded by
PyAV in earworm.audio.load.
"""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from earworm.youtube import parse_youtube_id

MIN_DURATION_S = 60
MAX_DURATION_S = 12 * 60


class DurationError(ValueError):
    """The video is too short or too long to be a song."""


@dataclass(frozen=True)
class VideoInfo:
    youtube_id: str
    title: str
    channel: str
    duration_s: float
    artist: str | None = None  # YouTube Music metadata, when present
    track: str | None = None


@dataclass(frozen=True)
class DownloadedAudio:
    info: VideoInfo
    path: Path


class Downloader(Protocol):
    def info(self, youtube_id: str) -> VideoInfo: ...

    def download(self, youtube_id: str, out_dir: Path) -> Path: ...


class YtDlpDownloader:
    """yt-dlp, audio stream only, one video (never a playlist)."""

    def __init__(self, **extra_options) -> None:
        self.options = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "noplaylist": True,
            "format": "bestaudio[ext=m4a]/bestaudio/best",
            # recent YouTube pages need a JS runtime; Node is the one installed here
            "js_runtimes": {"node": {}},
            **extra_options,
        }

    @staticmethod
    def _url(youtube_id: str) -> str:
        return f"https://www.youtube.com/watch?v={youtube_id}"

    def info(self, youtube_id: str) -> VideoInfo:
        import yt_dlp

        with yt_dlp.YoutubeDL(self.options) as ydl:
            data = ydl.extract_info(self._url(youtube_id), download=False)
        return VideoInfo(
            youtube_id=youtube_id,
            title=data.get("title") or "",
            channel=data.get("channel") or data.get("uploader") or "",
            duration_s=float(data.get("duration") or 0),
            artist=data.get("artist") or (data.get("artists") or [None])[0],
            track=data.get("track"),
        )

    def download(self, youtube_id: str, out_dir: Path) -> Path:
        import yt_dlp

        options = {**self.options, "outtmpl": str(out_dir / "%(id)s.%(ext)s")}
        with yt_dlp.YoutubeDL(options) as ydl:
            data = ydl.extract_info(self._url(youtube_id), download=True)
            path = Path(ydl.prepare_filename(data))
        if not path.is_file():
            raise FileNotFoundError(f"yt-dlp did not produce {path.name}")
        return path


def check_duration(info: VideoInfo) -> None:
    if not MIN_DURATION_S <= info.duration_s <= MAX_DURATION_S:
        raise DurationError(
            f"{info.duration_s:.0f} s: songs must last between {MIN_DURATION_S} s "
            f"and {MAX_DURATION_S // 60} min"
        )


@contextmanager
def downloaded_audio(url: str, downloader: Downloader | None = None) -> Iterator[DownloadedAudio]:
    """Validate the link and the duration, download the audio into a fresh
    temporary folder, yield it, then delete the folder (success or error)."""
    youtube_id = parse_youtube_id(url)
    downloader = downloader or YtDlpDownloader()
    info = downloader.info(youtube_id)
    check_duration(info)  # before downloading anything
    with tempfile.TemporaryDirectory(prefix="earworm-") as tmp:
        path = downloader.download(youtube_id, Path(tmp))
        yield DownloadedAudio(info, path)
