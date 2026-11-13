"""Add one YouTube song to the catalogue (spec section 5).

    link -> already in catalogue? -> download (temporary) -> fingerprint
         -> same audio already indexed? -> audio deleted -> MusicBrainz -> save

The audio only exists inside `downloaded_audio`; MusicBrainz (which may ask a
human) runs after it has been deleted.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from earworm.audio import SAMPLE_RATE, load
from earworm.fingerprint import fingerprint
from earworm.youtube import parse_youtube_id
from indexer.catalog import LocalCatalog
from indexer.fetch import Downloader, VideoInfo, downloaded_audio
from indexer.musicbrainz import AUTO_ACCEPT, Metadata, MusicBrainz, confirm
from indexer.titles import guess_artist_track

DUP_EXCERPT_S = 10.0
DUP_POSITIONS = (0.25, 0.5, 0.75)  # where to take the probe excerpts in the song
DUP_VOTES = 2  # how many probes must recognise the same catalogue song


@dataclass
class AddResult:
    status: str  # added | duplicate_link | duplicate_audio
    song: dict | None
    detail: str = ""


def find_audio_duplicate(catalog: LocalCatalog, y: np.ndarray) -> dict | None:
    """Is this audio already in the catalogue (under another link)?

    Three 10 s probes, as if a user were recognising the song; it is a
    duplicate when at least two of them confidently find the same song."""
    if not catalog.songs:
        return None
    n = int(DUP_EXCERPT_S * SAMPLE_RATE)
    votes = Counter()
    for pos in DUP_POSITIONS:
        start = max(0, min(len(y) - n, int(pos * len(y)) - n // 2))
        matches = catalog.index.query(fingerprint(y[start : start + n]))
        if matches and matches[0].is_confident:
            votes[matches[0].song_id] += 1
    if votes:
        song_id, count = votes.most_common(1)[0]
        if count >= DUP_VOTES:
            return catalog.by_id(song_id)
    return None


def _metadata(
    info: VideoInfo, mb, interactive: bool, hint: tuple[str, str] | None = None
) -> tuple[Metadata | None, tuple[str, str], str]:
    """`hint` = (artist, title) known from elsewhere (a chart): better than the YouTube title."""
    if hint:
        artist, track = hint
    else:
        artist, track = guess_artist_track(info.track or info.title, info.artist or info.channel)
    try:
        cands = mb.search(artist, track)
    except Exception as e:  # network trouble must not lose the analysed song
        return None, (artist, track), f"MusicBrainz unavailable ({e}); YouTube metadata kept"
    best = cands[0] if cands else None
    if best is not None and best.confidence >= AUTO_ACCEPT:
        return mb.metadata(best), (artist, track), f"MusicBrainz auto ({best.confidence:.2f})"
    if interactive:
        chosen = confirm(artist, track, cands, mb)
        if chosen is not None:
            return mb.metadata(chosen), (artist, track), "MusicBrainz confirmed by hand"
    return None, (artist, track), "no confident MusicBrainz match; YouTube metadata kept"


def add_song(
    url: str,
    catalog: LocalCatalog,
    downloader: Downloader | None = None,
    mb: MusicBrainz | None = None,
    interactive: bool = True,
    source: str = "manual",
    hint: tuple[str, str] | None = None,
) -> AddResult:
    youtube_id = parse_youtube_id(url)
    existing = catalog.by_youtube_id(youtube_id)
    if existing:
        return AddResult("duplicate_link", existing, "this link is already in the catalogue")

    with downloaded_audio(url, downloader=downloader) as audio:
        y, _ = load(audio.path)
        info = audio.info
        hashes = fingerprint(y)
        duplicate = find_audio_duplicate(catalog, y)
        duration_s = len(y) / SAMPLE_RATE
    # from here on, the audio file no longer exists

    if duplicate:
        return AddResult(
            "duplicate_audio",
            duplicate,
            f"same audio as #{duplicate['id']} {duplicate['artist']} - {duplicate['title']}",
        )

    meta, (artist, track), detail = _metadata(info, mb or MusicBrainz(), interactive, hint)
    record = {
        "youtube_id": youtube_id,
        "title": meta.title if meta else track,
        "artist": meta.artist if meta else artist,
        "year": meta.year if meta else None,
        "mb_recording_id": meta.recording_id if meta else None,
        "mb_genres": meta.genres if meta else [],
        "mb_tags": meta.tags if meta else [],
        "cover_url": meta.cover_url if meta else None,
        "duration_s": round(duration_s, 2),
        "source": source,
        "youtube_title": info.title,
        "youtube_channel": info.channel,
    }
    return AddResult("added", catalog.add(record, hashes), detail)
