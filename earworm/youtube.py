"""YouTube link validation. Pure string handling: no network.

Shared by the local indexer and the server (proposals, T24).
"""

import re
from urllib.parse import parse_qs, urlparse

_ID = re.compile(r"[A-Za-z0-9_-]{11}")
_WATCH_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
_SHORT_HOSTS = {"youtu.be", "www.youtu.be"}


def parse_youtube_id(url: str) -> str:
    """Return the 11-character video id of a YouTube link, or raise ValueError.

    Accepted: youtube.com/watch?v=, m.youtube.com, music.youtube.com,
    youtube.com/shorts/<id>, youtu.be/<id> (extra parameters are ignored).
    """
    text = url.strip()
    if "://" not in text:
        text = "https://" + text
    parts = urlparse(text)
    if parts.scheme not in ("http", "https"):
        raise ValueError("not a web link")
    host = (parts.hostname or "").lower()
    segments = [s for s in parts.path.split("/") if s]

    candidate = None
    if host in _SHORT_HOSTS and len(segments) == 1:
        candidate = segments[0]
    elif host in _WATCH_HOSTS:
        if segments == ["watch"]:
            values = parse_qs(parts.query).get("v", [])
            candidate = values[0] if len(values) == 1 else None
        elif len(segments) == 2 and segments[0] == "shorts":
            candidate = segments[1]
    if candidate is None or not _ID.fullmatch(candidate):
        raise ValueError("not a YouTube video link")
    return candidate
