"""Guess (artist, track) from a YouTube video title and its channel name.

YouTube rarely gives clean metadata for official videos: we only have titles
like "Queen – Bohemian Rhapsody (Official Video Remastered)" and channels like
"Queen Official". The guess is then looked up in MusicBrainz (musicbrainz.py).
"""

import re
import unicodedata

_SEPARATORS = re.compile(r"\s+[-–—|]\s+|\s*:\s+")
_FEAT = re.compile(r"\s+[\(\[]?\s*(?:ft|feat|featuring)\b\.?.*$", re.IGNORECASE)
_NOISE_WORDS = (
    r"official|officiel|officielle|video|vid[eé]o|audio|lyrics?|paroles|clip|mv|m/v|hd|hq|4k"
    r"|remaster(?:ed)?|visuali[sz]er|explicit|music video|full album|topic"
)
_BRACKETED = re.compile(r"\s*[\(\[]([^\)\]]*)[\)\]]")
_TRAILING_NOISE = re.compile(
    r"(?:\s+(?:official|officiel(?:le)?|lyrics?|music|lyric|audio|video|vid[eé]o|clip|mv|hd|4k))+\s*$",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"'‘’“”]([^\"'‘’“”]{2,})[\"'‘’“”]")
_CHANNEL_SUFFIX = re.compile(
    r"(?:\s*-\s*topic|\s*vevo|\s+official|\s+officiel|\s+music|music|\s+tv)$", re.IGNORECASE
)


def normalize(text: str) -> str:
    """Lowercase, no accents, no punctuation: for comparing names."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[.'’`]", "", text)  # D.R.E. == DRE, don't == dont
    text = re.sub(r"[^\w]+", " ", text)
    return " ".join(text.split())


def clean_channel(channel: str) -> str:
    """'Queen Official' -> 'Queen', 'DrDreVEVO' -> 'DrDre', 'X - Topic' -> 'X'."""
    name = channel.strip()
    while True:
        stripped = _CHANNEL_SUFFIX.sub("", name).strip()
        if stripped == name or not stripped:
            return name
        name = stripped


def _drop_noise_brackets(title: str) -> str:
    def repl(m: re.Match) -> str:
        inside = m.group(1)
        is_noise = re.search(rf"\b(?:{_NOISE_WORDS})\b", inside, re.IGNORECASE)
        non_latin = any(ord(c) > 0x2FF and c.isalpha() for c in inside)
        square = m.group(0).strip().startswith("[")
        return " " if (is_noise or non_latin or square) else m.group(0)

    return " ".join(_BRACKETED.sub(repl, title).split())


def _strip_feat(text: str) -> str:
    return _FEAT.sub("", text).strip()


def _strip_trailing_noise(text: str) -> str:
    return _TRAILING_NOISE.sub("", text).strip()


def guess_artist_track(title: str, channel: str = "") -> tuple[str, str]:
    """Best guess of (artist, track); either may be empty if unknown."""
    text = _drop_noise_brackets(title)

    # 'Artist 'Track' Official MV' (K-pop style quotes)
    quoted = _QUOTED.search(text)
    if quoted:
        artist = _strip_trailing_noise(text[: quoted.start()]).strip(" -–—:")
        if artist:
            return _strip_feat(artist), _strip_feat(quoted.group(1).strip())

    parts = _SEPARATORS.split(text, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        artist = _strip_feat(parts[0].strip())
        track = _strip_feat(_strip_trailing_noise(parts[1]))
        return artist, track

    # no separator: the channel is probably the artist
    text = _strip_feat(_strip_trailing_noise(text))
    artist = clean_channel(channel)
    if artist and normalize(text).startswith(normalize(artist) + " "):
        # drop the artist's words from the front of the title
        n_words = len(artist.split())
        return artist, " ".join(text.split()[n_words:])
    return artist, text
