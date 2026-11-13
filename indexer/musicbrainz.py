"""MusicBrainz lookup: (artist, track) guess -> recording, year, genres, cover.

Rules of the MusicBrainz API (https://musicbrainz.org/doc/MusicBrainz_API):
at most 1 request per second, and a User-Agent naming the application and a
contact. The contact defaults to a neutral text; set EARWORM_CONTACT to the
project's GitHub URL once it exists. No personal data is sent.

Cover Art Archive is only probed (HEAD) to know whether a front cover exists;
the URL is stored, never the image.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from indexer.titles import normalize

MB_ROOT = "https://musicbrainz.org/ws/2"
CAA_ROOT = "https://coverartarchive.org"
VERSION = "0.1"
AUTO_ACCEPT = 0.80  # confidence above which no human check is needed

_VARIANT_WORDS = (  # another performance or arrangement, not the original
    "live",
    "remix",
    "karaoke",
    "instrumental",
    "cover",
    "acoustic",
    "demo",
    "rehearsal",
    "session",
)
# Never the original. ("version", "edit", "mix" are left out: MusicBrainz marks
# some originals as "album version" or "original mix".)
_EXCLUDED_IN_QUERY = (
    "live",
    "remix",
    "karaoke",
    "instrumental",
    "cover",
    "acoustic",
    "demo",
    "rehearsal",
)
_BAD_SECONDARY = {"Live", "Compilation", "Remix", "DJ-mix", "Karaoke", "Demo"}


def user_agent() -> str:
    contact = os.environ.get("EARWORM_CONTACT", "local development build, no contact yet")
    return f"Earworm/{VERSION} ( {contact} )"


class RateLimiter:
    """Guarantees at least `interval` seconds between two calls of wait()."""

    def __init__(self, interval: float = 1.1, clock=time.monotonic, sleep=time.sleep) -> None:
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            delay = self._last + self.interval - self.clock()
            if delay > 0:
                self.sleep(delay)
        self._last = self.clock()


def _http_get(url: str, method: str = "GET") -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": user_agent()})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


@dataclass
class Candidate:
    recording_id: str
    title: str
    artist: str
    year: int | None
    mb_score: int
    release_group_id: str | None
    disambiguation: str = ""
    confidence: float = 0.0


@dataclass
class Metadata:
    recording_id: str
    title: str
    artist: str
    year: int | None
    genres: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    cover_url: str | None = None
    release_group_id: str | None = None
    confidence: float = 1.0


def _lucene(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _credit(artist_credit: list[dict]) -> str:
    return "".join(c.get("name", "") + c.get("joinphrase", "") for c in artist_credit).strip()


def _year(date: str | None) -> int | None:
    return int(date[:4]) if date and date[:4].isdigit() else None


def _best_release_group(recording: dict) -> str | None:
    """Prefer an official album or single, not a compilation / live / remix."""
    best, best_rank = None, 99
    for rel in recording.get("releases", []):
        rg = rel.get("release-group") or {}
        if not rg.get("id"):
            continue
        rank = 0
        rank += rel.get("status") != "Official"
        rank += 2 * bool(_BAD_SECONDARY & set(rg.get("secondary-types") or []))
        rank += rg.get("primary-type") not in ("Album", "Single", "EP")
        if rank < best_rank:
            best, best_rank = rg["id"], rank
    return best


def parse_candidates(payload: dict) -> list[Candidate]:
    out = []
    for rec in payload.get("recordings", []):
        out.append(
            Candidate(
                recording_id=rec["id"],
                title=rec.get("title", ""),
                artist=_credit(rec.get("artist-credit", [])),
                year=_year(rec.get("first-release-date")),
                mb_score=int(rec.get("score", 0)),
                release_group_id=_best_release_group(rec),
                disambiguation=rec.get("disambiguation", ""),
            )
        )
    return out


def _similar(a: str, b: str) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _artist_similarity(guess: str, credit: str) -> float:
    """The guess may be only the main artist ('Luis Fonsi' vs 'Luis Fonsi feat. Daddy Yankee')."""
    g, c = normalize(guess), normalize(credit)
    if g and c and (c.startswith(g + " ") or c == g):
        return 1.0
    return _similar(guess, credit)


def _is_variant(c: Candidate, track: str) -> bool:
    text = normalize(f"{c.title} {c.disambiguation}")
    wanted = normalize(track)
    return any(f" {w} " in f" {text} " and f" {w} " not in f" {wanted} " for w in _VARIANT_WORDS)


def _rank(c: Candidate) -> tuple[float, int]:
    """More confident first; at equal confidence, the earliest (the original)."""
    return (-round(c.confidence, 2), c.year or 9999)


def score_candidates(cands: list[Candidate], artist: str, track: str) -> list[Candidate]:
    """Fill `confidence` and sort best first. Among near-equal candidates the
    earliest release wins (the original, not a reissue or compilation)."""
    for c in cands:
        sim_artist = _artist_similarity(artist, c.artist) if artist else 0.7
        conf = min(_similar(track, c.title), sim_artist) * (c.mb_score / 100)
        if _is_variant(c, track):
            conf *= 0.6
        c.confidence = round(conf, 3)
    if not cands:
        return []
    top = max(c.confidence for c in cands)
    ranked = sorted(
        cands,
        key=lambda c: (c.confidence < top - 0.05, c.year or 9999, -c.confidence),
    )
    # The same song exists as many recordings (reissues, compilations, remasters);
    # its year is the earliest among those that match as well as the best one.
    near_top = [c for c in ranked if c.confidence >= top - 0.05 and c.confidence > 0]
    years = [c.year for c in near_top if c.year]
    if years:
        for c in near_top:
            c.year = min(years)
    return ranked


class MusicBrainz:
    def __init__(
        self,
        http: Callable[[str, str], tuple[int, bytes]] = _http_get,
        limiter: RateLimiter | None = None,
        retries: int = 6,
    ) -> None:
        self.http = http
        self.limiter = limiter or RateLimiter()
        self.retries = retries

    def _get_json(self, path: str, params: dict) -> dict:
        url = f"{MB_ROOT}/{path}?" + urllib.parse.urlencode({**params, "fmt": "json"})
        for attempt in range(self.retries + 1):
            self.limiter.wait()
            status, body = self.http(url, "GET")
            if status == 200:
                return json.loads(body)
            if status in (429, 503) and attempt < self.retries:
                self.limiter.sleep(
                    3.0 * 2**attempt
                )  # MusicBrainz asks to slow down: 3 s, 6 s... ~3 min in all
                continue
            raise RuntimeError(f"MusicBrainz answered {status} for {path}")
        raise AssertionError("unreachable")

    def search(self, artist: str, track: str, limit: int = 50) -> list[Candidate]:
        """Candidates best first. Live / remix / demo versions are excluded in the
        query itself: some artists have hundreds of live recordings that would
        otherwise fill all the results."""
        base = f"recording:{_lucene(track)}"
        no_variants = " AND NOT comment:(" + " OR ".join(_EXCLUDED_IN_QUERY) + ")"
        queries = []
        if artist:
            queries += [
                f"{base} AND artist:{_lucene(artist)}{no_variants}",
                f"{base} AND artist:{_lucene(artist)}",
            ]
        queries += [base + no_variants]  # artist spelled differently
        best: list[Candidate] = []
        for query in queries:
            cands = parse_candidates(self._get_json("recording", {"query": query, "limit": limit}))
            if cands:
                best = score_candidates(cands, artist, track)
                break
        # "Dave Brubeck, The Dave Brubeck Quartet - Take Five": several artists in
        # the title; if unsure, try each name alone and keep the most confident.
        if "," in artist and (not best or best[0].confidence < AUTO_ACCEPT):
            for part in (p.strip() for p in artist.split(",")):
                if part:
                    alt = self.search(part, track, limit)
                    if alt and (not best or _rank(alt[0]) < _rank(best[0])):
                        best = alt
        return best

    def genres(self, cand: Candidate) -> tuple[list[str], list[str]]:
        """Genres and tags of the recording, else of its release group, else of the artist."""

        def names(data: dict, key: str) -> list[str]:
            items = sorted(data.get(key, []), key=lambda g: -g.get("count", 0))
            return [g["name"] for g in items if g.get("count", 0) > 0 or key == "genres"]

        rec = self._get_json(f"recording/{cand.recording_id}", {"inc": "genres tags artists"})
        genres, tags = names(rec, "genres"), names(rec, "tags")
        if not genres and cand.release_group_id:
            rg = self._get_json(f"release-group/{cand.release_group_id}", {"inc": "genres tags"})
            genres, tags = names(rg, "genres"), tags or names(rg, "tags")
        if not genres:
            credit = rec.get("artist-credit") or []
            if credit and credit[0].get("artist", {}).get("id"):
                art = self._get_json(f"artist/{credit[0]['artist']['id']}", {"inc": "genres tags"})
                genres, tags = names(art, "genres"), tags or names(art, "tags")
        return genres, tags

    def cover_url(self, release_group_id: str | None) -> str | None:
        if not release_group_id:
            return None
        url = f"{CAA_ROOT}/release-group/{release_group_id}/front-250"
        status, _ = self.http(url, "HEAD")
        return url if status in (200, 307) else None

    def metadata(self, cand: Candidate) -> Metadata:
        genres, tags = self.genres(cand)
        return Metadata(
            recording_id=cand.recording_id,
            title=cand.title,
            artist=cand.artist,
            year=cand.year,
            genres=genres,
            tags=tags,
            cover_url=self.cover_url(cand.release_group_id),
            release_group_id=cand.release_group_id,
            confidence=cand.confidence,
        )


def confirm(
    artist: str,
    track: str,
    cands: list[Candidate],
    mb: MusicBrainz,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
) -> Candidate | None:
    """Let a human pick the right recording. None = keep YouTube metadata only."""
    while True:
        show(f"\nGuess: {artist or '?'} - {track}")
        for i, c in enumerate(cands[:5], 1):
            extra = f" ({c.disambiguation})" if c.disambiguation else ""
            show(f"  {i}. {c.artist} - {c.title}{extra} [{c.year or '?'}]  conf {c.confidence:.2f}")
        answer = ask("Number, (m)anual search, (s)kip: ").strip().lower()
        if answer.isdigit() and 1 <= int(answer) <= min(5, len(cands)):
            return cands[int(answer) - 1]
        if answer == "s":
            return None
        if answer == "m":
            artist = ask("  artist: ").strip()
            track = ask("  title: ").strip()
            cands = mb.search(artist, track)
