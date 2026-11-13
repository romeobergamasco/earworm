"""Find the YouTube video of each chart entry (T14).

A YouTube search "<artist> <title>" (no download), then the best result is
chosen: plausible duration, the title words present, no live / cover / karaoke
/ sped-up version, and a bonus for official channels. Results are saved to
indexer/sources/youtube.csv as they come, so the work resumes where it stopped.
"""

import csv
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from indexer.charts import ChartEntry
from indexer.fetch import MAX_DURATION_S, MIN_DURATION_S
from indexer.titles import clean_channel, normalize

YOUTUBE_CSV = Path(__file__).parent / "sources" / "youtube.csv"
MIN_SCORE = 0.6
_BAD_WORDS = (
    "live", "cover", "karaoke", "instrumental", "remix", "reaction", "tutorial", "lesson",
    "slowed", "sped up", "nightcore", "8d", "1 hour", "10 hours", "loop", "piano version",
    "acoustic", "parody", "tribute", "reprise", "concert", "en direct", "medley",
    "rerecorded", "re recorded", "re record", "new version", "nouvelle version",
)  # fmt: skip
_OFFICIAL = ("vevo", "topic", "official", "officiel", "records", "music")


@dataclass
class Resolved:
    chart: str
    year: int
    rank: int
    artist: str
    title: str
    youtube_id: str
    yt_title: str
    yt_channel: str
    duration_s: float
    score: float


def _words(text: str) -> set[str]:
    return set(normalize(text).split())


def score_result(entry: ChartEntry, yt_title: str, channel: str, duration: float | None) -> float:
    """0..1, how likely this search result is the chart song's studio recording."""
    if not duration or not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        return 0.0
    yt = normalize(yt_title)
    title_words = _words(entry.title)
    if not title_words:
        return 0.0
    found = len(title_words & _words(yt_title)) / len(title_words)
    if found < 0.75:
        return 0.0
    score = 0.5 + 0.3 * found

    def name(text: str) -> str:  # "Sam the Sham & the Pharaohs" == "... and the Pharaohs"
        return normalize(text.replace("&", " and "))

    main_artist = name(re.split(r" featuring | feat\.? | with ", entry.artist)[0])
    chan = name(clean_channel(channel))
    yt = name(yt_title)
    if main_artist and (main_artist in yt or main_artist in chan or (chan and chan in main_artist)):
        score += 0.15
    if any(w in normalize(channel) for w in _OFFICIAL) or (chan and chan in main_artist):
        score += 0.05
    for w in _BAD_WORDS:
        if f" {w} " in f" {yt} " and f" {w} " not in f" {normalize(entry.title)} ":
            score -= 0.5
    return round(max(0.0, min(1.0, score)), 3)


def search_youtube(query: str, n: int = 5) -> list[dict]:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "js_runtimes": {"node": {}}}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(f"ytsearch{n}:{query}", download=False).get("entries") or []


def resolve(entry: ChartEntry, search=search_youtube) -> Resolved | None:
    best, best_score = None, 0.0
    for r in search(f"{entry.artist} {entry.title}"):
        s = score_result(
            entry,
            r.get("title") or "",
            r.get("channel") or r.get("uploader") or "",
            r.get("duration"),
        )
        if s > best_score:
            best, best_score = r, s
    if best is None or best_score < MIN_SCORE:
        return None
    return Resolved(
        entry.chart, entry.year, entry.rank, entry.artist, entry.title, best["id"],
        best.get("title") or "", best.get("channel") or best.get("uploader") or "",
        float(best.get("duration") or 0), best_score,
    )  # fmt: skip


def _key(chart: str, year, rank) -> tuple[str, int, int]:
    return chart, int(year), int(rank)


def load_resolved(path: Path = YOUTUBE_CSV) -> dict[tuple[str, int, int], dict]:
    """Already-processed entries; youtube_id is empty when nothing suitable was found."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {_key(r["chart"], r["year"], r["rank"]): r for r in csv.DictReader(f)}


def resolve_all(entries: list[ChartEntry], path: Path = YOUTUBE_CSV, search=search_youtube,
                pause: float = 1.0, log=print) -> None:  # fmt: skip
    done = load_resolved(path)
    fields = list(Resolved.__dataclass_fields__)
    new_file = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        todo = [e for e in entries if _key(e.chart, e.year, e.rank) not in done]
        for i, e in enumerate(todo, 1):
            try:
                res = resolve(e, search)
            except Exception as exc:  # network hiccup: leave it for the next run
                log(f"[{i}/{len(todo)}] {e.artist} - {e.title}: error {exc}")
                continue
            empty = {
                "youtube_id": "",
                "yt_title": "",
                "yt_channel": "",
                "duration_s": 0,
                "score": 0,
            }
            row = asdict(res) if res else {**asdict(e), **empty}
            row = {k: row.get(k, "") for k in fields}
            w.writerow(row)
            f.flush()
            found = f"{res.yt_title} [{res.yt_channel}] {res.score}" if res else "NOT FOUND"
            log(f"[{i}/{len(todo)}] {e.year} {e.chart} #{e.rank} {e.artist} - {e.title} -> {found}")
            time.sleep(pause)
