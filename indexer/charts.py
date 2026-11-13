"""Build the catalogue's source list from yearly charts (T14).

Two charts per year, from Wikipedia:
  * US: Billboard year-end singles (top 30/50 in the 1950s, Hot 100 since 1959),
    taken in rank order. It stands for "worldwide hits".
  * France: the year's number-one singles, ranked by weeks at number one.

How many songs per year depends on the decade (QUOTAS): fewer for the 1950s,
more for the decades people listen to most today. The final list is ordered
"all number ones first, then all number twos...", so that even a partly
imported catalogue covers every era.
"""

import csv
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

from indexer.tables import wikitables
from indexer.titles import normalize

SOURCES = Path(__file__).parent / "sources"
CHARTS_CSV = SOURCES / "charts.csv"

# decade -> (US songs per year, French number ones per year)
QUOTAS = {
    1950: (5, 3),
    1960: (8, 5),
    1970: (10, 5),
    1980: (10, 6),
    1990: (10, 6),
    2000: (9, 6),
    2010: (9, 6),
    2020: (8, 5),
}
FIRST_YEAR, LAST_YEAR = 1950, 2025


@dataclass
class ChartEntry:
    chart: str  # "us" | "fr"
    year: int
    rank: int  # 1 = top of that year's list
    artist: str
    title: str
    weeks: int = 0  # weeks at number one (France)


def _find_columns(header: list[str], wanted: dict[str, tuple[str, ...]]) -> dict[str, int] | None:
    cols = {}
    low = [h.lower() for h in header]
    for key, names in wanted.items():
        idx = next((i for i, h in enumerate(low) if any(n in h for n in names)), None)
        if idx is None:
            return None
        cols[key] = idx
    return cols


def _clean_title(text: str) -> str:
    text = re.split(r'"\s*/\s*"', text)[0]  # double A-side: keep the first song
    text = re.sub(r'^[\s"“”«»\']+|[\s"“”«»\']+$', "", text)
    return re.sub(r"\s+", " ", text)


def _header(table: list[list[str]], wanted: dict) -> tuple[dict[str, int], int] | None:
    """Columns found in the first or, for two-level headers ("Physique / Numérique"
    then "Artiste / Titre"), the second row; and the index of the first data row."""
    for i in (0, 1):
        if len(table) > i:
            cols = _find_columns(table[i], wanted)
            if cols is not None:
                return cols, i + 1
    return None


def _singles_section(html: str) -> str:
    """The page up to the first heading that mentions albums."""
    for m in re.finditer(r"<h[2-4][^>]*>(.*?)</h[2-4]>", html, re.S):
        if "album" in re.sub(r"<[^>]+>", "", m.group(1)).lower():
            return html[: m.start()]
    return html


def parse_billboard(html: str) -> list[tuple[str, str]]:
    """(artist, title) in rank order."""
    for table in wikitables(html):
        if not table:
            continue
        cols = _find_columns(table[0], {"title": ("title", "song"), "artist": ("artist",)})
        if cols is None:
            continue
        out = []
        for row in table[1:]:
            if len(row) <= max(cols.values()):
                continue
            title, artist = _clean_title(row[cols["title"]]), row[cols["artist"]].strip()
            if title and artist:
                out.append((artist, title))
        if len(out) >= 10:
            return out
    return []


def parse_france(html: str) -> list[tuple[str, str, int]]:
    """(artist, title, weeks at number one), most weeks first.

    Only the "singles" section is read (the pages also chart albums). When a
    table has two artist/title pairs (physical / digital, sales / streaming),
    the first pair is used."""
    wanted = {"title": ("titre", "single", "chanson"), "artist": ("artiste", "interprète")}
    for table in wikitables(_singles_section(html)):
        found = _header(table, wanted) if table else None
        if found is None or "album" in " ".join(table[0]).lower():
            continue
        cols, start = found
        weeks: OrderedDict[tuple[str, str], int] = OrderedDict()
        names: dict[tuple[str, str], tuple[str, str]] = {}
        for row in table[start:]:
            if len(row) <= max(cols.values()):
                continue
            title, artist = _clean_title(row[cols["title"]]), row[cols["artist"]].strip()
            if not title or not artist:
                continue
            key = (normalize(artist), normalize(title))
            weeks[key] = weeks.get(key, 0) + 1
            names.setdefault(key, (artist, title))
        if weeks:
            order = sorted(weeks, key=lambda k: -weeks[k])  # stable: ties keep chart order
            return [(*names[k], weeks[k]) for k in order]
    return []


def quotas(year: int) -> tuple[int, int]:
    return QUOTAS[year // 10 * 10]


def build(fetch) -> list[ChartEntry]:
    """All chart entries within quotas, deduplicated, in import order.

    `fetch(lang, title)` returns (resolved title, html) or None (indexer.wiki.fetch_html)."""
    from indexer.wiki import billboard_title, france_title

    entries: list[ChartEntry] = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        n_us, n_fr = quotas(year)
        page = fetch("en", billboard_title(year))
        if page:
            for rank, (artist, title) in enumerate(parse_billboard(page[1])[:n_us], 1):
                entries.append(ChartEntry("us", year, rank, artist, title))
        page = fetch("fr", france_title(year))
        if page:
            for rank, (artist, title, weeks) in enumerate(parse_france(page[1])[:n_fr], 1):
                entries.append(ChartEntry("fr", year, rank, artist, title, weeks))

    # same song in several lists (a US hit that was also number one in France, a
    # song charting two years in a row): keep its best placement only
    best: dict[tuple[str, str], ChartEntry] = {}
    for e in entries:
        key = (normalize(e.artist), normalize(e.title))
        if key not in best or (e.rank, e.year) < (best[key].rank, best[key].year):
            best[key] = e
    return sorted(best.values(), key=lambda e: (e.rank, e.year, e.chart))


def save(entries: list[ChartEntry], path: Path = CHARTS_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(entries[0])))
        w.writeheader()
        w.writerows(asdict(e) for e in entries)


def load(path: Path = CHARTS_CSV) -> list[ChartEntry]:
    with open(path, encoding="utf-8") as f:
        return [
            ChartEntry(
                r["chart"], int(r["year"]), int(r["rank"]), r["artist"], r["title"], int(r["weeks"])
            )
            for r in csv.DictReader(f)
        ]
