"""T10 acceptance against the real MusicBrainz API (network).

Skipped unless EARWORM_NETWORK=1. Takes ~1 min (1 request per second).
20 real YouTube titles + channels (tests/data/mb_titles.csv, collected with
a YouTube search) -> at least 15 must be matched automatically and correctly.
"""

import csv
import os
from pathlib import Path

import pytest

from indexer.musicbrainz import AUTO_ACCEPT, MusicBrainz
from indexer.titles import guess_artist_track, normalize

pytestmark = pytest.mark.skipif(
    os.environ.get("EARWORM_NETWORK") != "1", reason="network test: set EARWORM_NETWORK=1"
)
CASES = Path(__file__).parent / "data" / "mb_titles.csv"


def same(expected: str, got: str) -> bool:
    e, g = normalize(expected), normalize(got)
    return e == g or g.startswith(e + " ") or e.startswith(g + " ")


def test_twenty_real_titles():
    with open(CASES, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mb = MusicBrainz()
    auto_ok, report = 0, []
    for r in rows:
        artist, track = guess_artist_track(r["title"], r["channel"])
        cands = mb.search(artist, track)
        best = cands[0] if cands else None
        correct = bool(
            best
            and same(r["expected_track"], best.title)
            and same(r["expected_artist"], best.artist)
        )
        auto = bool(best and best.confidence >= AUTO_ACCEPT)
        auto_ok += auto and correct
        meta = mb.metadata(best) if auto and correct else None
        found = f"{best.artist} - {best.title}"[:40] if best else "-"
        report.append(
            f"{'OK  ' if auto and correct else ('MAN ' if not auto else 'FAUX')} "
            f"{r['title'][:45]:45s} -> {found:40s} "
            f"{best.year if best else ''} conf={best.confidence if best else 0:.2f}"
            + (
                f" genres={meta.genres[:3]} cover={'oui' if meta.cover_url else 'non'}"
                if meta
                else ""
            )
        )
    print("\n" + "\n".join(report) + f"\n\n{auto_ok}/{len(rows)} matched automatically")
    wrong_auto = [line for line in report if line.startswith("FAUX")]
    assert not wrong_auto, "an automatic match must never be wrong"
    assert auto_ok >= 15
