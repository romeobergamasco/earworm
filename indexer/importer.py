"""Batch import of the chart songs, resumable (T14).

    python -m indexer charts-build      Wikipedia charts -> indexer/sources/charts.csv
    python -m indexer charts-resolve    YouTube search   -> indexer/sources/youtube.csv
    python -m indexer import-charts --target 200

Every outcome is appended to data/import_log.csv; a later run skips what is
done (added, duplicate, refused) and only retries errors, so the import can be
stopped and resumed at any time.
"""

import csv
import time
from datetime import UTC, datetime
from pathlib import Path

from indexer.charts import ChartEntry
from indexer.fetch import DurationError
from indexer.pipeline import add_song

IMPORT_LOG = Path("data") / "import_log.csv"
_FIELDS = [
    "time",
    "youtube_id",
    "chart",
    "year",
    "rank",
    "artist",
    "title",
    "status",
    "song_id",
    "detail",
]


def done_ids(path: Path = IMPORT_LOG) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {r["youtube_id"] for r in csv.DictReader(f) if r["status"] != "error"}


def run_import(
    catalog,
    entries: list[ChartEntry],
    resolved: dict,
    target: int,
    log_path: Path = IMPORT_LOG,
    add=add_song,
    mb=None,
    out=print,
    exclude: set[str] | None = None,
    pause_s: float = 0,
    block_wait_s: float = 1800,
    max_blocks: int = 8,
    sleep=time.sleep,
) -> dict[str, int]:
    """Import in chart order until the catalogue holds `target` songs.

    Patient mode (as embed-missing): downloads are spaced by `pause_s`; when
    YouTube answers "not a bot", wait `block_wait_s` and go on, up to
    `max_blocks` blocks in a row (then stop: run again later).

    `exclude`: videos never to import; by default the negatives of the
    measurement test set (docs/eval/testset.csv), which must stay absent."""
    if exclude is None:
        from indexer.testset import negative_ids

        exclude = negative_ids()
    skip = done_ids(log_path) | exclude
    attempts, blocked = 0, 0
    counts: dict[str, int] = {}
    new_file = not log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new_file:
            w.writeheader()
        for e in entries:
            if len(catalog.songs) >= target:
                break
            r = resolved.get((e.chart, e.year, e.rank))
            vid = r["youtube_id"] if r else ""
            if not vid or vid in skip or catalog.by_youtube_id(vid):
                continue
            if attempts:
                sleep(pause_s)
            attempts += 1
            t0 = time.perf_counter()
            try:
                res = add(
                    f"https://www.youtube.com/watch?v={vid}",
                    catalog,
                    mb=mb,
                    interactive=False,
                    source="chart",
                    hint=(e.artist, e.title),
                )
                status, song_id, detail = res.status, res.song["id"], res.detail
            except DurationError as exc:
                status, song_id, detail = "refused", "", str(exc)
            except Exception as exc:  # network, decoding...: retried on the next run
                status, song_id, detail = "error", "", f"{type(exc).__name__}: {exc}"[:300]
            blocked = blocked + 1 if "not a bot" in detail else 0
            counts[status] = counts.get(status, 0) + 1
            skip.add(vid)
            w.writerow(
                {
                    "time": datetime.now(UTC).isoformat(timespec="seconds"),
                    "youtube_id": vid,
                    "chart": e.chart,
                    "year": e.year,
                    "rank": e.rank,
                    "artist": e.artist,
                    "title": e.title,
                    "status": status,
                    "song_id": song_id,
                    "detail": detail,
                }  # fmt: skip
            )
            f.flush()
            out(
                f"[{len(catalog.songs)}/{target}] {e.year} {e.chart} #{e.rank} "
                f"{e.artist} - {e.title}: {status} ({time.perf_counter() - t0:.0f} s)"
            )
            if blocked >= max_blocks:
                out("YouTube keeps blocking: stopping (run again later)")
                break
            if blocked:
                out(f"blocked by YouTube: waiting {block_wait_s / 60:.0f} min")
                sleep(block_wait_s)
    return counts
