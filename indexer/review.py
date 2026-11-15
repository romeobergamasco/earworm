"""Review the visitors' proposals, locally (docs/spec.md, section 2.3; T25).

    python -m indexer review

For each pending proposal: the YouTube title and the best MusicBrainz match
(no download yet), then (a)ccept, (r)eject with a reason, (s)kip or (q)uit.
Accepting runs the same pipeline as `indexer add` (download, fingerprint,
embedding, genre, MusicBrainz, audio deleted) with source "proposal".

Status of a proposal: pending -> accepted -> indexed | failed (with the error),
or pending -> rejected (with the reason). A proposal left "accepted" by an
interrupted review is shown again. The server reloads its index within 10 min
of a new song (catalog_meta version), so an accepted song becomes recognisable
online without any other step.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from indexer.titles import guess_artist_track

OPEN = ("pending", "accepted")
MAX_REASON = 500


@dataclass
class Preview:
    youtube_title: str
    channel: str
    duration_s: float | None
    match: str | None  # "Artist - Title (year), confidence 0.93"
    error: str | None = None
    query: tuple[str, str] | None = None  # (artist, title) searched in MusicBrainz


def watch_url(youtube_id: str) -> str:
    return f"https://www.youtube.com/watch?v={youtube_id}"


def hint_of(p: dict) -> tuple[str, str] | None:
    """(artist, title) typed by the visitor, when both are given."""
    if p.get("artist_hint") and p.get("title_hint"):
        return p["artist_hint"], p["title_hint"]
    return None


def open_proposals(db) -> list[dict]:
    return db.select(
        "proposals",
        "id,youtube_id,title_hint,artist_hint,status,created_at",
        status=f"in.({','.join(OPEN)})",
        order="created_at",
    )


def preview(p: dict, downloader, mb) -> Preview:
    """What the video is, without downloading it.

    MusicBrainz is searched with the YouTube title and, when given, with the
    visitor's artist and title; the more confident of the two is kept (a typo
    in either must not hide the song). `query` is then passed to the pipeline,
    which searches the same thing.
    """
    try:
        info = downloader.info(p["youtube_id"])
    except Exception as e:  # removed, private, blocked...
        return Preview("?", "?", None, None, error=f"{type(e).__name__}: {e}"[:300])
    queries = [guess_artist_track(info.track or info.title, info.artist or info.channel)]
    if (hint := hint_of(p)) and hint not in queries:
        queries.append(hint)
    best, query, error = None, queries[0], None
    for q in queries:
        try:
            cands = mb.search(*q)
        except Exception as e:
            error = f"MusicBrainz: {e}"[:300]
            continue
        if cands and (best is None or cands[0].confidence > best.confidence):
            best, query = cands[0], q
    match = (
        f"{best.artist} - {best.title} ({best.year or '?'}), confidence {best.confidence:.2f}"
        if best
        else None
    )
    return Preview(info.title, info.channel, info.duration_s, match, error, query)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def review(
    db,
    add: Callable,
    show_preview: Callable[[dict], Preview],
    ask: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> dict[str, int]:
    """`add(url, hint=...)` -> indexer.pipeline.AddResult (raises on failure);
    `hint` is the MusicBrainz query chosen by the preview."""
    todo = open_proposals(db)
    out(f"{len(todo)} proposal(s) to review")
    counts = {"indexed": 0, "rejected": 0, "failed": 0, "skipped": 0}

    def update(pid: int, **values) -> None:
        db.update("proposals", {**values, "reviewed_at": _now()}, id=f"eq.{pid}")

    for n, p in enumerate(todo, 1):
        out("")
        out(f"[{n}/{len(todo)}] proposal #{p['id']}  {watch_url(p['youtube_id'])}")
        if p["status"] == "accepted":
            out("  (accepted during an interrupted review: not indexed yet)")
        given = " - ".join(x for x in (p.get("artist_hint"), p.get("title_hint")) if x)
        if given:
            out(f"  visitor says: {given}")
        pv = show_preview(p)
        if pv.error and pv.youtube_title == "?":
            out(f"  YouTube: unavailable ({pv.error})")
        else:
            minutes = f"{pv.duration_s / 60:.1f} min" if pv.duration_s else "? min"
            out(f"  YouTube: {pv.youtube_title}  [{pv.channel}]  {minutes}")
            out(f"  MusicBrainz: {pv.match or pv.error or 'no match'}")

        while True:
            answer = ask("(a)ccept, (r)eject, (s)kip, (q)uit: ").strip().lower()
            if answer in ("a", "r", "s", "q"):
                break
        if answer == "q":
            counts["skipped"] += len(todo) - n + 1
            break
        if answer == "s":
            counts["skipped"] += 1
            continue
        if answer == "r":
            reason = ""
            while not reason:
                reason = " ".join(ask("  reason: ").split())[:MAX_REASON]
            update(p["id"], status="rejected", reason=reason)
            counts["rejected"] += 1
            out("  rejected")
            continue

        update(p["id"], status="accepted")
        try:
            res = add(watch_url(p["youtube_id"]), hint=pv.query)
        except Exception as e:
            update(p["id"], status="failed", reason=f"{type(e).__name__}: {e}"[:MAX_REASON])
            counts["failed"] += 1
            out(f"  failed: {type(e).__name__}: {e}"[:300])
            continue
        s = res.song
        if res.status == "added":
            update(p["id"], status="indexed", song_id=s["id"], reason=None)
            counts["indexed"] += 1
            out(f"  indexed: #{s['id']} {s['artist']} - {s['title']} ({s['year'] or '?'})")
            out(f"  {res.detail}")
        else:  # duplicate_link / duplicate_audio: the song is already there
            why = f"already in the catalogue: #{s['id']} {s['artist']} - {s['title']}"
            update(p["id"], status="rejected", reason=why, song_id=s["id"])
            counts["rejected"] += 1
            out(f"  {why}")
    out("")
    out(", ".join(f"{k}: {v}" for k, v in counts.items()))
    return counts
