"""Local indexer command line.

python -m indexer add <YouTube link> [more links...]   download, analyse, delete, catalogue
python -m indexer lookup <YouTube link>   metadata only (no audio download)
"""

import argparse
import sys
import time
from pathlib import Path

from earworm.youtube import parse_youtube_id
from indexer.fetch import YtDlpDownloader
from indexer.musicbrainz import AUTO_ACCEPT, Metadata, MusicBrainz, confirm
from indexer.titles import guess_artist_track


def lookup(url: str, interactive: bool = True, mb: MusicBrainz | None = None) -> Metadata | None:
    """YouTube link -> MusicBrainz metadata, asking a human when unsure."""
    mb = mb or MusicBrainz()
    info = YtDlpDownloader().info(parse_youtube_id(url))
    artist, track = guess_artist_track(info.track or info.title, info.artist or info.channel)
    print(f"YouTube: {info.title}  [{info.channel}]")
    cands = mb.search(artist, track)
    best = cands[0] if cands else None
    if best and best.confidence >= AUTO_ACCEPT:
        print(f"auto: {best.artist} - {best.title} ({best.year}), confidence {best.confidence:.2f}")
    elif interactive:
        best = confirm(artist, track, cands, mb)
    else:
        best = None
    if best is None:
        print("no MusicBrainz match kept: YouTube metadata only")
        return None
    return mb.metadata(best)


def cmd_lookup(args: argparse.Namespace) -> int:
    meta = lookup(args.url, interactive=not args.no_input)
    if meta:
        print(f"  {meta.artist} - {meta.title} ({meta.year or '?'})")
        print(f"  genres: {', '.join(meta.genres) or '-'}")
        print(f"  cover: {meta.cover_url or '-'}")
        print(f"  https://musicbrainz.org/recording/{meta.recording_id}")
    return 0 if meta else 1


def cmd_add(args: argparse.Namespace) -> int:
    from indexer.catalog import LocalCatalog
    from indexer.pipeline import add_song

    catalog = LocalCatalog(args.catalog)
    mb = MusicBrainz()
    counts: dict[str, int] = {}
    for i, url in enumerate(args.urls, 1):
        t0 = time.perf_counter()
        print(f"[{i}/{len(args.urls)}] {url}")
        try:
            res = add_song(url, catalog, mb=mb, interactive=not args.no_input, source=args.source)
        except Exception as e:  # one bad link must not stop a batch
            counts["error"] = counts.get("error", 0) + 1
            print(f"  error: {type(e).__name__}: {e}")
            continue
        counts[res.status] = counts.get(res.status, 0) + 1
        s = res.song
        print(f"  {res.status}: #{s['id']} {s['artist']} - {s['title']} ({s['year'] or '?'})")
        print(f"  {res.detail}  [{time.perf_counter() - t0:.0f} s]")
    summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
    print()
    print(f"{len(catalog.songs)} songs in {catalog.root}  |  {summary}")
    return 0 if not counts.get("error") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m indexer")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("add", help="add YouTube songs to the local catalogue")
    p.add_argument("urls", nargs="+")
    p.add_argument("--catalog", type=Path, default=Path("data") / "catalog")
    p.add_argument(
        "--source", default="manual", choices=["playlist", "chart", "proposal", "manual"]
    )
    p.add_argument(
        "--no-input", action="store_true", help="never ask; YouTube metadata when unsure"
    )
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("lookup", help="find a YouTube video's song in MusicBrainz")
    p.add_argument("url")
    p.add_argument("--no-input", action="store_true", help="never ask; skip when unsure")
    p.set_defaults(func=cmd_lookup)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
