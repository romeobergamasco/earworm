"""Local indexer command line.

python -m indexer lookup <YouTube link>   metadata only (no audio download)
"""

import argparse
import sys

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m indexer")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("lookup", help="find a YouTube video's song in MusicBrainz")
    p.add_argument("url")
    p.add_argument("--no-input", action="store_true", help="never ask; skip when unsure")
    args = parser.parse_args(argv)

    meta = lookup(args.url, interactive=not args.no_input)
    if meta:
        print(f"  {meta.artist} - {meta.title} ({meta.year or '?'})")
        print(f"  genres: {', '.join(meta.genres) or '-'}")
        print(f"  cover: {meta.cover_url or '-'}")
        print(f"  https://musicbrainz.org/recording/{meta.recording_id}")
    return 0 if meta else 1


if __name__ == "__main__":
    sys.exit(main())
