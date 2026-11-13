"""Local command line.

python -m earworm index <folder> [--out data/local_index.npz]
python -m earworm identify <file> [--index data/local_index.npz] [--start S] [--duration D]
"""

import argparse
import sys
import time
from pathlib import Path

from earworm.audio import SAMPLE_RATE, load
from earworm.fingerprint import fingerprint
from earworm.index import Index

DEFAULT_INDEX = Path("data") / "local_index.npz"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".webm"}


def cmd_index(args: argparse.Namespace) -> int:
    files = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        print(f"no audio files in {args.folder}", file=sys.stderr)
        return 2
    idx = Index()
    t0 = time.perf_counter()
    for song_id, path in enumerate(files, start=1):
        y, _ = load(path)
        h = fingerprint(y)
        idx.add(song_id, h, title=path.stem)
        print(f"[{song_id}/{len(files)}] {path.stem}  ({len(h)} hashes)")
    idx.freeze()
    idx.save(args.out)
    print(
        f"indexed {idx.n_songs} songs, {idx.n_hashes} hashes "
        f"in {time.perf_counter() - t0:.1f} s -> {args.out}"
    )
    return 0


def cmd_identify(args: argparse.Namespace) -> int:
    idx = Index.load(args.index)
    y, sr = load(args.file)
    if args.start or args.duration:
        start = int(args.start * sr)
        end = len(y) if args.duration is None else start + int(args.duration * sr)
        y = y[start:end]
    if len(y) < SAMPLE_RATE:
        print("excerpt too short (need at least 1 s)", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    matches = idx.query(fingerprint(y))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not matches or not matches[0].is_confident:
        best = f" (best guess score {matches[0].score})" if matches else ""
        print(f"Not found{best}.  [{elapsed_ms:.0f} ms]")
        return 1
    best = matches[0]
    print(f"{idx.title(best.song_id)}")
    print(
        f"  at {best.offset_s:.1f} s in the song, score {best.score}, "
        f"ratio {best.ratio:.1f}  [{elapsed_ms:.0f} ms]"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m earworm", description="Earworm, local tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="fingerprint every audio file of a folder")
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("identify", help="find which indexed song an audio file comes from")
    p.add_argument("file", type=Path)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--start", type=float, default=0.0, help="seconds to skip")
    p.add_argument("--duration", type=float, default=None, help="seconds to keep")
    p.set_defaults(func=cmd_identify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
