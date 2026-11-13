"""Local command line.

python -m earworm index <folder> [--out data/local_index.npz]
python -m earworm identify <files or folders> [--label L] [--start S] [--duration D]
python -m earworm listen [--trials 10] [--label calm] [--seconds 10]
"""

import argparse
import csv
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


def _identify_one(idx: Index, path: Path, start_s: float, duration_s: float | None) -> dict:
    y, sr = load(path)
    if start_s or duration_s:
        start = int(start_s * sr)
        end = len(y) if duration_s is None else start + int(duration_s * sr)
        y = y[start:end]
    if len(y) < SAMPLE_RATE:
        raise ValueError("excerpt too short (need at least 1 s)")
    t0 = time.perf_counter()
    matches = idx.query(fingerprint(y))
    ms = (time.perf_counter() - t0) * 1000
    best = matches[0] if matches else None
    found = best is not None and best.is_confident
    return {
        "file": path.name,
        "result": idx.title(best.song_id) if found else "",
        "score": best.score if best else 0,
        "ratio": round(best.ratio, 2) if best else 0.0,
        "offset_s": round(best.offset_s, 1) if found else "",
        "ms": round(ms),
    }


def cmd_identify(args: argparse.Namespace) -> int:
    idx = Index.load(args.index)
    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXT)
        else:
            files.append(p)
    if not files:
        print("no audio files given", file=sys.stderr)
        return 2

    rows = []
    for path in files:
        try:
            r = _identify_one(idx, path, args.start, args.duration)
        except ValueError as e:
            print(f"{path.name}: {e}", file=sys.stderr)
            return 2
        rows.append(r)
        if len(files) > 1:
            print(f"== {path.name}")
        if r["result"]:
            print(r["result"])
            print(
                f"  at {r['offset_s']:.1f} s in the song, score {r['score']}, "
                f"ratio {r['ratio']:.1f}  [{r['ms']} ms]"
            )
        else:
            print(f"Not found (best guess score {r['score']}).  [{r['ms']} ms]")

    if len(files) > 1:
        n_found = sum(bool(r["result"]) for r in rows)
        print()
        print(f"{n_found}/{len(rows)} found")
    if args.label:
        log = Path("results") / "identify_log.csv"
        log.parent.mkdir(parents=True, exist_ok=True)
        new = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["label", *rows[0]])
            if new:
                w.writeheader()
            w.writerows({"label": args.label, **r} for r in rows)
        print(f"logged -> {log}")
    return 0 if all(r["result"] for r in rows) else 1


def cmd_listen(args: argparse.Namespace) -> int:
    from earworm.listen import run_session

    idx = Index.load(args.index)
    print(f"index: {idx.n_songs} songs")
    save = Path("results") / "recordings" if args.save else None
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    from functools import partial

    from earworm.listen import record

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    recorder = partial(record, device=device)
    run_session(idx, args.trials, args.seconds, args.label, recorder=recorder, save_dir=save)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m earworm", description="Earworm, local tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="fingerprint every audio file of a folder")
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("identify", help="find which indexed song an audio file comes from")
    p.add_argument("paths", type=Path, nargs="+", help="audio files and/or folders")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--label", default=None, help="log results to results/identify_log.csv")
    p.add_argument("--start", type=float, default=0.0, help="seconds to skip")
    p.add_argument("--duration", type=float, default=None, help="seconds to keep")
    p.set_defaults(func=cmd_identify)

    p = sub.add_parser("listen", help="record from the microphone and identify")
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--label", default="test", help="e.g. calm / noisy, written to the log")
    p.add_argument("--save", action="store_true", help="keep the recordings in results/recordings")
    p.add_argument(
        "--device", default=None, help="input device number or name (see --list-devices)"
    )
    p.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    p.set_defaults(func=cmd_listen)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
