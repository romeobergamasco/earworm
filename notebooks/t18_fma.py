"""T18, second data set: CLAP and MERT segment embeddings of an FMA-small sample
(no YouTube involved). Same recipe as the catalogue (earworm.embeddings).

    python notebooks/t18_fma.py [--per-genre 50]

Writes data/embeddings/t18_fma/<track_id>.npz (vectors only) and tracks.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earworm import audio  # noqa: E402
from earworm.embeddings import ClapEmbedder, MertEmbedder, segments  # noqa: E402
from indexer.fma import clips, sample  # noqa: E402

OUT = ROOT / "data" / "embeddings" / "t18_fma"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-genre", type=int, default=50)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    tracks = sample(args.per_genre, ROOT / "data" / "fma")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tracks.json").write_text(json.dumps([t.__dict__ for t in tracks], indent=0), "utf-8")
    todo = [t for t in tracks if not (OUT / f"{t.track_id}.npz").exists()]
    print(f"{len(tracks)} FMA tracks, {len(todo)} to do", flush=True)
    clap, mert = ClapEmbedder(), MertEmbedder()
    for i, (t, path) in enumerate(clips(todo), 1):
        if path is None:
            print(f"[{i}/{len(todo)}] {t.track_id} unreadable, skipped", flush=True)
            continue
        try:
            y48, sr = audio.load(path, clap.sr)
            y24 = resample_poly(y48, 1, 2).astype(np.float32)
            c = clap.embed(segments(y48, clap.sr))
            m = mert.embed(segments(y24, mert.sr))
        except Exception as e:  # corrupt mp3
            print(f"[{i}/{len(todo)}] {t.track_id} failed: {e}", flush=True)
            continue
        np.savez(OUT / f"{t.track_id}.npz", clap=c.vectors.astype(np.float16),
                 mert=m.vectors.astype(np.float16), clap_s=c.seconds, mert_s=m.seconds)  # fmt: skip
        if i % 25 == 0:
            print(f"[{i}/{len(todo)}] {time.strftime('%H:%M:%S')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
