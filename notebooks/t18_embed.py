"""T18: segment embeddings (CLAP and MERT) of every catalogue song that has
MusicBrainz genres, for the comparison in notebooks/01_embeddings.ipynb.

    python notebooks/t18_embed.py [--limit N] [--pause 20]

For each song: download the audio into a temporary folder (deleted right
after), cut it into up to 6 segments of 10 s, embed them with both models and
save only the vectors in data/embeddings/t18/<song_id>.npz (git-ignored, no
audio). Already-done songs are skipped, so the script can be stopped and resumed.

YouTube answers "Sign in to confirm you're not a bot" after many downloads in a
row (first run: blocked after 23 songs). Hence a pause between songs, a 10 min
wait when it happens, and a stop after 3 blocks in a row.
"""

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from earworm import audio  # noqa: E402
from earworm.config import load_env, require  # noqa: E402
from earworm.db import Supabase  # noqa: E402
from earworm.embeddings import ClapEmbedder, MertEmbedder, segments  # noqa: E402
from earworm.remote import fetch_songs  # noqa: E402
from indexer.fetch import downloaded_audio  # noqa: E402

OUT = ROOT / "data" / "embeddings" / "t18"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pause", type=float, default=20, help="seconds between two downloads")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # titles with non-ASCII characters

    load_env()
    cfg = require("SUPABASE_URL", "SUPABASE_SECRET_KEY")
    songs = [s for s in fetch_songs(Supabase(cfg["SUPABASE_URL"], cfg["SUPABASE_SECRET_KEY"]))
             if s["mb_genres"]]  # fmt: skip
    random.Random(0).shuffle(songs)  # partial results stay representative
    songs = songs[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "songs.json").write_text(json.dumps(songs, ensure_ascii=False, indent=0), "utf-8")

    todo = [s for s in songs if not (OUT / f"{s['id']}.npz").exists()]
    print(f"{len(songs)} songs with genres, {len(todo)} to do", flush=True)
    clap, mert = ClapEmbedder(), MertEmbedder()
    blocked_in_a_row = 0
    for i, song in enumerate(todo, 1):
        if i > 1:
            time.sleep(args.pause)
        t0 = time.perf_counter()
        try:
            url = f"https://www.youtube.com/watch?v={song['youtube_id']}"
            with downloaded_audio(url) as dl:
                t_dl = time.perf_counter() - t0
                y48, sr = audio.load(dl.path, clap.sr)
            y24 = resample_poly(y48, 1, 2).astype(np.float32)  # 48 kHz -> 24 kHz
            c = clap.embed(segments(y48, clap.sr))
            m = mert.embed(segments(y24, mert.sr))
            np.savez(
                OUT / f"{song['id']}.npz",
                clap=c.vectors.astype(np.float16),
                mert=m.vectors.astype(np.float16),
                clap_s=c.seconds,
                mert_s=m.seconds,
                duration_s=len(y48) / sr,
            )
            name = f"{song['id']} {song['artist']} - {song['title']}"
            took = f"dl {t_dl:.0f}s clap {c.seconds:.1f}s mert {m.seconds:.1f}s"
            print(f"[{i}/{len(todo)}] {name}: {took}", flush=True)
            blocked_in_a_row = 0
        except Exception as e:  # one bad video must not stop the run
            print(f"[{i}/{len(todo)}] {song['id']} FAILED: {str(e)[:150]}", flush=True)
            if "not a bot" in str(e):
                blocked_in_a_row += 1
                if blocked_in_a_row >= 3:
                    print("YouTube keeps blocking: stopping (run again later)", flush=True)
                    return 1
                print("blocked by YouTube: waiting 10 min", flush=True)
                time.sleep(600)
            else:
                traceback.print_exc(limit=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
