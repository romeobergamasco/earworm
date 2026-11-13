"""Record from the computer's microphone and identify (T08).

    python -m earworm listen [--trials 10] [--label calm] [--seconds 10]

Each trial: press Enter, play the song near the mic, wait for the result, then
say whether it was right. Every trial is appended to results/listen_log.csv.
Recordings are not kept unless --save is given (they stay local, git-ignored).
"""

import csv
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import fingerprint
from earworm.index import Index, Match

LOG = Path("results") / "listen_log.csv"


def record(seconds: float, device: int | str | None = None) -> tuple[np.ndarray, int]:
    """Record mono audio at the device's native rate (default input device if None)."""
    import sounddevice as sd  # imported here: only this command needs PortAudio

    sr = int(sd.query_devices(device, kind="input")["default_samplerate"])
    y = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    return y[:, 0], sr


def to_model_rate(y: np.ndarray, sr: int) -> np.ndarray:
    """Resample to SAMPLE_RATE with a polyphase filter (no librosa needed)."""
    if sr == SAMPLE_RATE:
        return y.astype(np.float32)
    g = np.gcd(sr, SAMPLE_RATE)
    return resample_poly(y, SAMPLE_RATE // g, sr // g).astype(np.float32)


def level_dbfs(y: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(y.astype(np.float64) ** 2)), 1e-9)))


def identify_recording(
    idx: Index, y: np.ndarray, sr: int
) -> tuple[Match | None, float, Match | None]:
    """(confident match or None, processing time in ms, best guess even if not confident)."""
    t0 = time.perf_counter()
    matches = idx.query(fingerprint(to_model_rate(y, sr)))
    ms = (time.perf_counter() - t0) * 1000
    best = matches[0] if matches else None
    return (best if best and best.is_confident else None), ms, best


def run_session(
    idx: Index,
    trials: int,
    seconds: float,
    label: str,
    recorder: Callable[[float], tuple[np.ndarray, int]] = record,
    ask: Callable[[str], str] = input,
    log_path: Path = LOG,
    save_dir: Path | None = None,
) -> list[dict]:
    rows = []
    for i in range(1, trials + 1):
        ask(f"\n[{i}/{trials}] Start the music, then press Enter to listen {seconds:.0f} s... ")
        print("  listening...", flush=True)
        y, sr = recorder(seconds)
        found, ms, best = identify_recording(idx, y, sr)
        level = level_dbfs(y)
        if found:
            print(
                f"  -> {idx.title(found.song_id)}  (score {found.score}, ratio {found.ratio:.1f})"
            )
        else:
            guess = f" (best guess score {best.score})" if best else ""
            print(f"  -> not found{guess}")
        print(f"     level {level:.0f} dBFS, processing {ms:.0f} ms")
        if level < -85:  # digital silence; a raw mic in a quiet room is around -80..-60
            print("     (no sound at all: try another --device, see --list-devices)")
        answer = ask("  Was it the right song? [y/n, Enter to skip] ").strip().lower()
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "label": label,
            "result": idx.title(found.song_id) if found else "",
            "score": best.score if best else 0,
            "ratio": round(best.ratio, 2) if best else 0.0,
            "level_dbfs": round(level, 1),
            "ms": round(ms),
            "correct": {"y": "yes", "o": "yes", "n": "no"}.get(answer[:1], ""),
        }
        rows.append(row)
        if save_dir is not None:
            import soundfile as sf

            save_dir.mkdir(parents=True, exist_ok=True)
            sf.write(save_dir / f"{label}_{row['time'].replace(':', '-')}.wav", y, sr)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        if new:
            w.writeheader()
        w.writerows(rows)

    judged = [r for r in rows if r["correct"]]
    right = sum(r["correct"] == "yes" for r in judged)
    print(f"\n{label}: {right}/{len(judged)} right ({len(rows)} trials) -> {log_path}")
    return rows
