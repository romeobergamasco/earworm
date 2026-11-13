"""Synthetic robustness bench (T07).

Takes 10 s excerpts of the songs in a folder, degrades them the way a phone
in a noisy room would, and measures recognition and false positives.

Degradation chain for each excerpt:
    small-speaker high-pass (250 Hz) -> room reverb (RT60 0.5 s)
    -> + noise at a given SNR -> random gain (-20..0 dB)
Noise types:
    pink    1/f noise, a stand-in for wind / traffic / ventilation
    babble  speech-shaped noise with syllable-rate bursts, a stand-in for chatter

Negatives come for free: the index scores each song independently, so the
results of a query *with its true song removed* are exactly what a catalogue
that does not contain the song would return (leave-one-out). The scores that
the other songs get by chance also give their distribution, from which the
false-positive rate of a much larger catalogue is extrapolated.

Usage:
    python -m earworm.bench [data] [--per-song 10] [--sweep] [--md out.md]
"""

import argparse
import contextlib
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from earworm import fingerprint as fp
from earworm import index as index_mod
from earworm.audio import SAMPLE_RATE, load
from earworm.index import Index

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
SNRS = (None, 10, 5, 0, -5, -10)
CATALOGUE_SIZES = (10, 100, 1000, 5000)  # None = no added noise
NOISES = ("pink", "babble")
EXCERPT_S = 10.0


# --- Signal helpers -----------------------------------------------------------


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f noise (power falls 3 dB per octave), unit RMS."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    y = np.fft.irfft(spec / np.sqrt(f), n)
    return (y / np.sqrt(np.mean(y**2))).astype(np.float32)


def babble_noise(n: int, rng: np.random.Generator, sr: int = SAMPLE_RATE, talkers: int = 6):
    """Several 'talkers': speech-band noise switched on and off at syllable rate. Unit RMS."""
    sos = butter(2, [200, 3500], btype="bandpass", fs=sr, output="sos")
    total = np.zeros(n)
    for _ in range(talkers):
        voice = sosfilt(sos, rng.standard_normal(n))
        # syllables ~4 per second: random on/off, smoothed
        steps = rng.random(int(n / sr * 4) + 2) < 0.6
        env = np.repeat(steps.astype(float), sr // 4 + 1)[:n]
        env = np.convolve(env, np.hanning(sr // 20) / (sr // 40), mode="same")
        total += voice * env * rng.uniform(0.5, 1.0)
    return (total / np.sqrt(np.mean(total**2))).astype(np.float32)


def small_speaker(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Phone or laptop speaker: almost no bass below ~250 Hz."""
    sos = butter(2, 250, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, y).astype(np.float32)


def room_reverb(y: np.ndarray, rng, sr: int = SAMPLE_RATE, rt60: float = 0.5) -> np.ndarray:
    """Synthetic room: direct sound + exponentially decaying diffuse tail."""
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    rir = 0.4 * tail / np.sqrt(np.sum(tail**2))
    rir[0] += 1.0
    out = fftconvolve(y, rir)[: len(y)]
    return (out * np.sqrt(np.mean(y**2) / max(np.mean(out**2), 1e-12))).astype(np.float32)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """signal + noise scaled so that 10 log10(P_signal / P_noise) = snr_db."""
    p_sig = np.mean(signal.astype(np.float64) ** 2)
    p_noise = np.mean(noise.astype(np.float64) ** 2)
    scale = np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
    return (signal + scale * noise).astype(np.float32)


def degrade(clean, noise_kind, snr_db, rng, sr=SAMPLE_RATE):
    y = room_reverb(small_speaker(clean, sr), rng, sr)
    if snr_db is not None:
        gen = pink_noise if noise_kind == "pink" else babble_noise
        y = mix_at_snr(y, gen(len(y), rng), snr_db)
    y = y * 10 ** (rng.uniform(-20, 0) / 20)
    peak = np.abs(y).max()
    return (y / peak * 0.99 if peak > 1 else y).astype(np.float32)  # no clipping


# --- Parameters -----------------------------------------------------------------


@contextlib.contextmanager
def fingerprint_params(**overrides):
    """Temporarily change module-level fingerprint parameters (bench only)."""
    old = {k: getattr(fp, k) for k in overrides}
    for k, v in overrides.items():
        setattr(fp, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(fp, k, v)


# --- Running --------------------------------------------------------------------


@dataclass
class Trial:
    song: int
    noise: str
    snr: float | None
    pos_song: int  # best song with the true song in the index
    pos_score: int
    pos_ratio: float
    neg_score: int  # best song with the true song removed
    neg_ratio: float
    chance: tuple[int, ...] = ()  # best-bin score of every other song


def _scores_without(matches, song_id):
    others = [m for m in matches if m.song_id != song_id]
    if not others:
        return 0, 0.0
    second = others[1].score if len(others) > 1 else 0
    return others[0].score, others[0].score / max(1, second)


def run(signals: list[np.ndarray], per_song: int, seed: int = 0) -> tuple[list[Trial], dict]:
    t0 = time.perf_counter()
    idx = Index()
    for sid, y in enumerate(signals):
        idx.add(sid, fp.fingerprint(y))
    idx.freeze()
    t_index = time.perf_counter() - t0

    rng = np.random.default_rng(seed)
    n = int(EXCERPT_S * SAMPLE_RATE)
    trials, q_hashes = [], []
    for sid, y in enumerate(signals):
        for _ in range(per_song):
            start = int(rng.integers(int(0.05 * len(y)), int(0.95 * len(y)) - n))
            clean = y[start : start + n]
            for noise, snr in itertools.product(NOISES, SNRS):
                if snr is None and noise != NOISES[0]:
                    continue  # one clean condition is enough
                q = fp.fingerprint(degrade(clean, noise, snr, rng))
                q_hashes.append(len(q))
                res = idx.query(q, top_k=len(signals))
                best = res[0] if res else None
                neg_score, neg_ratio = _scores_without(res, sid)
                trials.append(
                    Trial(
                        song=sid,
                        noise="clean" if snr is None else noise,
                        snr=snr,
                        pos_song=best.song_id if best else -1,
                        pos_score=best.score if best else 0,
                        pos_ratio=best.ratio if best else 0.0,
                        neg_score=neg_score,
                        neg_ratio=neg_ratio,
                        chance=tuple(m.score for m in res if m.song_id != sid)
                        + (0,) * (len(signals) - 1 - sum(m.song_id != sid for m in res)),
                    )
                )
    info = {
        "hashes_per_s": idx.n_hashes / (sum(len(y) for y in signals) / SAMPLE_RATE),
        "query_hashes": float(np.median(q_hashes)),
        "index_s": t_index,
        "total_s": time.perf_counter() - t0,
    }
    return trials, info


# --- Scoring --------------------------------------------------------------------


def outcome(trials: list[Trial], min_score: float, min_ratio: float) -> dict:
    """Per-condition accuracy, misidentifications and false-positive rate."""
    out = {}
    conds = sorted(
        {(t.noise, t.snr) for t in trials}, key=lambda c: (c[0], -(99 if c[1] is None else c[1]))
    )
    for cond in conds:
        ts = [t for t in trials if (t.noise, t.snr) == cond]
        found = [t.pos_score >= min_score and t.pos_ratio >= min_ratio for t in ts]
        correct = sum(f and t.pos_song == t.song for f, t in zip(found, ts, strict=True))
        wrong = sum(f and t.pos_song != t.song for f, t in zip(found, ts, strict=True))
        fp_neg = sum(t.neg_score >= min_score and t.neg_ratio >= min_ratio for t in ts)
        out[cond] = {"n": len(ts), "acc": correct / len(ts), "wrong": wrong, "fp": fp_neg}
    return out


def choose_thresholds(trials: list[Trial], max_fp_rate: float = 0.0, target_snr: float = 5):
    """For each ratio threshold, the lowest score threshold with no wrong song
    and a false-positive rate <= max_fp_rate; keep the pair that recognises the
    most excerpts at `target_snr`. No safety margin here: see docs/results.md."""
    neg = np.array([(t.neg_score, t.neg_ratio) for t in trials])
    target = [t for t in trials if t.snr == target_snr]
    best, best_acc = None, -1.0
    for r in (1.5, 2.0, 2.5, 3.0, 4.0):
        for s in range(3, 300):
            fp_rate = np.mean((neg[:, 0] >= s) & (neg[:, 1] >= r))
            wrong = sum(
                t.pos_score >= s and t.pos_ratio >= r and t.pos_song != t.song for t in trials
            )
            if fp_rate <= max_fp_rate and wrong == 0:
                acc = np.mean(
                    [t.pos_score >= s and t.pos_ratio >= r and t.pos_song == t.song for t in target]
                )
                if acc > best_acc:
                    best, best_acc = (s, r), acc
                break
    return best


def chance_score_pmf(trials: list[Trial], max_score: int = 400) -> np.ndarray:
    """Probability that one absent song gets chance score s, for s = 0..max_score.

    Measured on the bench where there is data; beyond that, the tail of
    log10 P(score >= s) is extrapolated linearly (see chance_score_quantile)."""
    pooled = np.concatenate([np.array(t.chance) for t in trials if t.chance])
    ss = np.arange(max_score + 2)
    surv = np.array([np.mean(pooled >= x) for x in ss])  # P(score >= s)
    counts = surv * len(pooled)
    tail = (surv <= 0.05) & (counts >= 10)
    if tail.sum() >= 2:
        slope, intercept = np.polyfit(ss[tail], np.log10(surv[tail]), 1)
        start = int(ss[tail].max())
        fitted = 10 ** (intercept + slope * ss)
        surv[start:] = np.minimum(surv[start], fitted[start:])
    pmf = surv[:-1] - surv[1:]
    return np.clip(pmf, 0, None) / np.clip(pmf, 0, None).sum()


def false_positive_rate(
    pmf: np.ndarray,
    catalogue: int,
    min_score: int,
    min_ratio: float,
    draws: int = 20_000,
    seed: int = 0,
) -> float:
    """Monte Carlo: a query for a song absent from a `catalogue`-song index; the
    other songs score independently from `pmf`. Rate at which the best one passes
    BOTH thresholds (score and ratio to the second best)."""
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < draws:
        n = min(2000, draws - done)
        x = rng.choice(len(pmf), size=(n, catalogue - 1), p=pmf).astype(np.int32)
        top2 = -np.partition(-x, 1, axis=1)[:, :2]
        best, second = top2[:, 0], top2[:, 1]
        hits += int(np.sum((best >= min_score) & (best >= min_ratio * np.maximum(second, 1))))
        done += n
    return hits / draws


def min_safe_score(
    trials: list[Trial], catalogue: int, min_ratio: float, max_fp: float = 0.001
) -> int:
    """Lowest score threshold whose simulated false-positive rate is <= max_fp."""
    pmf = chance_score_pmf(trials)
    for s in range(2, len(pmf)):
        if false_positive_rate(pmf, catalogue, s, min_ratio, draws=5000) <= max_fp:
            # confirm with more draws
            if false_positive_rate(pmf, catalogue, s, min_ratio) <= max_fp:
                return s
    return len(pmf)


def chance_score_quantile(trials: list[Trial], catalogue: int, fp_rate: float = 0.01) -> int:
    """Score that the best of `catalogue - 1` absent songs reaches with probability
    <= fp_rate (ratio test ignored, so conservative).

    Songs score independently, so P(max >= s) = 1 - (1 - q(s)) ** (catalogue - 1),
    where q(s) = P(one absent song scores >= s). q is measured on the bench, but a
    10-song bench cannot observe q below ~1e-4, so its tail is extrapolated:
    log10 q(s) is close to linear in s (fitted where q <= 5 % and >= 10 samples).
    """
    pooled = np.concatenate([np.array(t.chance) for t in trials if t.chance])
    target_q = 1 - (1 - fp_rate) ** (1 / (catalogue - 1))
    ss = np.arange(1, int(pooled.max()) + 1)
    q = np.array([np.mean(pooled >= x) for x in ss])
    counts = q * len(pooled)
    tail = (q <= 0.05) & (counts >= 10)
    if tail.sum() >= 2:
        slope, intercept = np.polyfit(ss[tail], np.log10(q[tail]), 1)
        if slope < 0:
            return int(np.ceil((np.log10(target_q) - intercept) / slope))
    return int(pooled.max()) + 1  # not enough tail data: fall back to "above anything seen"


def markdown_table(res: dict) -> str:
    lines = ["| condition | SNR | n | recognised | wrong song | false positives (song absent) |"]
    lines.append("|---|---|---|---|---|---|")
    for (noise, snr), r in res.items():
        snr_txt = "-" if snr is None else f"{snr} dB"
        lines.append(
            f"| {noise} | {snr_txt} | {r['n']} | {100 * r['acc']:.0f} % | {r['wrong']} | "
            f"{r['fp']} / {r['n']} |"
        )
    return "\n".join(lines)


def load_folder(folder: Path) -> tuple[list[str], list[np.ndarray]]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
    return [p.stem for p in files], [load(p)[0] for p in files]


SWEEP = {
    "MAX_PEAKS_PER_SECOND": (20, 30, 45),
    "FAN_OUT": (3, 5, 8),
}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Earworm robustness bench")
    parser.add_argument("folder", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--per-song", type=int, default=10)
    parser.add_argument("--sweep", action="store_true", help="try several parameter sets")
    parser.add_argument("--seed", type=int, default=0, help="excerpt and noise draw")
    parser.add_argument(
        "--fixed", action="store_true", help="use the thresholds of earworm.index, do not tune"
    )
    parser.add_argument("--md", type=Path, default=None, help="write the markdown report here")
    args = parser.parse_args(argv)

    names, signals = load_folder(args.folder)
    print(f"{len(signals)} songs, {args.per_song} excerpts each\n")
    report = []

    configs = (
        [dict(zip(SWEEP, v, strict=True)) for v in itertools.product(*SWEEP.values())]
        if args.sweep
        else [{}]
    )
    for cfg in configs:
        with fingerprint_params(**cfg):
            trials, info = run(signals, args.per_song, seed=args.seed)
        # Thresholds: ratio >= 2, and the lowest score whose simulated false-positive
        # rate stays <= 0.05 % for every catalogue size from 10 to 5000 songs (small
        # catalogues are the risky ones: the ratio test protects less), and that the
        # 10-song bench itself needs for zero false positives / wrong songs.
        thr = choose_thresholds(trials)
        s_bench = thr[0] if thr else index_mod.MIN_SCORE
        r = 2.0
        safe = {n: min_safe_score(trials, n, r, max_fp=0.0005) for n in CATALOGUE_SIZES}
        s = max(s_bench, *safe.values())
        if args.fixed:
            s, r = index_mod.MIN_SCORE, index_mod.MIN_RATIO
        res = outcome(trials, s, r)
        label = ", ".join(f"{k}={v}" for k, v in cfg.items()) or "current parameters"
        acc = {
            snr: np.mean([v["acc"] for (_, sn), v in res.items() if sn == snr])
            for snr in SNRS
            if snr is not None
        }
        print(
            f"{label}: {info['hashes_per_s']:.0f} hashes/s, score>={s} ratio>={r} | "
            + "  ".join(f"{k} dB {100 * a:.0f}%" for k, a in acc.items())
            + f"  ({info['total_s']:.0f} s)"
        )
        neg = sorted(t.neg_score for t in trials)
        block = [
            f"### {label}",
            "",
            f"- {info['hashes_per_s']:.0f} hashes/s stored, "
            f"~{info['query_hashes']:.0f} per 10 s query",
            "- thresholds "
            + ("(fixed, earworm.index)" if args.fixed else "(simulated, see below)")
            + f": score >= {s}, ratio >= {r}",
            "- lowest safe score (ratio >= 2, <= 0.05 % false positives) by catalogue size: "
            + ", ".join(f"{n} songs -> {v}" for n, v in safe.items())
            + f"; the 10-song bench alone needs {s_bench}",
            f"- best chance score of an absent song: max {neg[-1]}, 99th percentile "
            f"{neg[int(0.99 * (len(neg) - 1))]}",
            "",
            markdown_table(res),
            "",
        ]
        report.append("\n".join(block))
        if not args.sweep:
            print("\n" + "\n".join(block))

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text("\n".join(report), encoding="utf-8")
        print(f"\nreport -> {args.md}")


if __name__ == "__main__":
    main()
