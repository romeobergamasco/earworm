"""Save a PNG of a spectrogram with its peaks (constellation map) on top.

Usage: python -m earworm.tools.plot_peaks <audio file> [--start 30] [--duration 15] [--out file.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, file output only
import matplotlib.pyplot as plt  # noqa: E402

from earworm import fingerprint as fp  # noqa: E402
from earworm.audio import load  # noqa: E402


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--start", type=float, default=30.0, help="start of the window, seconds")
    parser.add_argument("--duration", type=float, default=15.0, help="window length, seconds")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    y, sr = load(args.audio)
    S = fp.spectrogram(y)
    peaks = fp.find_peaks(S)

    fps = fp.frames_per_second(sr)
    f0 = int(args.start * fps)
    f1 = min(S.shape[1], int((args.start + args.duration) * fps))
    if f0 >= f1:
        raise SystemExit(f"window starts after the end of the file ({S.shape[1] / fps:.0f} s)")
    max_bin = int(fp.MAX_FREQ_HZ * fp.N_FFT / sr) + 1
    in_win = (peaks[:, 0] >= f0) & (peaks[:, 0] < f1)
    p = peaks[in_win]

    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)
    ax.imshow(
        S[:max_bin, f0:f1],
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(fp.frame_to_seconds(f0), fp.frame_to_seconds(f1), 0, fp.bin_to_hz(max_bin)),
        vmin=S[:max_bin, f0:f1].max() - 70,
    )
    ax.scatter(
        fp.frame_to_seconds(p[:, 0]),
        fp.bin_to_hz(p[:, 1]),
        s=10,
        c="cyan",
        edgecolors="black",
        linewidths=0.3,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    density = len(peaks) / (S.shape[1] / fps)
    ax.set_title(f"{args.audio.stem} - {len(p)} peaks in window, {density:.1f} peaks/s overall")
    fig.tight_layout()

    out = args.out or Path("results") / "plots" / f"{args.audio.stem}_peaks.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(out)
    return out


if __name__ == "__main__":
    main()
