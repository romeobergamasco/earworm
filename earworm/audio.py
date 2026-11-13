"""Audio loading: any file librosa can read -> mono float32 at SAMPLE_RATE."""

from pathlib import Path

import librosa
import numpy as np

# 11 025 Hz keeps everything up to ~5.5 kHz, where the robust spectral peaks live,
# and makes the fingerprinting 4x cheaper than at 44.1 kHz.
SAMPLE_RATE = 11025


def load(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 signal resampled to `sr`.

    Returns (signal, sample_rate).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    y, out_sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32, copy=False), int(out_sr)
