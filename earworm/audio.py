"""Audio loading: any audio file -> mono float32 at SAMPLE_RATE."""

from pathlib import Path

import numpy as np

# librosa (heavy) is imported inside load(): the server only needs SAMPLE_RATE.

# 11 025 Hz keeps everything up to ~5.5 kHz, where the robust spectral peaks live,
# and makes the fingerprinting 4x cheaper than at 44.1 kHz.
SAMPLE_RATE = 11025

# Formats libsndfile reads natively; everything else (m4a from phone voice
# recorders, webm / opus / m4a from YouTube...) goes through PyAV, which ships
# its own FFmpeg, so no system install is needed.
_SOUNDFILE_EXT = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


def _decode_with_av(path: Path) -> tuple[np.ndarray, int]:
    import av

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"no audio stream in {path}")
        sr = stream.rate
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
        chunks = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray()[0])
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray()[0])
    y = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return y.astype(np.float32), int(sr)


def load(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 signal resampled to `sr`.

    Returns (signal, sample_rate).
    """
    import librosa

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in _SOUNDFILE_EXT:
        y, _ = librosa.load(path, sr=sr, mono=True)
    else:
        y, native = _decode_with_av(path)
        if native != sr:
            y = librosa.resample(y, orig_sr=native, target_sr=sr)
    return y.astype(np.float32, copy=False), int(sr)
