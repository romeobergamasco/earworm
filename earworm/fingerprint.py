"""Fingerprinting: spectrogram (T02), then peaks (T03) and hashes (T04)."""

import librosa
import numpy as np

from earworm.audio import SAMPLE_RATE

N_FFT = 2048  # ~186 ms window at 11 025 Hz, frequency resolution ~5.4 Hz
HOP_LENGTH = 512  # ~46 ms between frames
AMIN = 1e-5  # floor before log, i.e. -100 dB


def spectrogram(y: np.ndarray) -> np.ndarray:
    """Magnitude STFT in dB, shape (N_FFT // 2 + 1, n_frames).

    Absolute dB (ref = 1.0), not normalised to the loudest bin, so that the
    peak thresholds of T03 behave the same on loud and quiet recordings.
    """
    if y.size == 0:
        raise ValueError("empty signal")
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann", center=True)
    db = librosa.amplitude_to_db(np.abs(stft), ref=1.0, amin=AMIN, top_db=None)
    return db.astype(np.float32, copy=False)


def bin_to_hz(bin_index: int | np.ndarray, sr: int = SAMPLE_RATE) -> float | np.ndarray:
    return bin_index * sr / N_FFT


def frame_to_seconds(frame: int | np.ndarray, sr: int = SAMPLE_RATE) -> float | np.ndarray:
    return frame * HOP_LENGTH / sr
