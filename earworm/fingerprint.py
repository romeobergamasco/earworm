"""Fingerprinting: spectrogram (T02), then peaks (T03) and hashes (T04)."""

import librosa
import numpy as np
from scipy.ndimage import maximum_filter

from earworm.audio import SAMPLE_RATE

N_FFT = 2048  # ~186 ms window at 11 025 Hz, frequency resolution ~5.4 Hz
HOP_LENGTH = 512  # ~46 ms between frames
AMIN = 1e-5  # floor before log, i.e. -100 dB

# --- Peak picking (T03) ---------------------------------------------------
# A peak is a bin that is the maximum of its neighbourhood (frequency x time),
# louder than the typical level of its frame by PEAK_MARGIN_DB, and above an
# absolute floor so that silence yields nothing. Then, to keep the density
# predictable, only the MAX_PEAKS_PER_SECOND strongest peaks of each second
# are kept.
PEAK_NEIGHBORHOOD_BINS = 21  # ~113 Hz
PEAK_NEIGHBORHOOD_FRAMES = 11  # ~0.5 s
PEAK_MARGIN_DB = 10.0
PEAK_FLOOR_DB = -40.0
MIN_FREQ_HZ = 100.0  # phone speakers and mics barely reproduce lower bass
MAX_FREQ_HZ = 5000.0
MAX_PEAKS_PER_SECOND = 30


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


def frames_per_second(sr: int = SAMPLE_RATE) -> float:
    return sr / HOP_LENGTH


def find_peaks(S: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Pick the spectral peaks of a dB spectrogram.

    Returns an int32 array of shape (n_peaks, 2): columns (frame, bin),
    sorted by frame then bin.
    """
    lo = int(np.ceil(MIN_FREQ_HZ * N_FFT / sr))
    hi = int(np.floor(MAX_FREQ_HZ * N_FFT / sr)) + 1
    band = S[lo:hi]

    local_max = maximum_filter(
        band,
        size=(PEAK_NEIGHBORHOOD_BINS, PEAK_NEIGHBORHOOD_FRAMES),
        mode="constant",
        cval=-np.inf,
    )
    frame_level = np.median(band, axis=0, keepdims=True)
    is_peak = (band == local_max) & (band >= frame_level + PEAK_MARGIN_DB) & (band >= PEAK_FLOOR_DB)
    bins, frames = np.nonzero(is_peak)
    amps = band[bins, frames]
    bins = bins + lo

    # Density cap: keep the strongest peaks within each block of ~1 s.
    block_len = max(1, round(frames_per_second(sr)))
    per_block = max(1, round(MAX_PEAKS_PER_SECOND * block_len / frames_per_second(sr)))
    block = frames // block_len
    order = np.lexsort((-amps, block))  # by block, loudest first
    block_sorted = block[order]
    starts = np.searchsorted(block_sorted, block_sorted, side="left")
    rank = np.arange(len(order)) - starts
    keep = order[rank < per_block]

    peaks = np.stack([frames[keep], bins[keep]], axis=1).astype(np.int32)
    return peaks[np.lexsort((peaks[:, 1], peaks[:, 0]))]
