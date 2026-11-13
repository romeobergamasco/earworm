"""Fingerprinting: spectrogram (T02), then peaks (T03) and hashes (T04)."""

import struct
import zlib

import numpy as np
from scipy.ndimage import maximum_filter

from earworm.audio import SAMPLE_RATE

# Bump whenever a parameter below changes: stored fingerprints become
# incompatible and the catalogue must be re-indexed.
PARAMS_VERSION = 1

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

# --- Hashing (T04) ----------------------------------------------------------
# Each peak (the "anchor") is paired with the next FAN_OUT peaks that fall in
# its target zone: 1 <= dt <= MAX_DT frames later, |df| <= MAX_DF bins.
# A pair becomes a 26-bit hash  f1 (10 bits) | f2 (10 bits) | dt (6 bits),
# stored with the anchor's frame (its offset in the song).
FAN_OUT = 5
MAX_DT = 63  # 6 bits, ~2.9 s
MAX_DF = 200  # bins, ~1 kHz
_SEARCH = 48  # how many following peaks to look at when filling the fan-out
_F_BITS, _DT_BITS = 10, 6


_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann, as librosa


def spectrogram(y: np.ndarray) -> np.ndarray:
    """Magnitude STFT in dB, shape (N_FFT // 2 + 1, n_frames), n_frames = 1 + len(y) // HOP.

    Plain numpy (same output as librosa.stft with center=True, zero padding):
    no first-call warm-up, and the server does not need librosa.
    Absolute dB (ref = 1.0), not normalised to the loudest bin, so that the
    peak thresholds behave the same on loud and quiet recordings.
    """
    if y.size == 0:
        raise ValueError("empty signal")
    padded = np.pad(np.asarray(y, dtype=np.float32), N_FFT // 2)
    frames = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)[::HOP_LENGTH]
    mag = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)).T
    return (20.0 * np.log10(np.maximum(mag, AMIN))).astype(np.float32)


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


def pack_hash(f1: int, f2: int, dt: int) -> int:
    return (f1 << (_F_BITS + _DT_BITS)) | (f2 << _DT_BITS) | dt


def unpack_hash(h: int) -> tuple[int, int, int]:
    mask_f, mask_dt = (1 << _F_BITS) - 1, (1 << _DT_BITS) - 1
    return (h >> (_F_BITS + _DT_BITS)) & mask_f, (h >> _DT_BITS) & mask_f, h & mask_dt


def hashes(peaks: np.ndarray) -> np.ndarray:
    """Pair peaks into hashes.

    `peaks`: (n, 2) int array of (frame, bin), sorted by frame.
    Returns a uint32 array of shape (m, 2): columns (hash, anchor_frame),
    sorted by hash then frame.
    """
    n = len(peaks)
    frames = peaks[:, 0].astype(np.int64)
    bins = peaks[:, 1].astype(np.int64)
    if bins.size and (bins.min() < 0 or bins.max() >= 1 << _F_BITS):
        raise ValueError("peak bin does not fit in 10 bits")

    anchors, targets = [], []
    taken = np.zeros(n, dtype=np.int64)  # pairs already made per anchor
    idx = np.arange(n)
    for k in range(1, min(_SEARCH, n - 1) + 1):
        i, j = idx[: n - k], idx[k:]
        dt = frames[j] - frames[i]
        ok = (dt >= 1) & (dt <= MAX_DT) & (np.abs(bins[j] - bins[i]) <= MAX_DF)
        ok &= taken[i] < FAN_OUT
        taken[i[ok]] += 1
        anchors.append(i[ok])
        targets.append(j[ok])

    if not anchors:
        return np.zeros((0, 2), dtype=np.uint32)
    a = np.concatenate(anchors)
    t = np.concatenate(targets)
    h = (bins[a] << (_F_BITS + _DT_BITS)) | (bins[t] << _DT_BITS) | (frames[t] - frames[a])
    out = np.stack([h, frames[a]], axis=1).astype(np.uint32)
    return out[np.lexsort((out[:, 1], out[:, 0]))]


def fingerprint(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Signal -> (hash, anchor_frame) array. The whole pipeline in one call."""
    return hashes(find_peaks(spectrogram(y), sr))


# --- Blob format ------------------------------------------------------------
# b"EWFP" | version (uint8) | count (uint32) | zlib( hash deltas uint32[count]
#                                                   | anchor frames uint16[count] )
# All little-endian. Hashes are sorted, so their deltas are small and compress
# well: measured on 10 real songs, ~2.9x smaller than raw (hash, frame) pairs,
# i.e. ~80 MB instead of ~240 MB for 1000 songs (see docs/spec.md, section 4).
_MAGIC = b"EWFP"
_BLOB_VERSION = 1
_HEADER = struct.Struct("<4sBI")
_MAX_FRAME = (1 << 16) - 1  # ~50 min of audio, far above the 12 min limit


def to_bytes(h: np.ndarray) -> bytes:
    h = np.asarray(h, dtype=np.uint32)
    if len(h) and np.any(np.diff(h[:, 0].astype(np.int64)) < 0):
        raise ValueError("hashes must be sorted")
    if len(h) and h[:, 1].max() > _MAX_FRAME:
        raise ValueError("anchor frame does not fit in 16 bits")
    deltas = np.diff(h[:, 0], prepend=np.uint32(0)).astype("<u4")
    body = deltas.tobytes() + h[:, 1].astype("<u2").tobytes()
    return _HEADER.pack(_MAGIC, _BLOB_VERSION, len(h)) + zlib.compress(body, 9)


def from_bytes(blob: bytes) -> np.ndarray:
    if len(blob) < _HEADER.size:
        raise ValueError("blob too short")
    magic, version, count = _HEADER.unpack_from(blob)
    if magic != _MAGIC or version != _BLOB_VERSION:
        raise ValueError("not a fingerprint blob (bad magic or version)")
    try:
        body = zlib.decompress(blob[_HEADER.size :])
    except zlib.error as e:
        raise ValueError(f"corrupt fingerprint blob: {e}") from e
    if len(body) != 6 * count:
        raise ValueError("blob length does not match its header")
    hashes_ = np.cumsum(np.frombuffer(body, dtype="<u4", count=count), dtype=np.uint32)
    frames = np.frombuffer(body, dtype="<u2", offset=4 * count, count=count)
    return np.stack([hashes_, frames.astype(np.uint32)], axis=1)
