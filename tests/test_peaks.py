"""T03: spectral peak detection (constellation map)."""

import numpy as np

from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import (
    MAX_PEAKS_PER_SECOND,
    N_FFT,
    find_peaks,
    frames_per_second,
    spectrogram,
)


def tones(freqs, duration_s=3.0, sr=SAMPLE_RATE):
    t = np.arange(int(duration_s * sr)) / sr
    y = sum(0.2 * np.sin(2 * np.pi * f * t) for f in freqs)
    return y.astype(np.float32)


def hz_to_bin(f):
    return round(f * N_FFT / SAMPLE_RATE)


def test_peaks_format():
    peaks = find_peaks(spectrogram(tones([1000])))
    assert peaks.ndim == 2 and peaks.shape[1] == 2
    assert peaks.dtype == np.int32
    # sorted by frame, then bin
    order = np.lexsort((peaks[:, 1], peaks[:, 0]))
    assert np.array_equal(order, np.arange(len(peaks)))


def test_three_tones_give_peaks_on_those_three_frequencies():
    freqs = [440, 1000, 2500]
    peaks = find_peaks(spectrogram(tones(freqs)))
    expected = np.array([hz_to_bin(f) for f in freqs])

    # every peak sits on one of the three tones
    dist = np.abs(peaks[:, 1][:, None] - expected[None, :]).min(axis=1)
    assert dist.max() <= 1
    # and every tone gets peaks
    for b in expected:
        assert np.any(np.abs(peaks[:, 1] - b) <= 1)


def test_silence_gives_no_peaks():
    y = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)
    assert len(find_peaks(spectrogram(y))) == 0


def test_density_is_capped_on_white_noise():
    rng = np.random.default_rng(0)
    y = (0.3 * rng.standard_normal(5 * SAMPLE_RATE)).astype(np.float32)
    S = spectrogram(y)
    peaks = find_peaks(S)
    seconds = S.shape[1] / frames_per_second()
    assert len(peaks) / seconds <= MAX_PEAKS_PER_SECOND * 1.05
    assert len(peaks) > 0


def test_plot_peaks_writes_png(tmp_path):
    import soundfile as sf

    from earworm.tools.plot_peaks import main

    wav = tmp_path / "tones.wav"
    sf.write(wav, tones([440, 1000], duration_s=4.0), SAMPLE_RATE)
    out = main([str(wav), "--start", "0", "--duration", "3", "--out", str(tmp_path / "p.png")])
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
