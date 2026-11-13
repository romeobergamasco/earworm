"""T02: audio loading and spectrogram."""

import numpy as np
import pytest
import soundfile as sf

from earworm.audio import SAMPLE_RATE, load
from earworm.fingerprint import HOP_LENGTH, N_FFT, bin_to_hz, spectrogram


def sine(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


@pytest.fixture
def stereo_wav(tmp_path):
    sr = 22050
    left = sine(1000, 3.0, sr)
    right = sine(1000, 3.0, sr)
    path = tmp_path / "tone.wav"
    sf.write(path, np.stack([left, right], axis=1), sr)
    return path


def test_load_returns_mono_float32_at_target_rate(stereo_wav):
    y, sr = load(stereo_wav)
    assert sr == SAMPLE_RATE == 11025
    assert y.ndim == 1
    assert y.dtype == np.float32
    assert abs(len(y) - 3.0 * SAMPLE_RATE) <= 1


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.wav")


def test_spectrogram_shape():
    y = sine(1000, 2.0, SAMPLE_RATE)
    S = spectrogram(y)
    assert S.shape == (N_FFT // 2 + 1, 1 + len(y) // HOP_LENGTH)
    assert S.dtype == np.float32


def test_spectrogram_peak_at_1000_hz():
    y = sine(1000, 2.0, SAMPLE_RATE)
    S = spectrogram(y)
    peak_bin = int(np.argmax(S.mean(axis=1)))
    expected_bin = round(1000 * N_FFT / SAMPLE_RATE)
    assert abs(peak_bin - expected_bin) <= 1
    assert abs(bin_to_hz(peak_bin) - 1000) < SAMPLE_RATE / N_FFT


def test_spectrogram_rejects_empty_signal():
    with pytest.raises(ValueError):
        spectrogram(np.zeros(0, dtype=np.float32))


def test_mp3_and_wav_give_same_shape(tmp_path):
    sr = 44100
    rng = np.random.default_rng(0)
    # 1000 Hz tone + light noise, 5 s
    y = sine(1000, 5.0, sr) + 0.05 * rng.standard_normal(5 * sr).astype(np.float32)
    wav, mp3 = tmp_path / "x.wav", tmp_path / "x.mp3"
    sf.write(wav, y, sr)
    sf.write(mp3, y, sr, format="MP3")

    S_wav = spectrogram(load(wav)[0])
    S_mp3 = spectrogram(load(mp3)[0])
    assert S_wav.shape == S_mp3.shape
    # Same dominant frequency in both
    assert np.argmax(S_wav.mean(axis=1)) == np.argmax(S_mp3.mean(axis=1))
