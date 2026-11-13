"""T07: helpers of the robustness bench."""

import numpy as np
import pytest

from earworm import fingerprint as fp
from earworm.audio import SAMPLE_RATE
from earworm.bench import (
    Trial,
    babble_noise,
    choose_thresholds,
    degrade,
    fingerprint_params,
    mix_at_snr,
    outcome,
    pink_noise,
    small_speaker,
)


@pytest.mark.parametrize("snr", [0, 5, 10, 15])
def test_mix_at_snr(snr):
    rng = np.random.default_rng(0)
    s = rng.standard_normal(50_000).astype(np.float32)
    n = pink_noise(50_000, rng)
    mixed = mix_at_snr(s, n, snr)
    noise_part = mixed - s
    measured = 10 * np.log10(np.mean(s**2) / np.mean(noise_part**2))
    assert measured == pytest.approx(snr, abs=0.01)


def band_power(y, lo, hi):
    spec = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SAMPLE_RATE)
    return spec[(f >= lo) & (f < hi)].sum()


def test_pink_noise_loses_3db_per_octave():
    y = pink_noise(2**18, np.random.default_rng(1))
    assert np.sqrt(np.mean(y**2)) == pytest.approx(1, abs=1e-3)
    # same power per octave for 1/f noise
    ratio = band_power(y, 200, 400) / band_power(y, 1600, 3200)
    assert 0.7 < ratio < 1.4


def test_babble_is_speech_band_and_bursty():
    y = babble_noise(5 * SAMPLE_RATE, np.random.default_rng(2))
    assert band_power(y, 300, 3000) > 10 * band_power(y, 4500, 5500)
    # a single talker comes in bursts (six together are almost steady, like a real room)
    one = babble_noise(5 * SAMPLE_RATE, np.random.default_rng(2), talkers=1)
    frame_rms = np.sqrt(np.mean(one[: 200 * 250].reshape(200, 250) ** 2, axis=1))
    assert frame_rms.std() / frame_rms.mean() > 0.4


def test_small_speaker_removes_bass():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    bass = np.sin(2 * np.pi * 60 * t).astype(np.float32)
    mid = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    assert np.std(small_speaker(bass)) < 0.1 * np.std(bass)
    assert np.std(small_speaker(mid)) > 0.9 * np.std(mid)


def test_degrade_never_clips():
    rng = np.random.default_rng(3)
    y = np.clip(rng.standard_normal(SAMPLE_RATE), -1, 1).astype(np.float32)
    out = degrade(y, "babble", 0, rng)
    assert np.abs(out).max() <= 1.0 and out.dtype == np.float32


def test_fingerprint_params_restores_values():
    before = fp.FAN_OUT
    with fingerprint_params(FAN_OUT=before + 7):
        assert fp.FAN_OUT == before + 7
    assert fp.FAN_OUT == before


def trial(song, pos_song, pos_score, pos_ratio, neg_score, neg_ratio, snr=5):
    return Trial(song, "pink", snr, pos_song, pos_score, pos_ratio, neg_score, neg_ratio)


def test_outcome_counts():
    ts = [
        trial(1, 1, 100, 20.0, 4, 1.0),  # recognised
        trial(2, 2, 8, 1.1, 5, 1.2),  # too weak: missed
        trial(3, 7, 50, 5.0, 50, 5.0),  # wrong song found, and FP when absent
    ]
    r = outcome(ts, min_score=15, min_ratio=2.0)[("pink", 5)]
    assert r["n"] == 3
    assert r["acc"] == pytest.approx(1 / 3)
    assert r["wrong"] == 1
    assert r["fp"] == 1


def test_choose_thresholds_excludes_chance_matches():
    ts = [trial(i, i, 30 + i, 10.0, 6 + i % 3, 1.5) for i in range(20)]
    s, r = choose_thresholds(ts)
    assert all(not (t.neg_score >= s and t.neg_ratio >= r) for t in ts)
    assert all(t.pos_score >= s and t.pos_ratio >= r for t in ts)


def test_room_reverb_keeps_level_and_smears():
    from earworm.bench import room_reverb

    rng = np.random.default_rng(5)
    click = np.zeros(SAMPLE_RATE, dtype=np.float32)
    click[100] = 1.0
    out = room_reverb(click, rng)
    assert np.mean(out**2) == pytest.approx(np.mean(click**2), rel=1e-3)
    assert np.abs(out[2000:]).max() > 1e-3  # energy long after the click


def test_chance_score_quantile_grows_with_catalogue():
    from earworm.bench import chance_score_quantile

    rng = np.random.default_rng(6)
    ts = [Trial(0, "pink", 5, 0, 99, 9.0, 3, 1.0, tuple(rng.poisson(3, 9))) for _ in range(200)]
    small, big = chance_score_quantile(ts, 10), chance_score_quantile(ts, 1000)
    assert small < big


def test_false_positive_rate_uses_ratio_to_second_best():
    from earworm.bench import false_positive_rate

    # two other songs; each scores 0 or 10 with probability 1/2
    pmf = np.zeros(11)
    pmf[0] = pmf[10] = 0.5
    # best >= 10 with ratio >= 2 only when exactly one scores 10: P = 1/2
    assert false_positive_rate(pmf, 3, 10, 2.0, draws=20_000) == pytest.approx(0.5, abs=0.02)
    # everyone tied at 3: the ratio test never passes
    tied = np.zeros(4)
    tied[3] = 1.0
    assert false_positive_rate(tied, 50, 2, 2.0, draws=2000) == 0.0
