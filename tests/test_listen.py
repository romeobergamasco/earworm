"""T08: microphone session, with a fake recorder (no sound card needed)."""

import csv

import numpy as np
from scipy.signal import resample_poly

from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import fingerprint
from earworm.index import Index
from earworm.listen import level_dbfs, run_session, to_model_rate
from tests.test_cli import song


def test_to_model_rate_from_44100():
    t = np.arange(44100) / 44100
    y = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    out = to_model_rate(y, 44100)
    assert out.dtype == np.float32
    assert abs(len(out) - SAMPLE_RATE) <= 1
    spec = np.abs(np.fft.rfft(out))
    assert abs(np.argmax(spec) * SAMPLE_RATE / len(out) - 1000) < 2


def test_level_dbfs():
    assert level_dbfs(np.full(1000, 0.1, dtype=np.float32)) == np.float32(-20.0)


def test_session_logs_trials(tmp_path, capsys):
    songs = {1: song(0), 2: song(1)}
    idx = Index()
    for sid, y in songs.items():
        idx.add(sid, fingerprint(y), title=f"Song {sid}")
    idx.freeze()

    takes = iter(
        [
            # 10 s of song 2 as a 44.1 kHz "microphone" would deliver it
            (resample_poly(songs[2][5 * SAMPLE_RATE : 15 * SAMPLE_RATE], 4, 1), 44100),
            (0.01 * np.random.default_rng(0).standard_normal(441000).astype(np.float32), 44100),
        ]
    )
    answers = iter(["", "y", "", "n"])
    log = tmp_path / "log.csv"
    rows = run_session(
        idx,
        2,
        10,
        "calm",
        recorder=lambda s: next(takes),
        ask=lambda _: next(answers),
        log_path=log,
    )
    assert rows[0]["result"] == "Song 2" and rows[0]["correct"] == "yes"
    assert rows[1]["result"] == "" and rows[1]["correct"] == "no"
    with open(log, encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 2
    assert "1/2 right" in capsys.readouterr().out
