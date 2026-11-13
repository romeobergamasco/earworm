"""T05 acceptance on the user's own audio files in data/ (skipped if absent).

data/ is git-ignored: these tests only run on the developer's machine.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from earworm.audio import SAMPLE_RATE, load
from earworm.fingerprint import HOP_LENGTH, fingerprint
from earworm.index import Index

DATA = Path(__file__).resolve().parents[1] / "data"
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
FILES = sorted(p for p in DATA.glob("*") if p.suffix.lower() in AUDIO_EXT) if DATA.is_dir() else []

pytestmark = pytest.mark.skipif(len(FILES) < 2, reason="needs audio files in data/")


@pytest.fixture(scope="module")
def library():
    signals = {i: load(p)[0] for i, p in enumerate(FILES)}
    idx = Index()
    for i, y in signals.items():
        idx.add(i, fingerprint(y))
    idx.freeze()
    return idx, signals


def test_clean_10s_excerpts_are_recognised_with_offset(library):
    idx, signals = library
    rng = np.random.default_rng(42)
    n = 10 * SAMPLE_RATE
    failures, times = [], []
    for sid, y in signals.items():
        start = int(rng.integers(0, len(y) - n))
        q = fingerprint(y[start : start + n])
        t0 = time.perf_counter()
        res = idx.query(q)
        times.append(time.perf_counter() - t0)
        expected_frame = start / HOP_LENGTH
        if not res or res[0].song_id != sid or abs(res[0].offset - expected_frame) > 1:
            failures.append((FILES[sid].name, start / SAMPLE_RATE, res[:2]))
    assert not failures, failures
    assert max(times) < 0.2, f"slowest query {max(times) * 1000:.0f} ms"


def test_query_with_fingerprinting_under_200ms(library):
    idx, signals = library
    y = signals[0]
    excerpt = y[len(y) // 3 : len(y) // 3 + 10 * SAMPLE_RATE]
    fingerprint(excerpt)  # warm-up
    t0 = time.perf_counter()
    idx.query(fingerprint(excerpt))
    assert time.perf_counter() - t0 < 0.2
