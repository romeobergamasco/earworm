"""T04: pairing peaks into hashes, and the blob format."""

import numpy as np
import pytest

from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import (
    FAN_OUT,
    MAX_DT,
    from_bytes,
    hashes,
    pack_hash,
    to_bytes,
    unpack_hash,
)


def test_pack_hash_bit_layout():
    # f1 (10 bits) | f2 (10 bits) | dt (6 bits)
    assert pack_hash(186, 464, 5) == (186 << 16) | (464 << 6) | 5
    assert pack_hash(1023, 1023, 63) == (1 << 26) - 1
    assert unpack_hash(pack_hash(186, 464, 5)) == (186, 464, 5)


def test_hashes_from_hand_built_peaks():
    # (frame, bin), sorted by frame
    peaks = np.array([[10, 100], [12, 120], [15, 90], [200, 300]], dtype=np.int32)
    h = hashes(peaks)
    assert h.dtype == np.uint32 and h.shape[1] == 2

    expected = {
        (pack_hash(100, 120, 2), 10),
        (pack_hash(100, 90, 5), 10),
        (pack_hash(120, 90, 3), 12),
        # 200 is too far from the others (dt > MAX_DT): no pair
    }
    assert set(map(tuple, h.tolist())) == expected


def test_same_frame_peaks_are_not_paired():
    peaks = np.array([[10, 100], [10, 200]], dtype=np.int32)
    assert len(hashes(peaks)) == 0


def test_fan_out_limits_pairs_per_anchor():
    # one anchor followed by many close peaks
    peaks = np.array([[0, 100]] + [[k, 100 + k] for k in range(1, 30)], dtype=np.int32)
    h = hashes(peaks)
    from_anchor0 = h[h[:, 1] == 0]
    assert len(from_anchor0) == FAN_OUT


def test_pairs_respect_max_dt():
    peaks = np.array([[0, 100], [MAX_DT, 110], [MAX_DT + 1, 120]], dtype=np.int32)
    h = hashes(peaks)
    pairs = {(unpack_hash(int(x))[2], int(o)) for x, o in h}
    assert (MAX_DT, 0) in pairs
    assert all(dt <= MAX_DT for dt, _ in pairs)


def test_hashes_sorted_and_deterministic():
    rng = np.random.default_rng(0)
    frames = np.sort(rng.integers(0, 500, 300))
    peaks = np.stack([frames, rng.integers(20, 900, 300)], axis=1).astype(np.int32)
    peaks = peaks[np.lexsort((peaks[:, 1], peaks[:, 0]))]
    a, b = hashes(peaks), hashes(peaks.copy())
    assert np.array_equal(a, b)
    order = np.lexsort((a[:, 1], a[:, 0]))
    assert np.array_equal(order, np.arange(len(a)))


def test_same_audio_gives_same_hashes():
    from earworm.fingerprint import fingerprint

    rng = np.random.default_rng(1)
    y = (0.2 * rng.standard_normal(5 * SAMPLE_RATE)).astype(np.float32)
    assert np.array_equal(fingerprint(y), fingerprint(y.copy()))


def test_blob_round_trip():
    rng = np.random.default_rng(2)
    h = np.stack(
        [np.sort(rng.integers(0, 1 << 26, 5000)), rng.integers(0, 20000, 5000)], axis=1
    ).astype(np.uint32)
    blob = to_bytes(h)
    assert isinstance(blob, bytes)
    assert np.array_equal(from_bytes(blob), h)


def test_blob_round_trip_empty():
    h = np.zeros((0, 2), dtype=np.uint32)
    assert np.array_equal(from_bytes(to_bytes(h)), h)


def test_from_bytes_rejects_garbage():
    with pytest.raises(ValueError):
        from_bytes(b"not a fingerprint blob")


def test_blob_is_compressed():
    rng = np.random.default_rng(3)
    h = np.stack(
        [np.sort(rng.integers(0, 1 << 26, 20000)), rng.integers(0, 8000, 20000)], axis=1
    ).astype(np.uint32)
    assert len(to_bytes(h)) < 8 * len(h) * 0.75


def test_to_bytes_rejects_unsorted():
    h = np.array([[5, 0], [3, 1]], dtype=np.uint32)
    with pytest.raises(ValueError):
        to_bytes(h)
