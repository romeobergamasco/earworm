"""T05: in-memory index and offset-histogram matching (synthetic hashes)."""

import numpy as np
import pytest

from earworm.index import Index, Match


def fake_song(rng, n_hashes=3000, n_frames=4000):
    h = rng.integers(0, 1 << 26, n_hashes)
    f = rng.integers(0, n_frames, n_hashes)
    out = np.stack([h, f], axis=1).astype(np.uint32)
    return out[np.lexsort((out[:, 1], out[:, 0]))]


def excerpt(song_hashes, start, length, keep=1.0, rng=None):
    """Hashes of song frames [start, start+length), re-based to the excerpt."""
    sel = (song_hashes[:, 1] >= start) & (song_hashes[:, 1] < start + length)
    q = song_hashes[sel].copy()
    q[:, 1] -= start
    if keep < 1.0:
        q = q[rng.random(len(q)) < keep]
    return q


@pytest.fixture
def index_and_songs():
    rng = np.random.default_rng(0)
    songs = {sid: fake_song(rng) for sid in range(1, 11)}
    idx = Index()
    for sid, h in songs.items():
        idx.add(sid, h)
    idx.freeze()
    return idx, songs, rng


def test_finds_right_song_and_offset(index_and_songs):
    idx, songs, _ = index_and_songs
    q = excerpt(songs[7], start=1200, length=215)
    best = idx.query(q)[0]
    assert isinstance(best, Match)
    assert best.song_id == 7
    assert best.offset == 1200
    assert best.score >= 0.9 * len(q)
    assert best.ratio > 5


def test_survives_losing_most_hashes(index_and_songs):
    idx, songs, rng = index_and_songs
    q = excerpt(songs[3], start=500, length=215, keep=0.1, rng=rng)
    noise = np.stack([rng.integers(0, 1 << 26, 2000), rng.integers(0, 215, 2000)], axis=1).astype(
        np.uint32
    )
    best = idx.query(np.concatenate([q, noise]))[0]
    assert best.song_id == 3 and best.offset == 500


def test_tolerates_one_frame_jitter(index_and_songs):
    idx, songs, rng = index_and_songs
    q = excerpt(songs[5], start=800, length=215)
    # half of the hashes land one frame late (frame-grid misalignment)
    late = rng.random(len(q)) < 0.5
    q[late, 1] += 1
    best = idx.query(q)[0]
    assert best.song_id == 5
    assert abs(best.offset - 800) <= 1
    assert best.score >= 0.9 * len(q)


def test_unknown_audio_scores_low(index_and_songs):
    idx, _, rng = index_and_songs
    q = fake_song(rng, n_hashes=1400, n_frames=215)
    results = idx.query(q)
    assert not results or results[0].score < 10


def test_results_sorted_and_limited(index_and_songs):
    idx, songs, _ = index_and_songs
    q = excerpt(songs[2], start=100, length=215)
    res = idx.query(q, top_k=3)
    assert len(res) <= 3
    assert all(a.score >= b.score for a, b in zip(res, res[1:], strict=False))
    assert len({m.song_id for m in res}) == len(res)


def test_empty_query_and_empty_index():
    idx = Index()
    idx.freeze()
    assert idx.query(np.zeros((0, 2), dtype=np.uint32)) == []
    assert idx.query(np.array([[1, 0]], dtype=np.uint32)) == []


def test_query_before_freeze_raises():
    idx = Index()
    idx.add(1, np.array([[1, 0]], dtype=np.uint32))
    with pytest.raises(RuntimeError):
        idx.query(np.array([[1, 0]], dtype=np.uint32))


def test_add_same_song_twice_raises():
    idx = Index()
    idx.add(1, np.array([[1, 0]], dtype=np.uint32))
    with pytest.raises(ValueError):
        idx.add(1, np.array([[2, 0]], dtype=np.uint32))


def test_stats():
    idx = Index()
    idx.add(1, np.array([[1, 0], [2, 3]], dtype=np.uint32))
    idx.add(2, np.array([[5, 0]], dtype=np.uint32))
    idx.freeze()
    assert idx.n_songs == 2
    assert idx.n_hashes == 3


def test_large_song_ids():
    rng = np.random.default_rng(4)
    idx = Index()
    songs = {sid: fake_song(rng, n_hashes=500) for sid in (3, 5000, 2_000_000)}
    for sid, h in songs.items():
        idx.add(sid, h)
    idx.freeze()
    q = excerpt(songs[2_000_000], start=300, length=215)
    best = idx.query(q)[0]
    assert best.song_id == 2_000_000 and best.offset == 300
