"""T06: index persistence and the `python -m earworm` command line."""

import numpy as np
import pytest
import soundfile as sf

from earworm.__main__ import main
from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import PARAMS_VERSION
from earworm.index import Index


def song(seed, seconds=40):
    """A reproducible 'song': random tones over pink-ish noise."""
    rng = np.random.default_rng(seed)
    n = seconds * SAMPLE_RATE
    t = np.arange(n) / SAMPLE_RATE
    y = 0.05 * np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    for start in range(0, seconds, 1):
        f = rng.uniform(200, 3000)
        seg = slice(start * SAMPLE_RATE, (start + 1) * SAMPLE_RATE)
        y[seg] += 0.3 * np.sin(2 * np.pi * f * t[seg])
    y += 0.02 * rng.standard_normal(n)
    return y.astype(np.float32)


@pytest.fixture
def library(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    names = ["Alpha - First", "Beta - Second", "Gamma - Third"]
    for seed, name in enumerate(names):
        sf.write(lib / f"{name}.wav", song(seed), SAMPLE_RATE)
    (lib / "notes.txt").write_text("not audio")
    return lib, names


def test_save_and_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    idx = Index()
    for sid in (1, 2):
        h = np.stack([np.sort(rng.integers(0, 1 << 26, 100)), rng.integers(0, 500, 100)], 1)
        idx.add(sid, h.astype(np.uint32), title=f"song {sid}")
    idx.freeze()
    path = tmp_path / "i.npz"
    idx.save(path)

    loaded = Index.load(path)
    assert loaded.n_songs == 2 and loaded.n_hashes == 200
    assert loaded.title(2) == "song 2"
    r = idx._rows[:20]
    q = np.stack([r >> np.uint64(32), r & np.uint64(0xFFFF)], axis=1).astype(np.uint32)
    assert loaded.query(q)[0] == idx.query(q)[0]


def test_load_rejects_other_params_version(tmp_path, monkeypatch):
    idx = Index()
    idx.add(1, np.array([[1, 0]], dtype=np.uint32))
    idx.freeze()
    path = tmp_path / "i.npz"
    idx.save(path)
    monkeypatch.setattr("earworm.index.PARAMS_VERSION", PARAMS_VERSION + 1)
    with pytest.raises(ValueError, match="re-index"):
        Index.load(path)


def test_cli_index_then_identify(library, tmp_path, capsys):
    lib, names = library
    out = tmp_path / "idx.npz"
    assert main(["index", str(lib), "--out", str(out)]) == 0
    assert out.is_file()
    assert "3 songs" in capsys.readouterr().out

    # excerpt cut from the second song, saved as its own file
    y = song(1)[12 * SAMPLE_RATE : 22 * SAMPLE_RATE]
    clip = tmp_path / "clip.wav"
    sf.write(clip, y, SAMPLE_RATE)
    assert main(["identify", str(clip), "--index", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Beta - Second" in printed
    assert "12." in printed  # offset in seconds


def test_cli_identify_with_start_and_duration(library, tmp_path, capsys):
    lib, _ = library
    out = tmp_path / "idx.npz"
    main(["index", str(lib), "--out", str(out)])
    capsys.readouterr()
    full = lib / "Gamma - Third.wav"
    args = ["identify", str(full), "--index", str(out), "--start", "5", "--duration", "10"]
    assert main(args) == 0
    assert "Gamma - Third" in capsys.readouterr().out


def test_cli_unknown_song_is_not_found(library, tmp_path, capsys):
    lib, _ = library
    out = tmp_path / "idx.npz"
    main(["index", str(lib), "--out", str(out)])
    capsys.readouterr()
    clip = tmp_path / "unknown.wav"
    sf.write(clip, song(99)[: 10 * SAMPLE_RATE], SAMPLE_RATE)
    assert main(["identify", str(clip), "--index", str(out)]) == 1
    assert "not found" in capsys.readouterr().out.lower()


def test_cli_identify_folder(library, tmp_path, capsys):
    lib, _ = library
    out = tmp_path / "idx.npz"
    main(["index", str(lib), "--out", str(out)])
    clips = tmp_path / "clips"
    clips.mkdir()
    sf.write(clips / "a.wav", song(0)[3 * SAMPLE_RATE : 13 * SAMPLE_RATE], SAMPLE_RATE)
    sf.write(clips / "b.wav", song(99)[: 10 * SAMPLE_RATE], SAMPLE_RATE)
    capsys.readouterr()
    assert main(["identify", str(clips), "--index", str(out)]) == 1
    printed = capsys.readouterr().out
    assert "Alpha - First" in printed and "1/2 found" in printed
