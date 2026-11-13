# Earworm

Music recognition from ~10 s of microphone audio, using hand-written audio fingerprints (spectrogram peaks + hashing), plus similar songs (pretrained audio embeddings) and a predicted genre (custom classifier).

> Work in progress. See [`docs/spec.md`](docs/spec.md) (French) and [`docs/plan.md`](docs/plan.md).

**Audio policy:** audio is downloaded locally only to be analysed and is deleted right after. Only fingerprints and embeddings are kept. No audio is ever stored in the repository or served by the app.

## Development setup

Requires Python 3.12.

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Check everything works:

```bash
pytest
ruff check .
```

## Try it locally

Put a few audio files in `data/` (git-ignored), then:

```bash
python -m earworm index data
python -m earworm identify "data/Some Song.mp3" --start 60 --duration 10
python -m earworm identify my_clip.mp3
```

`index` writes `data/local_index.npz`; `identify` prints the song, where the excerpt sits in it, and the match score (exit code 1 when nothing matches).

To see what the fingerprinter "looks at" (spectrogram + selected peaks):

```bash
python -m earworm.tools.plot_peaks "data/Some Song.mp3" --start 30
```
