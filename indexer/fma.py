"""FMA-small, the public reference set used to compare embeddings (T18) and
genre classifiers (T21) — https://github.com/mdeff/fma

8 balanced genres of 30 s clips under free licences. Only the chosen clips are
downloaded, by reading the official zip files partially (HTTP ranges); each
clip is deleted as soon as it has been used.

    for clip in clips(sample(per_genre=50), work_dir): ...
"""

import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

METADATA_ZIP = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
SMALL_ZIP = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"


@dataclass(frozen=True)
class Track:
    track_id: int
    genre: str
    artist: str  # FMA artist id, as text (for artist-grouped splits)


def sample(per_genre: int, work: Path, seed: int = 0) -> list[Track]:
    """`per_genre` random tracks of each genre of FMA-small (same seed, same tracks)."""
    import pandas as pd
    from remotezip import RemoteZip

    work.mkdir(parents=True, exist_ok=True)
    csv = work / "fma_tracks.csv"
    if not csv.exists():
        with RemoteZip(METADATA_ZIP) as z:
            csv.write_bytes(z.read("fma_metadata/tracks.csv"))
    tracks = pd.read_csv(csv, index_col=0, header=[0, 1], low_memory=False)
    small = tracks[tracks[("set", "subset")] == "small"]
    pick = small.groupby(("track", "genre_top")).sample(n=per_genre, random_state=seed)
    return [
        Track(int(t), str(row[("track", "genre_top")]), str(row[("artist", "id")]))
        for t, row in pick.iterrows()
    ]


def clips(tracks: list[Track]) -> Iterator[tuple[Track, Path | None]]:
    """Each track with the path of its mp3, valid until the next one (then deleted).
    The path is None when the file cannot be read (a few FMA files are corrupt)."""
    from remotezip import RemoteZip

    with RemoteZip(SMALL_ZIP) as z, tempfile.TemporaryDirectory(prefix="earworm-fma-") as tmp:
        for t in tracks:
            name = f"fma_small/{t.track_id:06d}"[:13] + f"/{t.track_id:06d}.mp3"
            try:
                path = Path(z.extract(name, tmp))
            except Exception:
                yield t, None
                continue
            try:
                yield t, path
            finally:
                path.unlink(missing_ok=True)
