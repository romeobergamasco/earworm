"""Genre labels (T20): MusicBrainz genres -> one class of the v1 taxonomy.

The taxonomy and the genre -> class table live in indexer/genre_map.yaml.
`label()` is used when a song is added (pipeline) and by
`python -m indexer genre-labels`, which (re)labels the whole catalogue.
"""

from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

MAP_PATH = Path(__file__).with_name("genre_map.yaml")


@dataclass(frozen=True)
class Taxonomy:
    version: int
    names: dict[str, str]  # class -> display name
    of_genre: dict[str, str]  # MusicBrainz genre (lower case) -> class
    ignored: frozenset[str]
    weak: frozenset[str]


@dataclass(frozen=True)
class Label:
    label: str | None  # None: nothing usable in the genres
    clear: bool  # False when a tie had to be broken by MusicBrainz's order
    unknown: tuple[str, ...]  # genres missing from genre_map.yaml


def load_taxonomy(path: Path = MAP_PATH) -> Taxonomy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    of_genre: dict[str, str] = {}
    for cls, spec in raw["classes"].items():
        for g in spec["genres"]:
            g = g.lower()
            if g in of_genre:
                raise ValueError(f"genre {g!r} is in two classes: {of_genre[g]} and {cls}")
            of_genre[g] = cls
    ignored = frozenset(g.lower() for g in raw.get("ignore", []))
    if both := ignored & of_genre.keys():
        raise ValueError(f"genres both mapped and ignored: {sorted(both)}")
    return Taxonomy(
        version=int(raw["version"]),
        names={c: s["name"] for c, s in raw["classes"].items()},
        of_genre=of_genre,
        ignored=ignored,
        weak=frozenset(raw.get("weak", [])),
    )


@cache
def taxonomy() -> Taxonomy:
    return load_taxonomy()


def vote_levels(names: list[str]) -> list[int]:
    """Weight of each genre, recovered from MusicBrainz's order.

    MusicBrainz lists genres by vote count (most voted first), then
    alphabetically among equal counts; we only stored the names. So each time
    the alphabetical order goes backwards, the count has dropped: the list
    splits into runs of decreasing votes. First run -> highest weight.
    E.g. [soul | deep soul, pop, r&b | ambient, gospel...] -> 3, 2, 2, 2, 1, 1...
    A fully alphabetical list means equal votes: all weights equal.
    """
    if not names:
        return []
    run = [0]
    for prev, cur in zip(names, names[1:], strict=False):
        run.append(run[-1] + (cur < prev))
    return [run[-1] + 1 - r for r in run]


def label(genres: list[str], tags: list[str] | None = None, tax: Taxonomy | None = None) -> Label:
    """Class of a song from its MusicBrainz genres (in MusicBrainz's order);
    its tags are used the same way when it has no genre."""
    tax = tax or taxonomy()
    names = [g.lower() for g in (genres or [])] or [t.lower() for t in (tags or [])]
    # free-text tags are not expected to all be in the table: only genres are reported
    unknown = tuple(g for g in names if g not in tax.of_genre and g not in tax.ignored)
    unknown = unknown if genres else ()
    votes = [(tax.of_genre[g], w) for g, w in zip(names, vote_levels(names), strict=True)
             if g in tax.of_genre]  # fmt: skip
    strong = [(c, w) for c, w in votes if c not in tax.weak] or votes
    if not strong:
        return Label(None, True, unknown)
    score: Counter[str] = Counter()
    for c, w in strong:
        score[c] += w
    best = max(score.values())
    tied = [c for c in score if score[c] == best]
    first = next(c for c, _ in strong if c in tied)  # earliest = most voted genre
    return Label(first, len(tied) == 1, unknown)


def stats(songs: list[dict], tax: Taxonomy | None = None) -> dict:
    """Per-class counts of a catalogue (labels recomputed from the genres)."""
    tax = tax or taxonomy()
    labels = [label(s.get("mb_genres") or [], s.get("mb_tags") or [], tax) for s in songs]
    per_class = Counter(lab.label for lab in labels if lab.label)
    ties = Counter(lab.label for lab in labels if lab.label and not lab.clear)
    unknown = Counter(g for lab in labels for g in lab.unknown)
    return {
        "songs": len(songs),
        "labelled": sum(per_class.values()),
        "per_class": {c: per_class.get(c, 0) for c in tax.names},
        "ties": dict(ties),
        "unknown_genres": dict(unknown),
    }


def relabel(db, dry_run: bool = False, out=print) -> dict:
    """Recompute every song's genre_label from its genres and store the changes."""
    from earworm.remote import fetch_songs

    tax = taxonomy()
    songs = fetch_songs(db)
    changed = 0
    for s in songs:
        new = label(s["mb_genres"] or [], s["mb_tags"] or [], tax).label
        if new != s.get("genre_label"):
            changed += 1
            if not dry_run:
                db.update("songs", {"genre_label": new}, id=f"eq.{s['id']}")
    st = stats(songs, tax)
    out(f"taxonomy v{tax.version}: {st['labelled']}/{st['songs']} songs labelled "
        f"({st['labelled'] / max(1, st['songs']):.0%}), {changed} labels "
        f"{'would change' if dry_run else 'updated'}")  # fmt: skip
    for c, n in sorted(st["per_class"].items(), key=lambda kv: -kv[1]):
        out(f"  {tax.names[c]:28s} {n:4d}  (tie broken: {st['ties'].get(c, 0)})")
    if st["unknown_genres"]:
        out(f"genres missing from genre_map.yaml: {st['unknown_genres']}")
    return st
