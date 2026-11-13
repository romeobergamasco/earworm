"""T20: genre taxonomy, labels and the training-data export."""

import json

import numpy as np
import pytest

from indexer.dataset import build
from indexer.embed import save_segments
from indexer.genres import label, load_taxonomy, stats, taxonomy, vote_levels


def test_the_yaml_is_consistent():
    tax = taxonomy()
    assert 8 <= len(tax.names) <= 12
    assert tax.weak == {"pop"} and tax.of_genre["hip hop"] == "hip-hop"


def test_a_genre_in_two_classes_is_refused(tmp_path):
    f = tmp_path / "map.yaml"
    f.write_text(
        "version: 1\nclasses:\n  a: {name: A, genres: [x]}\n  b: {name: B, genres: [x]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two classes"):
        load_taxonomy(f)


def test_vote_levels_follow_musicbrainz_order():
    # sorted by votes, then by name: each alphabetical step back = fewer votes
    assert vote_levels(["soul", "deep soul", "pop", "r&b", "ambient", "jazz"]) == [3, 2, 2, 2, 1, 1]
    assert vote_levels(["dance", "house", "latin"]) == [1, 1, 1]  # equal votes
    assert vote_levels([]) == []


def test_pop_only_counts_when_nothing_else():
    assert label(["pop", "rock"]).label == "rock"
    assert label(["pop", "ballad"]).label == "pop"


def test_most_voted_genres_win_over_many_small_ones():
    # Sean Paul - Temperature: two top-voted reggae genres, four minor electronic ones
    genres = ["dancehall", "reggae", "pop", "dance", "hip hop", "alternative rock", "club",
              "electronic", "euro house"]  # fmt: skip
    assert label(genres).label == "reggae-caribbean"


def test_ties_go_to_the_first_genre():
    lab = label(["chanson française", "pop", "rock"])
    assert lab.label == "chanson" and lab.clear is False


def test_tags_are_used_without_genres_and_unknowns_reported():
    assert label([], ["hip-hop", "french"]).label == "hip-hop"
    assert label([], ["austrian"]).label is None
    lab = label(["rock", "some new genre"])
    assert lab.label == "rock" and lab.unknown == ("some new genre",)
    assert label([], []).label is None


def test_stats():
    songs = [{"mb_genres": ["rock"]}, {"mb_genres": ["jazz", "blues"]}, {"mb_genres": []}]
    st = stats(songs)
    assert (
        st["labelled"] == 2 and st["per_class"]["rock"] == 1 and st["per_class"]["jazz-blues"] == 1
    )


def test_export_one_row_per_segment_without_audio(tmp_path):
    songs = [
        {"id": 1, "artist": "A", "title": "x", "mb_genres": ["rock"]},
        {"id": 2, "artist": "B", "title": "y", "mb_genres": []},
        {"id": 3, "artist": "C", "title": "z", "mb_genres": ["jazz"]},  # no vectors on disk
    ]
    save_segments(tmp_path, "m", 1, np.ones((3, 4)))
    save_segments(tmp_path, "m", 2, np.zeros((2, 4)))
    d = build(songs, "m", root=tmp_path, t18=None)
    assert d["X"].shape == (5, 4) and d["song_id"].tolist() == [1, 1, 1, 2, 2]
    assert d["label"].tolist() == ["rock"] * 3 + [""] * 2 and d["artist"][0] == "A"
    meta = json.loads(str(d["meta"]))
    assert meta["missing_segments"] == [3] and meta["taxonomy_version"] == 1
    assert set(d) == {"X", "song_id", "label", "artist", "meta"}
