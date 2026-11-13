"""T10: turning a YouTube title + channel into an (artist, track) guess."""

import pytest

from indexer.titles import clean_channel, guess_artist_track, normalize


@pytest.mark.parametrize(
    "title, channel, artist, track",
    [
        # the user's own files
        ("Luis Fonsi - Despacito ft. Daddy Yankee", "Luis Fonsi", "Luis Fonsi", "Despacito"),
        (
            "Queen – Bohemian Rhapsody (Official Video Remastered)",
            "Queen Official",
            "Queen",
            "Bohemian Rhapsody",
        ),
        (
            "Michael Jackson - Billie Jean (Official Video)",
            "Michael Jackson",
            "Michael Jackson",
            "Billie Jean",
        ),
        (
            "Aretha Franklin Respect Official Lyric Video",
            "Aretha Franklin",
            "Aretha Franklin",
            "Respect",
        ),
        (
            "Metallica: Enter Sandman (Official Music Video)",
            "Metallica",
            "Metallica",
            "Enter Sandman",
        ),
        ("Dr. Dre - Still D.R.E. ft. Snoop Dogg", "DrDreVEVO", "Dr. Dre", "Still D.R.E."),
        (
            "The Weeknd - Blinding Lights (Official Audio)",
            "TheWeekndVEVO",
            "The Weeknd",
            "Blinding Lights",
        ),
        (
            "Bob Marley & The Wailers - Three Little Birds (Official Music Video)",
            "Bob Marley",
            "Bob Marley & The Wailers",
            "Three Little Birds",
        ),
        # other common shapes
        ("Take Five", "The Dave Brubeck Quartet - Topic", "The Dave Brubeck Quartet", "Take Five"),
        ("Stromae - Alors on danse (Clip Officiel)", "Stromae", "Stromae", "Alors on danse"),
        (
            "Daft Punk - Get Lucky (Official Audio) ft. Pharrell Williams, Nile Rodgers",
            "Daft Punk",
            "Daft Punk",
            "Get Lucky",
        ),
        ("Angèle - Balance ton quoi [CLIP OFFICIEL]", "Angèle", "Angèle", "Balance ton quoi"),
        (
            "Nirvana - Smells Like Teen Spirit (Official Music Video) [4K]",
            "Nirvana",
            "Nirvana",
            "Smells Like Teen Spirit",
        ),
        ("Eminem - Lose Yourself [HD]", "EminemMusic", "Eminem", "Lose Yourself"),
        ("Adele - Hello (Official Lyric Video)", "Adele", "Adele", "Hello"),
        ("PNL - Au DD [Clip Officiel]", "PNL", "PNL", "Au DD"),
        (
            "Mark Ronson - Uptown Funk (Official Video) ft. Bruno Mars",
            "Mark Ronson",
            "Mark Ronson",
            "Uptown Funk",
        ),
        ("BTS (방탄소년단) 'Dynamite' Official MV", "HYBE LABELS", "BTS", "Dynamite"),
        (
            "Rick Astley - Never Gonna Give You Up (Official Music Video)",
            "Rick Astley",
            "Rick Astley",
            "Never Gonna Give You Up",
        ),
        ("Gorillaz - Feel Good Inc. (Official Video)", "Gorillaz", "Gorillaz", "Feel Good Inc."),
    ],
)
def test_guess_artist_track(title, channel, artist, track):
    assert guess_artist_track(title, channel) == (artist, track)


@pytest.mark.parametrize(
    "channel, expected",
    [
        ("Queen Official", "Queen"),
        ("DrDreVEVO", "DrDre"),
        ("The Dave Brubeck Quartet - Topic", "The Dave Brubeck Quartet"),
        ("EminemMusic", "Eminem"),
        ("Stromae", "Stromae"),
        ("Rammstein Official", "Rammstein"),
    ],
)
def test_clean_channel(channel, expected):
    assert clean_channel(channel) == expected


def test_normalize_for_comparison():
    assert normalize("Beyoncé – Crazy in Love!") == "beyonce crazy in love"
    assert normalize("Still D.R.E.") == normalize("Still DRE")
    assert normalize("  Alors   on danse ") == "alors on danse"
