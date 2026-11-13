"""T09: YouTube link validation (shared by the indexer and, later, the server)."""

import pytest

from earworm.youtube import parse_youtube_id

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtube.com/watch?v={VID}&list=PL123&index=4",
        f"https://m.youtube.com/watch?feature=share&v={VID}",
        f"http://www.youtube.com/watch?v={VID}#t=30",
        f"https://youtu.be/{VID}",
        f"https://youtu.be/{VID}?si=abcdef&t=12",
        f"https://music.youtube.com/watch?v={VID}&feature=share",
        f"https://www.youtube.com/shorts/{VID}",
        f"  https://www.youtube.com/watch?v={VID}  ",
        f"www.youtube.com/watch?v={VID}",
    ],
)
def test_accepted_links(url):
    assert parse_youtube_id(url) == VID


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://www.youtube.com/watch?v=short",
        f"https://www.youtube.com/watch?v={VID}x",  # 12 characters
        f"https://evil.example.com/watch?v={VID}",
        f"https://youtube.com.evil.example/watch?v={VID}",
        "https://www.youtube.com/playlist?list=PL123",
        "https://www.youtube.com/@somechannel",
        f"javascript:alert(1)//youtu.be/{VID}",
        f"https://youtu.be/{VID[:-1]}!",
    ],
)
def test_rejected_links(url):
    with pytest.raises(ValueError):
        parse_youtube_id(url)
