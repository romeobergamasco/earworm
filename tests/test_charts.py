"""T14: chart parsing, quotas, deduplication and YouTube result scoring (offline)."""

from indexer.charts import ChartEntry, build, parse_billboard, parse_france, quotas
from indexer.resolve import resolve, score_result

BILLBOARD = (
    """<table class="wikitable"><tr><th>No.</th><th>Title</th><th>Artist(s)</th></tr>
<tr><td>1</td><td>"Bette Davis Eyes"</td><td>Kim Carnes</td></tr>
<tr><td>2</td><td>"Endless Love"</td><td>Diana Ross and Lionel Richie</td></tr>
"""
    + "".join(f'<tr><td>{i}</td><td>"Song {i}"</td><td>Band {i}</td></tr>' for i in range(3, 13))
    + "</table>"
)

FRANCE = """<table class="wikitable"><tr><th>Semaine</th><th>Album</th><th>Artiste</th></tr>
<tr><td>1</td><td>Some Album</td><td>X</td></tr></table>
<table class="wikitable"><tr><th>Date</th><th>Titre</th><th>Artiste</th></tr>
<tr><td>1</td><td>« Ella, elle l'a »</td><td>France Gall</td></tr>
<tr><td>2</td><td rowspan="3">« Voyage, voyage »</td><td rowspan="3">Desireless</td></tr>
<tr><td>3</td></tr><tr><td>4</td></tr>
<tr><td>5</td><td>« Ella, elle l'a »</td><td>France Gall</td></tr>
</table>"""


def test_parse_billboard_in_rank_order():
    rows = parse_billboard(BILLBOARD)
    assert rows[:2] == [
        ("Kim Carnes", "Bette Davis Eyes"),
        ("Diana Ross and Lionel Richie", "Endless Love"),
    ]


def test_parse_france_ranks_by_weeks_and_skips_albums():
    assert parse_france(FRANCE) == [
        ("Desireless", "Voyage, voyage", 3),
        ("France Gall", "Ella, elle l'a", 2),
    ]


def test_quotas_by_decade():
    assert quotas(1955) == (6, 3) and quotas(1987) == (12, 6) and quotas(2024) == (10, 5)


def test_build_dedups_and_orders_number_ones_first():
    pages = {
        ("en", 1981): BILLBOARD,
        ("fr", 1987): FRANCE,
        ("en", 1987): BILLBOARD.replace("Kim Carnes", "KIM CARNES"),  # same song charting again
    }

    def fetch(lang, title):
        year = int(title[-4:])
        html = pages.get((lang, year))
        return (title, html) if html else None

    entries = build(fetch)
    keys = [(e.chart, e.year, e.rank, e.title) for e in entries]
    assert keys[:3] == [
        ("us", 1981, 1, "Bette Davis Eyes"),
        ("fr", 1987, 1, "Voyage, voyage"),
        ("us", 1981, 2, "Endless Love"),
    ]
    assert sum(e.title == "Bette Davis Eyes" for e in entries) == 1
    assert len([e for e in entries if e.year == 1981]) == 12  # 1980s US quota


ENTRY = ChartEntry("us", 1981, 1, "Kim Carnes", "Bette Davis Eyes")


def test_score_prefers_official_studio_version():
    official = score_result(
        ENTRY, "Kim Carnes - Bette Davis Eyes (Official Video)", "KimCarnesVEVO", 230
    )
    topic = score_result(ENTRY, "Bette Davis Eyes", "Kim Carnes - Topic", 229)
    live = score_result(ENTRY, "Kim Carnes - Bette Davis Eyes (Live 1982)", "Kim Carnes", 250)
    cover = score_result(ENTRY, "Bette Davis Eyes (Cover)", "Some Singer", 230)
    other = score_result(ENTRY, "Kim Carnes - Voyeur", "Kim Carnes", 230)
    long_mix = score_result(ENTRY, "Kim Carnes - Bette Davis Eyes 1 hour", "Kim Carnes", 3600)
    assert official >= 0.95 and topic >= 0.95
    assert live < 0.6 and cover < 0.6 and other == 0 and long_mix == 0


def test_resolve_picks_best_and_gives_up_when_nothing_fits():
    results = [
        {
            "id": "LIVEaaaaaaa",
            "title": "Bette Davis Eyes LIVE",
            "channel": "Kim Carnes",
            "duration": 240,
        },
        {
            "id": "GOODaaaaaaa",
            "title": "Kim Carnes - Bette Davis Eyes",
            "channel": "Kim Carnes",
            "duration": 230,
        },
    ]
    assert resolve(ENTRY, search=lambda q: results).youtube_id == "GOODaaaaaaa"
    assert resolve(ENTRY, search=lambda q: results[:1]) is None


def test_import_resumes_and_stops_at_target(tmp_path):
    from indexer.importer import run_import

    class Catalog:
        def __init__(self):
            self.songs = []

        def by_youtube_id(self, vid):
            return next((s for s in self.songs if s["youtube_id"] == vid), None)

    class Result:
        def __init__(self, status, song, detail=""):
            self.status, self.song, self.detail = status, song, detail

    calls = []

    def fake_add(url, catalog, **kw):
        vid = url[-11:]
        calls.append((vid, kw["hint"], kw["source"]))
        if vid == "DUPdupDUPdu":
            return Result("duplicate_audio", {"id": 1})
        if vid == "ERRerrERRer":
            raise RuntimeError("network down")
        song = {"id": len(catalog.songs) + 1, "youtube_id": vid}
        catalog.songs.append(song)
        return Result("added", song)

    entries = [ChartEntry("us", 1980 + i, 1, f"A{i}", f"T{i}") for i in range(5)]
    vids = ["AAAAAAAAAAA", "DUPdupDUPdu", "ERRerrERRer", "BBBBBBBBBBB", "CCCCCCCCCCC"]
    resolved = {("us", 1980 + i, 1): {"youtube_id": v} for i, v in enumerate(vids)}
    log = tmp_path / "log.csv"

    cat = Catalog()
    counts = run_import(
        cat, entries, resolved, target=2, log_path=log, add=fake_add, out=lambda *_: None
    )
    assert counts == {"added": 2, "duplicate_audio": 1, "error": 1}
    assert calls[0] == ("AAAAAAAAAAA", ("A0", "T0"), "chart")

    calls.clear()  # second run: only the error is retried, then the next new song
    run_import(cat, entries, resolved, target=3, log_path=log, add=fake_add, out=lambda *_: None)
    assert [c[0] for c in calls] == ["ERRerrERRer", "CCCCCCCCCCC"]


def test_france_two_level_header_and_album_section():
    html = """<h2>Classement des singles</h2>
    <table class="wikitable">
    <tr><th>№</th><th colspan="2">Ventes</th><th colspan="2">Streaming</th></tr>
    <tr><th>№</th><th>Artiste</th><th>Titre</th><th>Artiste</th><th>Titre</th></tr>
    <tr><td>1</td><td>Kendji Girac</td><td>Andalouse</td><td>X</td><td>Y</td></tr>
    <tr><td>2</td><td>Kendji Girac</td><td>Andalouse</td><td>X</td><td>Y</td></tr></table>
    <h2>Classement des albums</h2>
    <table class="wikitable"><tr><th>№</th><th>Artiste</th><th>Titre</th></tr>
    <tr><td>1</td><td>Renaud</td><td>Renaud</td></tr></table>"""
    assert parse_france(html) == [("Kendji Girac", "Andalouse", 2)]


def test_double_a_side_keeps_first_title():
    html = BILLBOARD.replace('"Bette Davis Eyes"', '"Candle in the Wind 1997" / "Something About"')
    assert parse_billboard(html)[0] == ("Kim Carnes", "Candle in the Wind 1997")


def test_rerecordings_are_not_the_original():
    e = ChartEntry("us", 1965, 1, "Sam the Sham and the Pharaohs", "Wooly Bully")
    assert score_result(e, "Wooly Bully (Rerecorded)", "Sam The Sham - Topic", 140) < 0.6
    assert score_result(e, "Wooly Bully", "Sam The Sham & The Pharaohs - Topic", 140) >= 0.95
