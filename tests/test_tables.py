"""T14: HTML table extraction with rowspan."""

from indexer.tables import wikitables

HTML = """
<p>intro</p>
<table class="wikitable sortable">
  <tr><th>Date</th><th>Titre</th><th>Artiste</th></tr>
  <tr><td>5 janvier</td><td rowspan="3">« Tombe la neige »</td>
      <td rowspan="3">Adamo<sup class="reference">[1]</sup></td></tr>
  <tr><td>12 janvier</td></tr>
  <tr><td>19 janvier</td></tr>
  <tr><td>26 janvier</td><td>"She Loves You"</td><td>The Beatles<br>(UK)</td></tr>
</table>
<table class="infobox"><tr><td>not a wikitable</td></tr></table>
<table class="wikitable"><tr><th colspan="2">No.</th></tr><tr><td>1</td><td>x [a]</td></tr></table>
"""


def test_rowspan_and_cleanup():
    tables = wikitables(HTML)
    assert len(tables) == 2
    rows = tables[0]
    assert rows[0] == ["Date", "Titre", "Artiste"]
    assert rows[1] == ["5 janvier", "Tombe la neige", "Adamo"]
    assert rows[2] == ["12 janvier", "Tombe la neige", "Adamo"]
    assert rows[3] == ["19 janvier", "Tombe la neige", "Adamo"]
    assert rows[4] == ["26 janvier", "She Loves You", "The Beatles (UK)"]


def test_colspan_and_footnote_letters():
    assert wikitables(HTML)[1] == [["No.", "No."], ["1", "x"]]
