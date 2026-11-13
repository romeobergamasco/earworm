"""Extract the 'wikitable' tables of a rendered Wikipedia page, standard library only.

Cells spanning several rows (rowspan) are copied into every row they cover, so
each row comes out complete. Footnote markers like [1] are removed.
"""

import re
from html.parser import HTMLParser

_FOOTNOTE = re.compile(r"\[\s*(?:\d+|[a-z]|note \d+|N \d+)\s*\]")


def _clean(text: str) -> str:
    text = _FOOTNOTE.sub("", text)
    return re.sub(r"\s+", " ", text).strip().strip('"“”«» ').strip()


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._stack: list[dict] = []  # nested tables
        self._skip = 0  # inside <sup class="reference"> or <style>

    @property
    def _t(self):
        return self._stack[-1] if self._stack else None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("style", "script") or (tag == "sup" and "reference" in a.get("class", "")):
            self._skip += 1
            return
        if tag == "table":
            self._stack.append(
                {
                    "wiki": "wikitable" in a.get("class", ""),
                    "rows": [],
                    "row": None,
                    "cell": None,
                    "spans": {},
                }
            )
        t = self._t
        if t is None:
            return
        if tag == "tr":
            t["row"] = []
        elif tag in ("td", "th") and t["row"] is not None:
            t["cell"] = {
                "text": [],
                "rowspan": int(re.sub(r"\D", "", a.get("rowspan", "1")) or 1),
                "colspan": int(re.sub(r"\D", "", a.get("colspan", "1")) or 1),
            }
        elif tag == "br" and t["cell"] is not None:
            t["cell"]["text"].append(" ")

    def handle_endtag(self, tag):
        if tag in ("style", "script") or (tag == "sup" and self._skip):
            self._skip = max(0, self._skip - 1)
            return
        t = self._t
        if t is None:
            return
        if tag in ("td", "th") and t["cell"] is not None and t["row"] is not None:
            t["row"].append(t["cell"])
            t["cell"] = None
        elif tag == "tr" and t["row"] is not None:
            t["rows"].append(self._expand(t, t["row"]))
            t["row"] = None
        elif tag == "table":
            done = self._stack.pop()
            if done["wiki"]:
                self.tables.append(done["rows"])

    def handle_data(self, data):
        t = self._t
        if not self._skip and t is not None and t["cell"] is not None:
            t["cell"]["text"].append(data)

    @staticmethod
    def _expand(t: dict, cells: list[dict]) -> list[str]:
        """Place cells in columns, filling the columns taken by earlier rowspans."""
        spans = t["spans"]  # column -> (text, rows left)
        row: list[str] = []
        col = 0
        queue = list(cells)
        while queue or any(c >= col for c in spans):
            if col in spans:
                text, left = spans[col]
                row.append(text)
                spans[col] = (text, left - 1)
                if left - 1 == 0:
                    del spans[col]
                col += 1
                continue
            if not queue:
                break
            cell = queue.pop(0)
            text = _clean("".join(cell["text"]))
            for _ in range(cell["colspan"]):
                row.append(text)
                if cell["rowspan"] > 1:
                    spans[col] = (text, cell["rowspan"] - 1)
                col += 1
        return row


def wikitables(html: str) -> list[list[list[str]]]:
    """Every 'wikitable' as a list of rows, each a list of cell texts."""
    p = _Parser()
    p.feed(html)
    return p.tables
