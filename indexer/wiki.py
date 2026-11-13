"""Polite, cached access to Wikipedia pages (rendered HTML).

Wikimedia rate limits (https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits):
a client without contact details in its User-Agent is "unidentified" and gets
10 requests per minute. We stay under that (one request every 6.5 s), honour
Retry-After, and cache every page on disk so it is never fetched twice.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from indexer.musicbrainz import RateLimiter, user_agent

CACHE = Path("data") / "cache" / "wiki"
_limiter = RateLimiter(interval=6.5)


def _cache_path(lang: str, title: str) -> Path:
    safe = re.sub(r"[^\w\-]+", "_", title)
    return CACHE / lang / f"{safe}.json"


def fetch_html(lang: str, title: str, retries: int = 5) -> tuple[str, str] | None:
    """(resolved title, HTML) of a page, following redirects; None if missing."""
    path = _cache_path(lang, title)
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return None if cached.get("missing") else (cached["title"], cached["html"])

    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    url = f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    for attempt in range(retries + 1):
        _limiter.wait()
        req = urllib.request.Request(url, headers={"User-Agent": user_agent() + " python-urllib"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries:
                wait = float(e.headers.get("Retry-After") or 5) * (attempt + 1)
                time.sleep(max(wait, 5))
                continue
            raise
    path.parent.mkdir(parents=True, exist_ok=True)
    if "error" in data:  # missingtitle
        path.write_text(json.dumps({"missing": True}), encoding="utf-8")
        return None
    page = data["parse"]
    path.write_text(json.dumps({"title": page["title"], "html": page["text"]}), encoding="utf-8")
    return page["title"], page["text"]


def billboard_title(year: int) -> str:
    # redirects to the "top 30" / "top 50" pages for the 1950s
    return f"Billboard Year-End Hot 100 singles of {year}"


def france_title(year: int) -> str:
    return f"Liste des titres musicaux numéro un en France en {year}"


if __name__ == "__main__":  # warm the cache: python -m indexer.wiki 1950 2025
    import sys

    first, last = int(sys.argv[1]), int(sys.argv[2])
    for year in range(first, last + 1):
        for lang, title in (("en", billboard_title(year)), ("fr", france_title(year))):
            t0 = time.perf_counter()
            res = fetch_html(lang, title)
            status = "missing" if res is None else f"{len(res[1]) // 1000} kB ({res[0]})"
            print(f"{year} {lang}: {status}  [{time.perf_counter() - t0:.1f} s]", flush=True)
