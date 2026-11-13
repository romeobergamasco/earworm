"""Earworm HTTP API (docs/spec.md, section 3) and the PWA's static files.

    POST /api/identify   body: PCM 16-bit little-endian mono, X-Sample-Rate: 11025
    GET  /api/songs/{id}
    GET  /api/health

Run locally:  uvicorn server.main:app --port 7860
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from earworm.audio import SAMPLE_RATE
from earworm.fingerprint import fingerprint
from server.store import CatalogStore

MIN_SECONDS, MAX_SECONDS = 3.0, 15.0
MAX_BYTES = int(MAX_SECONDS * SAMPLE_RATE) * 2
SONG_FIELDS = ("id", "title", "artist", "year", "youtube_id", "cover_url")

log = logging.getLogger("earworm.api")


def _song_card(song: dict) -> dict:
    return {k: song.get(k) for k in SONG_FIELDS}


def _similar(song: dict, songs: dict[int, dict]) -> list[dict]:
    """The 5 most similar songs (precomputed at catalogue load, server/store.py)."""
    return [_song_card(songs[i]) for i in song.get("similar_ids", []) if i in songs]


def create_app(
    store: CatalogStore, web_dir: Path | None = None, allowed_origins: list[str] | None = None
) -> FastAPI:
    """`allowed_origins`: web pages hosted elsewhere (the Hugging Face Static Space)
    that may call this API from the browser (CORS). Same-origin needs nothing."""
    app = FastAPI(
        title="Earworm", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json"
    )
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Sample-Rate"],
            max_age=86400,
        )

    def ready_state():
        state = store.snapshot()
        if state is None:
            raise HTTPException(503, "the catalogue is still loading, try again in a moment")
        return state

    @app.post("/api/identify")
    async def identify(request: Request):
        t0 = time.perf_counter()
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_BYTES:
            raise HTTPException(413, f"at most {MAX_SECONDS:.0f} s of audio")
        body = await request.body()
        if len(body) > MAX_BYTES:
            raise HTTPException(413, f"at most {MAX_SECONDS:.0f} s of audio")
        if request.headers.get("x-sample-rate") != str(SAMPLE_RATE):
            raise HTTPException(400, f"X-Sample-Rate must be {SAMPLE_RATE}")
        if len(body) % 2:
            raise HTTPException(400, "body must be 16-bit PCM")
        seconds = len(body) / 2 / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            raise HTTPException(400, f"at least {MIN_SECONDS:.0f} s of audio")
        index, songs = ready_state()

        y = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
        t1 = time.perf_counter()
        hashes = fingerprint(y)
        t2 = time.perf_counter()
        matches = index.query(hashes)
        t3 = time.perf_counter()

        best = matches[0] if matches else None
        timing = {
            "fingerprint": round((t2 - t1) * 1000),
            "match": round((t3 - t2) * 1000),
            "total": round((time.perf_counter() - t0) * 1000),
        }
        found = best is not None and best.is_confident and best.song_id in songs
        log.info(
            json.dumps(
                {
                    "event": "identify",
                    "seconds": round(seconds, 1),
                    "match": found,
                    "song_id": best.song_id if found else None,
                    "score": best.score if best else 0,
                    "ratio": round(best.ratio, 1) if best else 0,
                    **{f"ms_{k}": v for k, v in timing.items()},
                }
            )
        )
        if not found:
            return {"match": False, "timing_ms": timing}
        return {
            "match": True,
            "song": _song_card(songs[best.song_id]),
            "confidence": {
                "score": best.score,
                "ratio": round(best.ratio, 1),
                "offset_s": round(best.offset_s, 1),
            },
            "genre": songs[best.song_id].get("genre"),
            "similar": _similar(songs[best.song_id], songs),
            "timing_ms": timing,
        }

    @app.get("/api/songs/{song_id}")
    def get_song(song_id: int):
        _, songs = ready_state()
        if song_id not in songs:
            raise HTTPException(404, "unknown song")
        song = songs[song_id]
        return {
            "song": _song_card(song),
            "genre": song.get("genre"),
            "similar": _similar(song, songs),
        }

    @app.get("/api/health")
    def health():
        state = store.snapshot()
        if state is None:
            return JSONResponse({"status": "loading"})
        index, songs = state
        return {
            "status": "ok",
            "songs": len(songs),
            "hashes": index.n_hashes,
            "index_mb": round(index.nbytes / 2**20, 1),
            "index_loaded_at": store.loaded_at.isoformat(timespec="seconds"),
            "load_seconds": round(store.load_seconds or 0, 1),
            "catalog_version": store.version,
        }

    if web_dir is not None and web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
