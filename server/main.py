"""Production entry point: catalogue from Supabase, PWA from web/.

    uvicorn server.main:app --host 0.0.0.0 --port 7860

Settings: SUPABASE_URL, SUPABASE_SECRET_KEY (Hugging Face secrets, or .env locally),
RELOAD_SECONDS (how often to check for a new catalogue version, default 600),
MEMORY_BUDGET_MB (default 450: above it, a reload frees the old index first),
ALLOWED_ORIGINS (pages hosted elsewhere allowed to call the API, comma-separated).
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from earworm.config import load_env, require
from earworm.db import Supabase
from server.app import create_app
from server.store import CatalogStore, supabase_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

load_env()
_cfg = require("SUPABASE_URL", "SUPABASE_SECRET_KEY")
_loader, _version = supabase_loader(Supabase(_cfg["SUPABASE_URL"], _cfg["SUPABASE_SECRET_KEY"]))
store = CatalogStore(_loader, _version, budget_mb=float(os.environ.get("MEMORY_BUDGET_MB", 450)))
WEB = Path(__file__).resolve().parents[1] / "web"


@asynccontextmanager
async def _lifespan(_app):
    store.start(every_s=float(os.environ.get("RELOAD_SECONDS", 600)))  # loads in the background
    yield


# e.g. "https://someone-earworm.static.hf.space" (comma-separated); empty = same origin only
ORIGINS = [
    o.strip().rstrip("/") for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()
]

app = create_app(store, web_dir=WEB, allowed_origins=ORIGINS)
app.router.lifespan_context = _lifespan
