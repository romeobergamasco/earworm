"""Results of measurement sessions, sent by the PWA in eval mode (spec 7.1-7.2; T26).

The local script opens a session (a row in eval_runs with a random id) and
prints the PWA link with ?run_id=<id>. In that mode, after each identification,
the PWA posts what it showed and the latency it measured (end of capture ->
result displayed) to POST /api/eval/{run_id}. Only an open session younger
than two days is accepted: the id acts as a password, and nothing is written
otherwise. No audio, nothing about the visitor.
"""

import re
from datetime import UTC, datetime, timedelta
from typing import Protocol

RUN_ID = re.compile(r"[A-Za-z0-9_-]{8,40}")
MAX_AGE = timedelta(days=2)
TIMING_KEYS = ("fingerprint", "match", "total")


class EvalError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class EvalStore(Protocol):
    def run(self, run_id: str) -> dict | None: ...  # {"created_at": datetime, "closed_at": ...}
    def add(self, row: dict) -> None: ...


def _int(body: dict, key: str, lo: int, hi: int, required: bool = False) -> int | None:
    v = body.get(key)
    if v is None and not required:
        return None
    if isinstance(v, bool) or not isinstance(v, int | float) or not lo <= v <= hi:
        raise EvalError(422, f"{key}: a number between {lo} and {hi}")
    return round(v)


def parse(body) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("match"), bool):
        raise EvalError(422, 'expected {"match": true|false, "latency_ms": ..., ...}')
    row = {
        "match": body["match"],
        "song_id": _int(body, "song_id", 1, 2**53) if body["match"] else None,
        "score": _int(body, "score", 0, 10**7),
        "latency_ms": _int(body, "latency_ms", 0, 600_000, required=True),
        "resample_ms": _int(body, "resample_ms", 0, 600_000),
        "network_ms": _int(body, "network_ms", 0, 600_000),
        "capture_rate": _int(body, "capture_rate", 8000, 384_000),
    }
    if body["match"] and row["song_id"] is None:
        raise EvalError(422, "song_id is required when match is true")
    ratio = body.get("ratio")
    if ratio is not None and (isinstance(ratio, bool) or not isinstance(ratio, int | float)
                              or not 0 <= ratio <= 1e6):  # fmt: skip
        raise EvalError(422, "ratio: a positive number")
    row["ratio"] = ratio
    server = body.get("server")
    if server is not None:
        if not isinstance(server, dict):
            raise EvalError(422, "server: the timing_ms object of /api/identify")
        row["server_ms"] = {k: _int(server, k, 0, 600_000) for k in TIMING_KEYS if k in server}
    else:
        row["server_ms"] = None
    return row


def record(store: EvalStore, run_id: str, body, now: datetime | None = None) -> dict:
    if not RUN_ID.fullmatch(run_id):
        raise EvalError(404, "unknown session")
    run = store.run(run_id)
    now = now or datetime.now(UTC)
    if run is None or run.get("closed_at") or run["created_at"] < now - MAX_AGE:
        raise EvalError(404, "unknown or closed session")
    row = {"run_id": run_id, **parse(body)}
    store.add(row)
    return row


def _timestamp(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class SupabaseEvals:
    def __init__(self, db) -> None:
        self.db = db

    def run(self, run_id: str) -> dict | None:
        rows = self.db.select("eval_runs", "created_at,closed_at", id=f"eq.{run_id}", limit=1)
        if not rows:
            return None
        return {"created_at": _timestamp(rows[0]["created_at"]), "closed_at": rows[0]["closed_at"]}

    def add(self, row: dict) -> None:
        self.db.insert("eval_results", row)


class MemoryEvals:
    """For tests and the local demo."""

    def __init__(self) -> None:
        self.runs: dict[str, dict] = {}
        self.rows: list[dict] = []

    def open(self, run_id: str, created_at: datetime | None = None) -> None:
        self.runs[run_id] = {"created_at": created_at or datetime.now(UTC), "closed_at": None}

    def run(self, run_id: str) -> dict | None:
        return self.runs.get(run_id)

    def add(self, row: dict) -> None:
        self.rows.append(row)
