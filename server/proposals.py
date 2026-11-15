"""Song proposals from visitors (docs/spec.md, sections 2.2 and 3; T24).

A visitor pastes a YouTube link; it goes into the `proposals` table, which the
maintainer reviews locally (`python -m indexer review`, T25). Nothing is
indexed from the server.

No account and no personal data: a visitor is identified only by
HMAC-SHA256(salt, IP), which is enough to enforce the rate limit. The IP itself
is never stored nor logged. The salt is a server secret (PROPOSALS_SALT), so
the hash cannot be reversed by trying every IPv4 address.
"""

import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from earworm.youtube import parse_youtube_id

PER_HOUR, PER_DAY = 5, 20
MAX_HINT = 200  # same limit as the check constraint in db/001_init.sql


class ProposalError(Exception):
    """Refused proposal: `status` is the HTTP code (409, 422 or 429)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class Duplicate(Exception):
    """Raised by a store when the unique index on pending proposals refuses the row."""


class ProposalStore(Protocol):
    def in_catalogue(self, youtube_id: str) -> bool: ...
    def is_pending(self, youtube_id: str) -> bool: ...
    def recent(self, client_hash: str, since: datetime) -> list[datetime]: ...
    def add(self, row: dict) -> None: ...


@dataclass(frozen=True)
class Proposal:
    youtube_id: str
    title_hint: str | None
    artist_hint: str | None


def client_hash(ip: str, salt: str) -> str:
    return hmac.new(salt.encode(), ip.encode(), hashlib.sha256).hexdigest()


def _hint(value, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProposalError(422, f"{name} must be text")
    text = " ".join(value.split())  # also drops newlines and tabs
    if len(text) > MAX_HINT:
        raise ProposalError(422, f"{name}: at most {MAX_HINT} characters")
    return text or None


def parse(body) -> Proposal:
    if not isinstance(body, dict) or not isinstance(body.get("url"), str):
        raise ProposalError(422, 'expected {"url": "https://www.youtube.com/watch?v=..."}')
    try:
        youtube_id = parse_youtube_id(body["url"])
    except ValueError:
        raise ProposalError(422, "not a YouTube video link") from None
    title, artist = _hint(body.get("title"), "title"), _hint(body.get("artist"), "artist")
    return Proposal(youtube_id, title, artist)


def submit(store: ProposalStore, body, client: str, now: datetime | None = None) -> Proposal:
    """Validate, rate-limit and store one proposal; raise ProposalError if refused.

    Order: an invalid link is refused first (422), then the rate limit (429) is
    applied before anything about the catalogue is revealed, then duplicates (409).
    """
    proposal = parse(body)
    now = now or datetime.now(UTC)
    times = store.recent(client, now - timedelta(days=1))
    if len(times) >= PER_DAY or sum(t > now - timedelta(hours=1) for t in times) >= PER_HOUR:
        raise ProposalError(429, "too many proposals, try again later")
    if store.in_catalogue(proposal.youtube_id):
        raise ProposalError(409, "this song is already in the catalogue")
    if store.is_pending(proposal.youtube_id):
        raise ProposalError(409, "this song has already been proposed")
    try:
        store.add(
            {
                "youtube_id": proposal.youtube_id,
                "title_hint": proposal.title_hint,
                "artist_hint": proposal.artist_hint,
                "client_hash": client,
            }
        )
    except Duplicate:  # proposed by someone else in the meantime
        raise ProposalError(409, "this song has already been proposed") from None
    return proposal


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class SupabaseProposals:
    """ProposalStore over the Supabase REST client (earworm/db.py)."""

    def __init__(self, db) -> None:
        self.db = db

    def in_catalogue(self, youtube_id: str) -> bool:
        return bool(self.db.select("songs", "id", youtube_id=f"eq.{youtube_id}", limit=1))

    def is_pending(self, youtube_id: str) -> bool:
        rows = self.db.select(
            "proposals", "id", youtube_id=f"eq.{youtube_id}", status="eq.pending", limit=1
        )
        return bool(rows)

    def recent(self, client_hash: str, since: datetime) -> list[datetime]:
        rows = self.db.select(
            "proposals",
            "created_at",
            client_hash=f"eq.{client_hash}",
            created_at=f"gte.{since.isoformat()}",
            limit=PER_DAY + 1,
        )
        return [_timestamp(r["created_at"]) for r in rows]

    def add(self, row: dict) -> None:
        from earworm.db import SupabaseError

        try:
            self.db.insert("proposals", row)
        except SupabaseError as e:
            if e.status == 409:  # PostgREST: unique violation (proposals_one_pending)
                raise Duplicate from None
            raise


class MemoryProposals:
    """In-memory ProposalStore, for tests and the local demo."""

    def __init__(self, catalogue: set[str] | None = None) -> None:
        self.catalogue = set(catalogue or ())
        self.rows: list[dict] = []
        self.clock = lambda: datetime.now(UTC)

    def in_catalogue(self, youtube_id: str) -> bool:
        return youtube_id in self.catalogue

    def is_pending(self, youtube_id: str) -> bool:
        return any(r["youtube_id"] == youtube_id and r["status"] == "pending" for r in self.rows)

    def recent(self, client_hash: str, since: datetime) -> list[datetime]:
        return [
            r["created_at"]
            for r in self.rows
            if r["client_hash"] == client_hash and r["created_at"] >= since
        ]

    def add(self, row: dict) -> None:
        if MemoryProposals.is_pending(self, row["youtube_id"]):  # like the unique index
            raise Duplicate
        self.rows.append({**row, "status": "pending", "created_at": self.clock()})
