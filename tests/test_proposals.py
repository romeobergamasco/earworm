"""T24: POST /api/proposals (spec sections 2.2 and 3), no network."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from earworm.db import SupabaseError
from server.app import create_app
from server.proposals import (
    PER_DAY,
    PER_HOUR,
    Duplicate,
    MemoryProposals,
    ProposalError,
    SupabaseProposals,
    client_hash,
    submit,
)
from server.store import CatalogStore
from tests.test_server import loader

SALT = "test-salt"
IN_CATALOGUE = "AAAAAAAAAAA"
IP_HEADER = "CF-Connecting-IP"


def link(i: int) -> str:
    return f"https://www.youtube.com/watch?v=vid{i:08d}"


@pytest.fixture
def box():
    return MemoryProposals(catalogue={IN_CATALOGUE})


@pytest.fixture
def client(box):
    store = CatalogStore(loader, version_fn=lambda: 3)
    store.load()
    return TestClient(create_app(store, proposals=box, salt=SALT, ip_header=IP_HEADER))


def propose(client, body, ip="203.0.113.7", **headers):
    return client.post("/api/proposals", json=body, headers={IP_HEADER: ip, **headers})


def test_201_stores_the_proposal(client, box):
    r = propose(
        client, {"url": "youtu.be/dQw4w9WgXcQ?t=42", "title": "  Never  Gonna\n", "artist": ""}
    )
    assert r.status_code == 201
    assert r.json() == {"status": "pending", "youtube_id": "dQw4w9WgXcQ"}
    (row,) = box.rows
    assert row["youtube_id"] == "dQw4w9WgXcQ"
    assert row["title_hint"] == "Never Gonna" and row["artist_hint"] is None
    assert row["status"] == "pending"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=x",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/shorts/dQw4w9WgXcQ",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
    ],
)
def test_accepted_link_forms(client, url):
    assert propose(client, {"url": url}).status_code == 201


def test_409_already_in_catalogue(client, box):
    r = propose(client, {"url": f"https://youtu.be/{IN_CATALOGUE}"})
    assert r.status_code == 409 and "catalogue" in r.json()["detail"]
    assert box.rows == []


def test_409_already_pending(client):
    assert propose(client, {"url": link(1)}).status_code == 201
    r = propose(client, {"url": link(1)}, ip="198.51.100.1")  # someone else, same video
    assert r.status_code == 409 and "proposed" in r.json()["detail"]


def test_409_when_the_unique_index_wins_the_race(box):
    box.is_pending = lambda youtube_id: False  # check passed, then someone else inserted
    submit(box, {"url": link(1)}, "a")
    with pytest.raises(ProposalError) as e:
        submit(box, {"url": link(1)}, "b")
    assert e.value.status == 409


def test_a_rejected_video_can_be_proposed_again(client, box):
    assert propose(client, {"url": link(1)}).status_code == 201
    box.rows[0]["status"] = "rejected"
    assert propose(client, {"url": link(1)}, ip="198.51.100.1").status_code == 201


@pytest.mark.parametrize(
    "body",
    [
        {"url": "https://vimeo.com/123456"},
        {"url": "https://www.youtube.com/watch?v=short"},
        {"url": "javascript:alert(1)"},
        {"url": ""},
        {"url": 42},
        {"title": "no url"},
        ["https://youtu.be/dQw4w9WgXcQ"],
        {"url": "https://youtu.be/dQw4w9WgXcQ", "title": "x" * 201},
        {"url": "https://youtu.be/dQw4w9WgXcQ", "artist": ["not", "text"]},
    ],
)
def test_422_invalid(client, box, body):
    assert propose(client, body).status_code == 422
    assert box.rows == []


def test_422_not_json(client):
    r = client.post(
        "/api/proposals", content=b"{oops", headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 422


def test_429_sixth_proposal_in_the_hour(client, box):
    for i in range(PER_HOUR):
        assert propose(client, {"url": link(i)}).status_code == 201
    r = propose(client, {"url": link(99)})
    assert r.status_code == 429
    assert len(box.rows) == PER_HOUR
    assert (
        propose(client, {"url": link(99)}, ip="198.51.100.1").status_code == 201
    )  # others are not


def test_429_comes_before_revealing_the_catalogue(client):
    for i in range(PER_HOUR):
        propose(client, {"url": link(i)})
    assert propose(client, {"url": f"https://youtu.be/{IN_CATALOGUE}"}).status_code == 429


def test_rate_limit_windows(box):
    t0 = datetime(2026, 11, 15, 8, 0, tzinfo=UTC)
    n = 0
    for hour in range(4):  # 5 per hour, 4 hours: 20 in the day
        box.clock = lambda h=hour: t0 + timedelta(hours=h)
        for _ in range(PER_HOUR):
            submit(box, {"url": link(n)}, "c", now=box.clock())
            n += 1
    assert n == PER_DAY
    with pytest.raises(ProposalError) as e:  # a new hour, but the day is full
        submit(box, {"url": link(n)}, "c", now=t0 + timedelta(hours=5))
    assert e.value.status == 429
    submit(
        box, {"url": link(n)}, "c", now=t0 + timedelta(hours=24, minutes=1)
    )  # first ones expired


def test_x_forwarded_for_cannot_dodge_the_limit(client):
    for i in range(PER_HOUR):
        propose(client, {"url": link(i)}, **{"X-Forwarded-For": f"10.0.0.{i}"})
    r = propose(client, {"url": link(99)}, **{"X-Forwarded-For": "10.0.0.200"})
    assert r.status_code == 429


def test_the_ip_is_never_stored(client, box):
    ip = "203.0.113.7"
    propose(client, {"url": link(1)}, ip=ip)
    (row,) = box.rows
    assert all(ip not in str(v) for v in row.values())
    assert row["client_hash"] == client_hash(ip, SALT) and len(row["client_hash"]) == 64
    assert client_hash(ip, "other-salt") != row["client_hash"]


def test_without_proxy_header_the_peer_address_is_used(box):
    store = CatalogStore(loader, version_fn=lambda: 3)
    store.load()
    c = TestClient(create_app(store, proposals=box, salt=SALT, ip_header=None))
    c.post("/api/proposals", json={"url": link(1)})
    assert box.rows[0]["client_hash"] == client_hash("testclient", SALT)


def test_503_when_not_configured():
    store = CatalogStore(loader, version_fn=lambda: 3)
    store.load()
    c = TestClient(create_app(store))
    assert c.post("/api/proposals", json={"url": link(1)}).status_code == 503


class FakeRest:
    """Records the PostgREST calls of SupabaseProposals."""

    def __init__(self, select_result=(), insert_status=None):
        self.calls = []
        self.select_result = list(select_result)
        self.insert_status = insert_status

    def select(self, table, columns="*", **filters):
        self.calls.append(("select", table, columns, filters))
        return self.select_result

    def insert(self, table, row):
        self.calls.append(("insert", table, row))
        if self.insert_status:
            raise SupabaseError(self.insert_status, "duplicate key", "insert proposals")
        return [row]


def test_supabase_store_queries():
    db = FakeRest(select_result=[{"created_at": "2026-11-15T08:00:00.123+00:00"}])
    s = SupabaseProposals(db)
    since = datetime(2026, 11, 14, 8, 0, tzinfo=UTC)
    assert s.recent("h", since) == [datetime(2026, 11, 15, 8, 0, 0, 123000, tzinfo=UTC)]
    assert db.calls[-1][3] == {
        "client_hash": "eq.h",
        "created_at": "gte.2026-11-14T08:00:00+00:00",
        "limit": PER_DAY + 1,
    }
    assert s.in_catalogue("dQw4w9WgXcQ")
    assert db.calls[-1][1:3] == ("songs", "id")
    assert s.is_pending("dQw4w9WgXcQ")
    assert db.calls[-1][3]["status"] == "eq.pending"


def test_supabase_store_unique_violation_is_a_duplicate():
    with pytest.raises(Duplicate):
        SupabaseProposals(FakeRest(insert_status=409)).add({"youtube_id": "x"})
    with pytest.raises(SupabaseError):
        SupabaseProposals(FakeRest(insert_status=500)).add({"youtube_id": "x"})
