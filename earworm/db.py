"""Minimal Supabase client over its REST API (PostgREST), standard library only.

Used by the local indexer and the server. Keys never appear in errors or logs.
"""

import json
import urllib.error
import urllib.parse
import urllib.request


class SupabaseError(RuntimeError):
    def __init__(self, status: int, body: str, what: str) -> None:
        super().__init__(f"Supabase {what}: HTTP {status} {body[:300]}")
        self.status = status


class Supabase:
    def __init__(self, url: str, key: str, timeout: float = 60) -> None:
        root = url.strip().rstrip("/")
        if root.endswith("/rest/v1"):  # the "Data API" URL already includes it
            root = root[: -len("/rest/v1")]
        self.base = root + "/rest/v1"
        self.headers = {"apikey": key, "Content-Type": "application/json"}
        if key.startswith("eyJ"):  # legacy JWT keys also go in Authorization
            self.headers["Authorization"] = f"Bearer {key}"
        self.timeout = timeout

    def _request(self, method: str, path: str, params=None, body=None, prefer=None):
        url = f"{self.base}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = dict(self.headers)
        if prefer:
            headers["Prefer"] = prefer
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return resp.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")

    def _ok(self, what, status, payload):
        if not 200 <= status < 300:
            raise SupabaseError(status, str(payload), what)
        return payload

    def select(self, table: str, columns: str = "*", **filters) -> list[dict]:
        params = {"select": columns, **filters}
        return self._ok(f"select {table}", *self._request("GET", table, params))

    def select_all(
        self, table: str, columns: str = "*", order: str = "id", page: int = 1000, **filters
    ) -> list[dict]:
        """Every matching row, page by page (Supabase returns at most 1000 rows per request)."""
        rows: list[dict] = []
        while True:
            batch = self.select(
                table, columns, order=order, limit=str(page), offset=str(len(rows)), **filters
            )
            rows += batch
            if len(batch) < page:
                return rows

    def insert(self, table: str, rows: list[dict] | dict) -> list[dict]:
        return self._ok(
            f"insert {table}",
            *self._request("POST", table, body=rows, prefer="return=representation"),
        )

    def update(self, table: str, values: dict, **filters) -> list[dict]:
        return self._ok(
            f"update {table}",
            *self._request("PATCH", table, filters, values, prefer="return=representation"),
        )

    def delete(self, table: str, **filters) -> list[dict]:
        if not filters:
            raise ValueError("refusing to delete without a filter")
        return self._ok(
            f"delete {table}",
            *self._request("DELETE", table, filters, prefer="return=representation"),
        )

    def rpc(self, function: str, args: dict | None = None):
        status, payload = self._request("POST", f"rpc/{function}", body=args or {})
        return self._ok(f"rpc {function}", status, payload)

    def raw(self, method: str, path: str, params=None, body=None):
        """(status, payload) without raising: for checks that expect a refusal."""
        return self._request(method, path, params, body)


BYTEA_PREFIX = "\\" + "x"  # PostgreSQL hex format: backslash, x, then hex digits


def to_bytea(blob: bytes) -> str:
    """bytea literal for PostgREST JSON."""
    return BYTEA_PREFIX + blob.hex()


def from_bytea(value: str) -> bytes:
    if not value.startswith(BYTEA_PREFIX):
        raise ValueError("unexpected bytea format")
    return bytes.fromhex(value[2:])
