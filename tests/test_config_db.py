"""T12: .env loading and bytea helpers (offline)."""

import os

import pytest

from earworm.config import load_env, require
from earworm.db import Supabase, from_bytea, to_bytea


def test_load_env_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nA_EARWORM=1\nB_EARWORM = 'two'\nEMPTY_EARWORM=\n", encoding="utf-8")
    monkeypatch.setenv("A_EARWORM", "already")
    monkeypatch.delenv("B_EARWORM", raising=False)
    monkeypatch.delenv("EMPTY_EARWORM", raising=False)
    load_env(env)
    assert os.environ["A_EARWORM"] == "already"
    assert os.environ["B_EARWORM"] == "two"
    assert "EMPTY_EARWORM" not in os.environ


def test_require_names_missing_settings(monkeypatch):
    monkeypatch.delenv("NOPE_EARWORM", raising=False)
    with pytest.raises(RuntimeError, match="NOPE_EARWORM"):
        require("NOPE_EARWORM")


def test_bytea_round_trip():
    blob = bytes(range(256))
    assert to_bytea(blob)[:2] == chr(92) + "x"  # backslash, x
    assert from_bytea(to_bytea(blob)) == blob


def test_new_style_keys_stay_out_of_authorization_header():
    assert "Authorization" not in Supabase("https://x.supabase.co", "sb_secret_abc").headers
    legacy = Supabase("https://x.supabase.co", "eyJhbGciOi.legacy.jwt").headers
    assert legacy["Authorization"] == "Bearer eyJhbGciOi.legacy.jwt"


def test_delete_without_filter_is_refused():
    with pytest.raises(ValueError):
        Supabase("https://x.supabase.co", "k").delete("songs")


def test_project_url_with_or_without_rest_suffix():
    for url in (
        "https://x.supabase.co",
        "https://x.supabase.co/",
        "https://x.supabase.co/rest/v1/",
    ):
        assert Supabase(url, "k").base == "https://x.supabase.co/rest/v1"
