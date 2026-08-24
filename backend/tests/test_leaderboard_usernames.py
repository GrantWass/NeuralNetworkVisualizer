"""Tests for the leaderboard username guardrails in app.py.

Covers both layers:
  * the character allowlist (letters, digits, _ and -, 1-32 chars)
  * the profanity blocklist (including look-alike/separator/repeat evasions)
  * sanitize-on-read masking for entries stored before the filter existed
"""

import pytest
from fastapi import HTTPException

import app
from app import (
    contains_profanity,
    normalize_for_profanity_check,
    sanitize_username,
    validate_username,
)


def accepts(raw):
    validate_username(raw)  # should not raise


def rejects_with(raw, expected_detail_part):
    with pytest.raises(HTTPException) as exc_info:
        validate_username(raw)
    assert expected_detail_part in exc_info.value.detail, f"{raw!r} -> {exc_info.value.detail!r}"


def assert_not_profane(name):
    assert not contains_profanity(name), f"{name!r} wrongly flagged as profane"


# ── Layer 1: allowlist ────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "Alice", "neural_ninja", "top-score", "Grasshopper", "ClassOf2026",
    "Sweetheart", "hello", "a",
])
def test_allowlist_accepts_ordinary_names(name):
    accepts(name)


@pytest.mark.parametrize("raw", [
    "", "   ", "a" * 33, "has space", "bad.name!", "<script>", "naïve",
])
def test_allowlist_rejects_bad_characters_and_lengths(raw):
    rejects_with(raw, "1–32 characters")


# ── Layer 2: profanity blocklist ─────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "shit", "fucku", "bitchy", "penis", "dildo", "retard",
])
def test_rejects_plain_profanity(name):
    rejects_with(name, "offensive")


@pytest.mark.parametrize("name", [
    "Sh1t", "sh1tt", "B1tch", "n1gger", "F4gg0t", "a55hole", "d1ld0",
])
def test_rejects_lookalike_digit_substitutions(name):
    rejects_with(name, "offensive")


@pytest.mark.parametrize("name", [
    "f-u-c-k", "sh_it", "dumb_ass", "fuuuck", "shiiiiit", "FuUuUck",
])
def test_rejects_separator_and_repeat_evasions(name):
    rejects_with(name, "offensive")


def test_normalization_collapses_skeletons():
    assert normalize_for_profanity_check("Sh1t") == "shit"
    assert normalize_for_profanity_check("f-u-c-k") == "fuck"
    assert normalize_for_profanity_check("fuuuck") == "fuck"


@pytest.mark.parametrize("name", [
    "Grasshopper", "ClassDude", "Sweetheart", "Cassandra", "hello",
    "shellcollector", "Pisa",
])
def test_keeps_benign_names_that_resemble_bad_words(name):
    assert_not_profane(name)


# ── Display-side sanitization of legacy rows ─────────────────────────────────


@pytest.mark.parametrize("stored,expected", [
    ("sh1t", "***"),
    ("<i>slut</i>", "***"),
    ("  Bob<script> ", "Bobscript"),
    ("Grasshopper", "Grasshopper"),
])
def test_sanitize_username(stored, expected):
    assert sanitize_username(stored) == expected


# ── End-to-end through the API (SESSION_BACKEND=local) ────────────────────────


@pytest.fixture()
def client(clean_local_state):
    from fastapi.testclient import TestClient

    return TestClient(app.app)


def _submit(client, username, score=50.0):
    return client.post(
        "/leaderboard/submit",
        json={"dataset": "iris", "score": score, "epoch": 100, "username": username},
    )


def test_api_rejects_profane_submissions(client):
    resp = _submit(client, "Sh1t")
    assert resp.status_code == 400 and "offensive" in resp.json()["detail"], resp.text

    resp = _submit(client, "f-u-c-k")
    assert resp.status_code == 400 and "offensive" in resp.json()["detail"], resp.text


def test_api_accepts_clean_submission(client):
    resp = _submit(client, "Alice")
    assert resp.status_code == 200 and resp.json()["accepted"] is True, resp.text


def test_api_masks_legacy_entries_on_read(client):
    app._local_sessions["leaderboard:iris"] = [{
        "username": "<i>dumb_ass</i>",
        "score": 99.9,
        "epoch": 100,
        "submitted_at": 1,
    }]
    resp = client.get("/leaderboard/iris")
    usernames = [e["username"] for e in resp.json()["entries"]]
    assert "***" in usernames, resp.text
