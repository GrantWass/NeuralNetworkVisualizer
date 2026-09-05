"""Tests for the leaderboard username guardrails in app.py.

Covers both layers:
  * the character allowlist (letters, digits, _ and -, 1-32 chars)
  * the profanity blocklist (including look-alike/separator/repeat evasions)
  * benign names that merely resemble blocked words ("Nigeria", "Cassandra")
  * sanitize-on-read masking for entries stored before the filter existed
"""

import pathlib
import re

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


def test_normalization_reverses_leet_and_separators():
    assert normalize_for_profanity_check("Sh1t") == "shit"
    assert normalize_for_profanity_check("f-u-c-k") == "fuck"
    # Repeats are preserved: the patterns tolerate them, so the name under
    # test is never shortened into an unrelated word.
    assert normalize_for_profanity_check("fuuuck") == "fuuuck"


@pytest.mark.parametrize("name", [
    "Grasshopper", "ClassDude", "Sweetheart", "Cassandra", "hello",
    "shellcollector", "Pisa",
])
def test_keeps_benign_names_that_resemble_bad_words(name):
    assert_not_profane(name)


@pytest.mark.parametrize("name", [
    # Collapsing repeats on the blocklist shortened "nigger" to "niger",
    # which rejected every one of these. Regression guard.
    "Nigeria", "Nigerian", "NigerianDev", "Niger", "nigeria_2026",
])
def test_does_not_reject_nigeria(name):
    assert_not_profane(name)
    accepts(name)


def test_still_rejects_the_slur_with_repeats():
    for name in ["nigger", "niiigger", "n1gggerr"]:
        assert contains_profanity(name), name


def test_blocklist_matches_the_frontend_copy():
    """The TS mirror in username.ts must stay in sync with PROFANITY_BLOCKLIST."""
    ts_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "neural-network-visual/components/network/lib/username.ts"
    )
    source = ts_path.read_text()
    block = re.search(r"const PROFANITY_BLOCKLIST = \[(.*?)\];", source, re.S)
    assert block, "could not locate PROFANITY_BLOCKLIST in username.ts"
    ts_words = set(re.findall(r'"([^"]+)"', block.group(1)))
    assert ts_words == set(app.PROFANITY_BLOCKLIST), (
        f"only in backend: {set(app.PROFANITY_BLOCKLIST) - ts_words}; "
        f"only in frontend: {ts_words - set(app.PROFANITY_BLOCKLIST)}"
    )


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


# ── Epoch-cap gating: mid-run scores are provisional, not submittable ─────────


@pytest.mark.parametrize("dataset,cap", [
    ("iris", 100), ("auto_mpg", 200), ("mnist", 300),
])
def test_api_rejects_pre_cap_submissions(client, dataset, cap):
    resp = client.post(
        "/leaderboard/submit",
        json={"dataset": dataset, "score": 50.0, "epoch": cap - 1, "username": "EarlyBird"},
    )
    assert resp.status_code == 400, resp.text
    assert f"epoch {cap}" in resp.json()["detail"], resp.text
    # Nothing stored from the rejected submit
    assert client.get(f"/leaderboard/{dataset}").json()["entries"] == []


@pytest.mark.parametrize("dataset,cap", [
    ("iris", 100), ("auto_mpg", 200), ("mnist", 300),
])
def test_api_accepts_cap_and_overshoot_submissions(client, dataset, cap):
    for epoch in (cap, cap + 10):
        resp = client.post(
            "/leaderboard/submit",
            json={"dataset": dataset, "score": 50.0, "epoch": epoch, "username": f"Runner{epoch}"},
        )
        assert resp.status_code == 200 and resp.json()["accepted"] is True, resp.text


def test_api_xor_has_no_epoch_cap(client):
    resp = client.post(
        "/leaderboard/submit",
        json={"dataset": "xor", "score": 50.0, "epoch": 5, "username": "XorFan"},
    )
    assert resp.status_code == 200 and resp.json()["accepted"] is True, resp.text
