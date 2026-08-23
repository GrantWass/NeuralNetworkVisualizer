"""Sanity checks for the leaderboard username guardrails in app.py.

Covers both layers:
  * the character allowlist (letters, digits, _ and -, 1-32 chars)
  * the profanity blocklist (including look-alike/separator/repeat evasions)

Run directly (no pytest needed):
    python3 test_leaderboard_usernames.py
"""

from fastapi import HTTPException

from app import (
    contains_profanity,
    normalize_for_profanity_check,
    sanitize_username,
    validate_username,
)

CHECKS = []


def check(description, fn):
    try:
        fn()
        CHECKS.append((description, True, None))
    except AssertionError as exc:
        CHECKS.append((description, False, str(exc)))


def accepts(raw):
    validate_username(raw)  # should not raise


def rejects_with(raw, expected_detail_part):
    try:
        validate_username(raw)
    except HTTPException as exc:
        assert expected_detail_part in exc.detail, f"{raw!r} -> {exc.detail!r}"
        return
    raise AssertionError(f"{raw!r} was accepted")


def assert_not_profane(name):
    assert not contains_profanity(name), f"{name!r} wrongly flagged as profane"


def _assert_equal(actual, expected):
    assert actual == expected, f"expected {expected!r}, got {actual!r}"


# ── Layer 1: allowlist ────────────────────────────────────────────────────────

check("accepts ordinary names", lambda: [accepts(n) for n in [
    "Alice", "neural_ninja", "top-score", "Grasshopper", "ClassOf2026",
    "Sweetheart", "hello", "a",
]])

check("rejects empty / whitespace-only / too long / bad characters",
      lambda: [rejects_with(r, "1–32 characters") for r in [
          "", "   ", "a" * 33, "has space", "bad.name!", "<script>", "naïve",
      ]])

# ── Layer 2: profanity blocklist ─────────────────────────────────────────────

check("rejects plain profanity",
      lambda: [rejects_with(r, "offensive") for r in [
          "shit", "fucku", "bitchy", "penis", "dildo", "retard",
      ]])

check("rejects look-alike digit substitutions",
      lambda: [rejects_with(r, "offensive") for r in [
          "Sh1t", "sh1tt", "B1tch", "n1gger", "F4gg0t", "a55hole", "d1ld0",
      ]])

check("rejects separator and repeat evasions",
      lambda: [rejects_with(r, "offensive") for r in [
          "f-u-c-k", "sh_it", "dumb_ass", "fuuuck", "shiiiiit", "FuUuUck",
      ]])

check("keeps benign names that merely resemble bad words",
      lambda: [assert_not_profane(n) for n in [
          "Grasshopper", "ClassDude", "Sweetheart", "Cassandra", "hello",
          "shellcollector", "Pisa",
      ]])

# ── Display-side sanitization of legacy rows ─────────────────────────────────

check("masks stored names that trip the filter", lambda: [
    _assert_equal(sanitize_username("sh1t"), "***"),
    _assert_equal(sanitize_username("<i>slut</i>"), "***"),
])

check("still cleans and passes through benign legacy names", lambda: [
    _assert_equal(sanitize_username("  Bob<script> "), "Bobscript"),
    _assert_equal(sanitize_username("Grasshopper"), "Grasshopper"),
])


# ── End-to-end through the API (DynamoDB stubbed in memory) ──────────────────

class _StubTable:
    """Minimal stand-in for the DynamoDB leaderboard table."""

    def __init__(self):
        self.items = {}

    def get_item(self, Key):
        dataset = Key["dataset"]
        return {"Item": self.items[dataset]} if dataset in self.items else {}

    def put_item(self, Item):
        self.items[Item["dataset"]] = Item


def _api_checks():
    import app

    app._leaderboard_table = _StubTable()
    from fastapi.testclient import TestClient

    client = TestClient(app.app)

    def submit(username, score=50.0):
        return client.post(
            "/leaderboard/submit",
            json={"dataset": "iris", "score": score, "epoch": 100, "username": username},
        )

    resp = submit("Sh1t")
    assert resp.status_code == 400 and "offensive" in resp.json()["detail"], resp.text

    resp = submit("f-u-c-k")
    assert resp.status_code == 400 and "offensive" in resp.json()["detail"], resp.text

    resp = submit("Alice")
    assert resp.status_code == 200 and resp.json()["accepted"] is True, resp.text

    # A legacy row written before the filter existed must come back masked.
    app._leaderboard_table.put_item(Item={
        "dataset": "iris",
        "entries": [{"username": "<i>dumb_ass</i>", "score": "99.9", "epoch": 100,
                     "submitted_at": 1}],
        "updated_at": 1,
    })
    resp = client.get("/leaderboard/iris")
    usernames = [e["username"] for e in resp.json()["entries"]]
    assert "***" in usernames, resp.text


check("API rejects profane submissions and masks legacy entries", _api_checks)


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    failed = [c for c in CHECKS if not c[1]]
    for description, ok, err in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}  {description}" + (f"\n      {err}" if err else ""))
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    raise SystemExit(1 if failed else 0)
