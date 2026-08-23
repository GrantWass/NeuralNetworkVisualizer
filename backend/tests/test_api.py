import pytest
from fastapi.testclient import TestClient

import app


@pytest.fixture()
def client(clean_local_state):
    return TestClient(app.app)


def test_init_and_train_roundtrip(client):
    resp = client.post("/init_model", json={
        "layer_sizes": [4],
        "activations": ["relu"],
        "dataset": "xor",
    })
    assert resp.status_code == 200
    body = resp.json()
    session_id = body["session_id"]
    assert body["layer_sizes"] == [2, 4, 1]

    resp = client.post("/train", json={
        "session_id": session_id,
        "learning_rate": 0.5,
        "epochs": 3,
    })
    assert resp.status_code == 200
    results = resp.json()["training_results"]
    assert len(results) == 3
    # Only the last epoch carries full layer detail
    assert results[0]["layers"] == []
    assert results[-1]["layers"]
    assert results[-1]["epoch"] == 3
    # Every epoch reports held-out test metrics
    for r in results:
        assert r["test_name"] == r["name"]
        assert r["test_loss"] >= 0
        assert isinstance(r["test_metric"], (int, float))

    # Session survives save/load (JSON serialization path)
    resp = client.post("/train", json={"session_id": session_id, "epochs": 1})
    assert resp.status_code == 200

    resp = client.post("/clear_session", params={"session_id": session_id})
    assert resp.status_code == 200
    assert client.post("/train", json={"session_id": session_id, "epochs": 1}).status_code == 404


def test_init_model_validation(client):
    resp = client.post("/init_model", json={
        "layer_sizes": [4],
        "activations": ["relu", "tanh"],  # length mismatch
        "dataset": "xor",
    })
    assert resp.status_code == 400


def test_set_weights_out_of_range(client):
    session_id = client.post("/init_model", json={
        "layer_sizes": [4], "activations": ["relu"], "dataset": "xor",
    }).json()["session_id"]

    assert client.post("/set_weights", json={
        "session_id": session_id, "layer_index": 99,
        "from_index": 0, "to_index": 0, "new_value": 0.1,
    }).status_code == 400


def test_leaderboard_flow(client):
    dataset = "xor"
    for i, score in enumerate([12.0, 5.0]):
        resp = client.post("/leaderboard/submit", json={
            "dataset": dataset, "score": score, "epoch": score, "username": f"user{i}",
        })
        assert resp.status_code == 200
        assert resp.json()["accepted"] is True

    entries = resp.json()["entries"]
    assert entries[0]["username"] == "user1"  # lower is better for xor

    resp = client.get(f"/leaderboard/{dataset}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["higher_is_better"] is False
    assert [e["score"] for e in data["entries"]] == sorted(e["score"] for e in data["entries"])


def test_leaderboard_rejects_non_qualifying_score(client):
    dataset = "iris"  # higher is better
    for i in range(10):
        client.post("/leaderboard/submit", json={
            "dataset": dataset, "score": 50.0 + i, "epoch": 100, "username": f"user{i}",
        })
    resp = client.post("/leaderboard/submit", json={
        "dataset": dataset, "score": 10.0, "epoch": 100, "username": "bottom",
    })
    assert resp.json()["accepted"] is False


def test_leaderboard_validates_username(client):
    resp = client.post("/leaderboard/submit", json={
        "dataset": "xor", "score": 1.0, "epoch": 1, "username": "bad name!",
    })
    assert resp.status_code == 400


def test_leaderboard_unknown_dataset(client):
    assert client.get("/leaderboard/not_a_dataset").status_code == 400
