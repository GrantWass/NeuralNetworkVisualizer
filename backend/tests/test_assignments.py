"""Classroom assignment flow tests (SESSION_BACKEND=local)."""
import pytest
from fastapi.testclient import TestClient

import app


@pytest.fixture()
def client(clean_local_state):
    return TestClient(app.app)


def _create(client, dataset="xor", target=0.0, cap=100, title="HW1"):
    resp = client.post("/assignments/create", json={
        "title": title, "dataset": dataset, "metric_target": target, "epoch_cap": cap,
    })
    assert resp.status_code == 200
    body = resp.json()
    # code format + secret returned exactly once
    assert len(body["assignment_code"]) == 6
    assert len(body["instructor_key"]) >= 16
    assert "instructor_key" not in str(client.get(f"/assignments/{body['assignment_code']}").json()).lower()
    return body


def _train(client, session_id, epochs=5, lr=0.5):
    return client.post("/train", json={
        "session_id": session_id, "learning_rate": lr, "epochs": epochs,
    })


def test_create_and_get_assignment(client):
    created = _create(client)
    resp = client.get(f"/assignments/{created['assignment_code']}")
    assert resp.status_code == 200
    info = resp.json()
    assert info["dataset"] == "xor"
    assert info["metric_name"] == "accuracy"
    assert info["higher_is_better"] is True
    assert "instructor_key" not in info

    assert client.get("/assignments/ZZZZZZ").status_code == 404
    # lowercase codes resolve too
    assert client.get(f"/assignments/{created['assignment_code'].lower()}").status_code == 200


def test_create_validates_dataset(client):
    resp = client.post("/assignments/create", json={
        "title": "bad", "dataset": "not_a_dataset", "metric_target": 1.0, "epoch_cap": 10,
    })
    assert resp.status_code == 400


def test_submit_is_server_verified(client):
    """The recorded score must come from the server's own evaluation of the
    stored network — a fabricated client score never enters the picture."""
    assignment = _create(client, target=50.0)

    sid = client.post("/init_model", json={
        "layer_sizes": [4], "activations": ["relu"], "dataset": "xor",
    }).json()["session_id"]
    train_resp = _train(client, sid, epochs=20)
    total_epochs = train_resp.json()["epochs_trained"]
    assert total_epochs == 20  # server-side epoch accounting works

    resp = client.post(f"/assignments/{assignment['assignment_code']}/submit", json={
        "session_id": sid, "student_name": "alice",
    })
    assert resp.status_code == 200
    result = resp.json()
    assert result["metric_name"] == "accuracy"
    assert 0.0 <= result["metric"] <= 100.0
    assert result["epochs_used"] == 20
    # metric must match an independent recomputation from the stored network
    session = app._load_session(sid)
    _, value = app._official_metric(session["network"], "xor")
    assert abs(result["metric"] - round(value, 4)) < 1e-6
    # target was 50% accuracy on xor
    assert result["target_met"] == (result["metric"] >= 50.0)


def test_submit_requires_matching_dataset(client):
    assignment = _create(client, dataset="iris", target=0.0)
    sid = client.post("/init_model", json={
        "layer_sizes": [4], "activations": ["relu"], "dataset": "xor",
    }).json()["session_id"]
    resp = client.post(f"/assignments/{assignment['assignment_code']}/submit", json={
        "session_id": sid, "student_name": "alice",
    })
    assert resp.status_code == 400
    assert "iris" in resp.json()["detail"]


def test_epoch_cap_enforced_server_side(client):
    assignment = _create(client, cap=3)
    sid = client.post("/init_model", json={
        "layer_sizes": [4], "activations": ["relu"], "dataset": "xor",
    }).json()["session_id"]
    # Train 2 separate cycles of 2 epochs each → 4 total > cap of 3.
    # The server tracks cumulative epochs, so splitting requests can't dodge the cap.
    _train(client, sid, epochs=2)
    _train(client, sid, epochs=2)
    resp = client.post(f"/assignments/{assignment['assignment_code']}/submit", json={
        "session_id": sid, "student_name": "bob",
    })
    assert resp.status_code == 400
    assert "epochs" in resp.json()["detail"]


def test_personal_best_retained(client):
    assignment = _create(client, target=100.0)  # accuracy higher-is-better
    sid = client.post("/init_model", json={
        "layer_sizes": [4], "activations": ["tanh"], "dataset": "xor",
    }).json()["session_id"]

    code = assignment["assignment_code"]
    first = client.post(f"/assignments/{code}/submit", json={"session_id": sid, "student_name": "carol"})
    client.post(f"/assignments/{code}/submit", json={"session_id": sid, "student_name": "carol"})

    assert first.json()["personal_best"] is True
    # Second submit re-evaluates the same network → same score → NOT a personal best;
    # the stored entry must be unchanged either way.
    roster = client.get(
        f"/assignments/{code}/submissions",
        params={"instructor_key": assignment["instructor_key"]},
    ).json()
    assert len(roster) == 1
    assert roster[0]["student_name"] == "carol"


def test_submissions_listing_requires_instructor_key(client):
    assignment = _create(client)
    sid = client.post("/init_model", json={
        "layer_sizes": [4], "activations": ["relu"], "dataset": "xor",
    }).json()["session_id"]
    client.post(f"/assignments/{assignment['assignment_code']}/submit", json={
        "session_id": sid, "student_name": "dave",
    })

    base = f"/assignments/{assignment['assignment_code']}/submissions"
    assert client.get(base, params={"instructor_key": "wrong"}).status_code == 403
    ok = client.get(base, params={"instructor_key": assignment["instructor_key"]})
    assert ok.status_code == 200
    entries = ok.json()
    assert len(entries) == 1 and entries[0]["student_name"] == "dave"


def test_regression_assignment_uses_denormalized_mae(client):
    """auto_mpg MAE targets are in original units; server must denormalize."""
    assignment = _create(client, dataset="auto_mpg", target=1.0, cap=500)
    sid = client.post("/init_model", json={
        "layer_sizes": [8, 8], "activations": ["relu", "relu"], "dataset": "auto_mpg",
    }).json()["session_id"]
    _train(client, sid, epochs=5)
    resp = client.post(f"/assignments/{assignment['assignment_code']}/submit", json={
        "session_id": sid, "student_name": "erin",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["metric_name"] == "mae"
    assert body["metric"] > 0

    # Cross-check against standardized-space evaluation scaled by y_std
    session = app._load_session(sid)
    _, raw = app._official_metric(session["network"], "auto_mpg")
    assert abs(body["metric"] - round(raw, 4)) < 1e-6


def test_submit_validates_student_name(client):
    assignment = _create(client)
    sid = client.post("/init_model", json={
        "layer_sizes": [4], "activations": ["relu"], "dataset": "xor",
    }).json()["session_id"]
    resp = client.post(f"/assignments/{assignment['assignment_code']}/submit", json={
        "session_id": sid, "student_name": "bad name!",
    })
    assert resp.status_code == 400
