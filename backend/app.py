import hmac
import json
import os
import re
import secrets
import time
import uuid
from collections import OrderedDict
from decimal import Decimal
from typing import List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel, Field

from datasets import load_dataset  # Assume a function to load the dataset

# import math  # re-enable with /attention endpoint (used for sqrt(HEAD_DIM))
from NeuralNetwork import NeuralNetwork
from utils import calculate_metric, loss_function

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Browsers reject credentials with wildcard origins; only enable for explicit origins
    allow_credentials="*" not in ALLOWED_ORIGINS,
    allow_methods=["*"],  # Allow all HTTP methods like GET, POST, OPTIONS
    allow_headers=["*"],  # Allow all headers
)

AWS_REGION = "us-east-2"
SESSION_BUCKET = "nn-sessions-data"
SESSION_TTL_SECONDS = 3 * 60 * 60  # 3 hours

# "local" keeps sessions in-process (no AWS calls) — used for local dev & tests
SESSION_BACKEND = os.environ.get("SESSION_BACKEND", "aws")

_aws = None


def _get_aws():
    # Lazy-init so importing this module never touches AWS config/credentials
    global _aws
    if _aws is None:
        import boto3
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        _aws = {
            "sessions": dynamodb.Table("nn-sessions"),
            "leaderboard": dynamodb.Table("nn-leaderboard"),
            "assignments": dynamodb.Table("nn-assignments"),
            "submissions": dynamodb.Table("nn-submissions"),
            "s3": boto3.client("s3", region_name=AWS_REGION),
        }
    return _aws

LEADERBOARD_CONFIG = {
    "xor":      {"higher_is_better": False, "display": "Fewest epochs to 100%",     "epoch_cap": None},
    "iris":     {"higher_is_better": True,  "display": "Accuracy at epoch 100 (%)", "epoch_cap": 100},
    "auto_mpg": {"higher_is_better": False, "display": "MAE at epoch 200",           "epoch_cap": 200},
    "mnist":    {"higher_is_better": True,  "display": "Accuracy at epoch 300 (%)", "epoch_cap": 300},
}

# In-process LRU cache (L1) — avoids a round-trip when the same
# Lambda instance handles back-to-back requests for the same session.
_CACHE_MAX = 32
_session_cache: OrderedDict = OrderedDict()

# In-process store for SESSION_BACKEND=local
_local_sessions: dict = {}


def _cache_get(session_id: str) -> Optional[dict]:
    if session_id in _session_cache:
        _session_cache.move_to_end(session_id)
        return _session_cache[session_id]
    return None


def _cache_put(session_id: str, session: dict) -> None:
    _session_cache[session_id] = session
    _session_cache.move_to_end(session_id)
    if len(_session_cache) > _CACHE_MAX:
        _session_cache.popitem(last=False)


def _cache_evict(session_id: str) -> None:
    _session_cache.pop(session_id, None)


# ── Leaderboard username guardrails ──────────────────────────────────────────
# Two layers:
#   1. Allowlist — a name may only contain letters, digits, _ and -.
#      That single rule blocks HTML/script injection (<script>, quotes, &),
#      control characters, unicode look-alikes, and absurd lengths in one go.
#   2. Profanity blocklist — names are normalized (lowercase, common digit
#      substitutions reversed, separators removed, repeated letters collapsed)
#      so "Sh1t", "f-u-c-k" and "fuuuck" all reduce to something we can match
#      against words that are never acceptable as a name. Submissions are
#      rejected; entries stored before this filter existed are masked on read.

USERNAME_MAX_LENGTH = 32
_USERNAME_DISALLOWED = re.compile(r"[^a-zA-Z0-9_-]")

# Words never allowed in a leaderboard name. Must stay in sync with
# PROFANITY_BLOCKLIST in neural-network-visual/components/network/lib/username.ts.
# Mild-but-ambiguous words ("ass", "hell", "damn") are deliberately left out:
# substring matching would wrongly reject names like "class" or "shell". Words
# that shrink to ~3 letters when repeats are collapsed ("piss" → "pis") are
# excluded for the same reason — they'd match unrelated names like "Pisa".
PROFANITY_BLOCKLIST = frozenset({
    "fuck", "shit", "bitch", "bastard", "cunt", "whore", "slut",
    "wanker", "twat", "bullshit", "dickhead", "asshole",
    "arsehole", "jackass", "dumbass",
    # slurs
    "nigger", "nigga", "faggot", "chink", "kike", "tranny",
    "wetback", "towelhead", "retard",
    # crude anatomy/harassment terms
    "penis", "dildo",
})

# Reverse common letter/digit look-alikes before matching ("sh1t" → "shit").
_PROFANITY_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b"})


def normalize_for_profanity_check(name: str) -> str:
    """Reduce a name to its 'letter skeleton' so evasions collapse to the word."""
    lowered = name.lower().translate(_PROFANITY_LEET)
    without_separators = lowered.replace("-", "").replace("_", "")
    return re.sub(r"(.)\1+", r"\1", without_separators)


# Pre-collapse the list too, so needles match the collapsed haystack
# (e.g. "asshole" is searched for as "ashole" once repeats are collapsed).
_PROFANITY_BLOCKLIST_NORMALIZED = frozenset(
    normalize_for_profanity_check(word) for word in PROFANITY_BLOCKLIST
)


def contains_profanity(name: str) -> bool:
    skeleton = normalize_for_profanity_check(name)
    return any(word in skeleton for word in _PROFANITY_BLOCKLIST_NORMALIZED)


def validate_username(raw: str) -> str:
    """Reject anything outside the allowlist or on the blocklist; return the cleaned name."""
    username = raw.strip()
    if not username or len(username) > USERNAME_MAX_LENGTH or _USERNAME_DISALLOWED.search(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 1–32 characters: letters, digits, _ or -",
        )
    if contains_profanity(username):
        raise HTTPException(
            status_code=400,
            detail="Please choose a different username — offensive names aren't allowed.",
        )
    return username


def sanitize_username(raw: str) -> str:
    """Clean a name read back from storage before display.

    Entries written before these rules existed (or edited directly in
    DynamoDB) may still hold odd characters — strip them. Names that trip
    the profanity filter are masked rather than shown.
    """
    cleaned = _USERNAME_DISALLOWED.sub("", raw.strip())[:USERNAME_MAX_LENGTH]
    if contains_profanity(cleaned):
        return "***"
    return cleaned


# ── Leaderboard helpers ───────────────────────────────────────────────────────

def _leaderboard_read(dataset: str) -> tuple[list, Optional[int]]:
    if SESSION_BACKEND == "local":
        entries = _local_sessions.get(f"leaderboard:{dataset}", [])
        return [dict(e) for e in entries], None
    resp = _get_aws()["leaderboard"].get_item(Key={"dataset": dataset})
    item = resp.get("Item", {})
    return item.get("entries", []), item.get("updated_at")


def _leaderboard_write(dataset: str, raw_entries: list, prev_updated_at: Optional[int]) -> bool:
    # Conditional write so concurrent submissions can't clobber each other;
    # returns False when another writer modified the entry in the meantime.
    updated_at = int(time.time())
    if SESSION_BACKEND == "local":
        _local_sessions[f"leaderboard:{dataset}"] = raw_entries
        return True
    leaderboard_table = _get_aws()["leaderboard"]
    try:
        leaderboard_table.put_item(
            Item={
                "dataset": dataset,
                "entries": raw_entries,
                "updated_at": updated_at,
            },
            ConditionExpression="attribute_not_exists(updated_at) OR updated_at = :prev",
            ExpressionAttributeValues={":prev": prev_updated_at},
        )
        return True
    except leaderboard_table.meta.client.exceptions.ConditionalCheckFailedException:
        return False

def _is_better(dataset: str, new_score: float, existing_score: float) -> bool:
    return new_score > existing_score if LEADERBOARD_CONFIG[dataset]["higher_is_better"] else new_score < existing_score


def _qualifies_for_top10(dataset: str, score: float, entries: list) -> Optional[int]:
    for i, entry in enumerate(entries):
        if _is_better(dataset, score, float(entry["score"])):
            return i
    if len(entries) < 10:
        return len(entries)
    return None


# ── Leaderboard Pydantic models ───────────────────────────────────────────────

class LeaderboardEntry(BaseModel):
    rank: int
    username: str
    score: float
    epoch: int
    submitted_at: int

class LeaderboardResponse(BaseModel):
    dataset: str
    metric_display: str
    higher_is_better: bool
    epoch_cap: Optional[int]
    entries: List[LeaderboardEntry]

class LeaderboardSubmitRequest(BaseModel):
    dataset: str
    score: float
    epoch: int
    username: str

class LeaderboardSubmitResponse(BaseModel):
    accepted: bool
    rank: Optional[int]
    entries: List[LeaderboardEntry]


def _save_session(session_id: str, session: dict) -> None:
    _cache_put(session_id, session)
    # Store only serialized state (JSON, not pickle). epochs_trained is
    # server-side bookkeeping so epoch caps can be enforced authoritatively.
    payload = json.dumps({
        "network": session["network"].to_state(),
        "dataset": session["dataset"],
        "epochs_trained": session.get("epochs_trained", 0),
    }).encode()
    if SESSION_BACKEND == "local":
        _local_sessions[session_id] = payload
        return
    aws = _get_aws()
    s3_key = f"sessions/{session_id}.json"
    aws["s3"].put_object(Bucket=SESSION_BUCKET, Key=s3_key, Body=payload)
    aws["sessions"].put_item(Item={
        "session_id": session_id,
        "s3_key": s3_key,
        "ttl": int(time.time()) + SESSION_TTL_SECONDS,
    })


def _load_session(session_id: str) -> Optional[dict]:
    cached = _cache_get(session_id)
    if cached is not None:
        return cached
    if SESSION_BACKEND == "local":
        payload = _local_sessions.get(session_id)
    else:
        aws = _get_aws()
        resp = aws["sessions"].get_item(Key={"session_id": session_id})
        item = resp.get("Item")
        if item is None:
            return None
        s3_key = item.get("s3_key")
        if s3_key is None:
            return None
        payload = aws["s3"].get_object(Bucket=SESSION_BUCKET, Key=s3_key)["Body"].read()

    if payload is None:
        return None
    data = json.loads(payload)
    session = {
        "network": NeuralNetwork.from_state(data["network"]),
        "dataset": data["dataset"],
        "epochs_trained": data.get("epochs_trained", 0),
    }
    _cache_put(session_id, session)
    return session


def _delete_session(session_id: str) -> None:
    _cache_evict(session_id)
    if SESSION_BACKEND == "local":
        _local_sessions.pop(session_id, None)
        return
    _get_aws()["sessions"].delete_item(Key={"session_id": session_id})

# ------------------ Model Initialization Request ------------------ #
class InitModelRequest(BaseModel):
    layer_sizes: List[int]  # List of layer sizes, excluding input & output
    activations: List[str]  # Activation function for each layer (except input)
    dataset: str  # Name of dataset (e.g., "iris", "auto_mpg")

class InitModelResponse(BaseModel):
    message: str
    session_id: str
    layer_sizes: List[int]
    original_train_data: list
    network: dict  # Serialized network as dictionary
    y_mean: Optional[float] = None
    y_std: Optional[float] = None

@app.post("/init_model", response_model=InitModelResponse)
def init_model(request: InitModelRequest):
    session_id = str(uuid.uuid4())  # Generate a unique session ID

    # Load dataset
    X_train, _, Y_train, _, input_size, output_size, output_activation, original_train_data, y_mean, y_std = load_dataset(request.dataset)

    layers = [input_size] + request.layer_sizes + [output_size]

    # Ensure activations length matches hidden + output layers
    if len(request.activations) != len(request.layer_sizes):
        raise HTTPException(status_code=400, detail="Activations length must match number of layers.")

    activations = request.activations + [output_activation]
    network = NeuralNetwork(layers, activations, optimizer="batch")

    # Store only the model + dataset name — reload data on demand to stay under DynamoDB 400KB limit
    _save_session(session_id, {
        "network": network,
        "dataset": request.dataset,
        "epochs_trained": 0,
    })

    return InitModelResponse(
        message="Model initialized successfully",
        session_id=session_id,
        layer_sizes=layers,
        original_train_data=original_train_data[:30],
        network=network.to_dict(),
        y_mean=y_mean,
        y_std=y_std,
    )


# ------------------ Training Request ------------------ #
class TrainRequest(BaseModel):
    session_id: str  # User's session ID
    learning_rate: float = Field(default=0.01, gt=0, le=100)
    epochs: int = Field(default=10, ge=1, le=999)

class LayerDetail(BaseModel):
    weights: list
    biases: list
    Z: list
    A: list
    dW: list
    db: list
    dZ: list
    activation: str

class TrainResult(BaseModel):
    epoch: int
    input: list
    loss: float
    name: str  # Metric name (e.g., accuracy, mae)
    metric: Union[float, str]  # Metric could be accuracy or mae
    test_loss: float
    test_name: str
    test_metric: Union[float, str]
    layers: List[LayerDetail]

class TrainResponse(BaseModel):
    training_results: List[TrainResult]
    epochs_trained: int  # cumulative for this session (server-side bookkeeping)

@app.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    session = _load_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /init_model first.")

    network = session["network"]
    X_train, X_test, Y_train, Y_test, _, _, output_activation, _, _, _ = load_dataset(session["dataset"])

    training_results = []

    for epoch in range(request.epochs):
        result = network.train_step(X_train, Y_train, request.learning_rate)

        is_last = epoch == request.epochs - 1
        is_second_to_last = epoch == request.epochs - 2

        if is_last:
            # Full detail for visualization
            layers = [
                LayerDetail(
                    activation=layer["activation"],
                    Z=layer["Z"][:30].astype(np.float32).tolist(),
                    A=layer["A"][:30].astype(np.float32).tolist(),
                    dW=layer["dW"].astype(np.float32).tolist(),
                    db=layer["db"].astype(np.float32).tolist(),
                    dZ=layer["dZ"][:30].astype(np.float32).tolist(),
                    weights=layer["weights"].astype(np.float32).tolist(),
                    biases=layer["biases"].astype(np.float32).tolist(),
                ) for layer in result["layers"]
            ]
        elif is_second_to_last:
            # Weights + biases only — used for delta highlighting (prevWeights)
            layers = [
                LayerDetail(
                    weights=layer["weights"].astype(np.float32).tolist(),
                    biases=layer["biases"].astype(np.float32).tolist(),
                    Z=[], A=[], dW=[], db=[], dZ=[],
                    activation=layer["activation"]
                ) for layer in result["layers"]
            ]
        else:
            # Metrics only — no layer data needed for earlier epochs
            layers = []

        metric_name = "accuracy" if "accuracy" in result else "mae"
        metric_value = result.get("accuracy") if "accuracy" in result else result.get("mae")

        # Held-out test metrics so the UI can plot generalization, not just fit
        Y_test_hat = network.forward(X_test)
        test_loss = loss_function(Y_test_hat, Y_test, "mse" if output_activation == "linear" else "cross-entropy")
        test_metric = calculate_metric(Y_test_hat, Y_test, output_activation)

        training_results.append(TrainResult(
            epoch=epoch + 1,
            input=X_train[:30].tolist() if is_last else [],
            loss=result["loss"],
            name=metric_name,
            metric=metric_value,
            test_loss=test_loss,
            test_name=metric_name,
            test_metric=test_metric,
            layers=layers
        ))

    session["epochs_trained"] = session.get("epochs_trained", 0) + request.epochs
    _save_session(request.session_id, session)
    return TrainResponse(training_results=training_results, epochs_trained=session["epochs_trained"])


# ------------------ Set Weight ------------------ #
class SetWeightRequest(BaseModel):
    session_id: str
    layer_index: int   # which layer (0-indexed)
    from_index: int    # row in weight matrix (input neuron)
    to_index: int      # column in weight matrix (output neuron)
    new_value: float

class SetWeightResponse(BaseModel):
    layers: List[LayerDetail]

@app.post("/set_weights", response_model=SetWeightResponse)
def set_weight(request: SetWeightRequest):
    session = _load_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /init_model first.")

    network = session["network"]
    X_train, _, _, _, _, _, _, _, _, _ = load_dataset(session["dataset"])

    if request.layer_index < 0 or request.layer_index >= len(network.layers):
        raise HTTPException(status_code=400, detail="Invalid layer_index")
    layer = network.layers[request.layer_index]
    if request.from_index < 0 or request.from_index >= layer.weights.shape[0]:
        raise HTTPException(status_code=400, detail="Invalid from_index")
    if request.to_index < 0 or request.to_index >= layer.weights.shape[1]:
        raise HTTPException(status_code=400, detail="Invalid to_index")
    layer.weights[request.from_index][request.to_index] = request.new_value

    network.forward(X_train)

    layers = [
        LayerDetail(
            activation=lyr.activation,
            Z=lyr.Z[:30].astype(np.float32).tolist(),
            A=lyr.A[:30].astype(np.float32).tolist(),
            dW=lyr.dW.astype(np.float32).tolist() if lyr.dW is not None else [],
            db=lyr.db.astype(np.float32).tolist() if lyr.db is not None else [],
            dZ=lyr.dZ[:30].astype(np.float32).tolist() if lyr.dZ is not None else [],
            weights=lyr.weights.astype(np.float32).tolist(),
            biases=lyr.biases.astype(np.float32).tolist(),
        ) for lyr in network.layers
    ]

    _save_session(request.session_id, session)
    return SetWeightResponse(layers=layers)


# ------------------ Predict (MNIST) ------------------ #
class PredictRequest(BaseModel):
    session_id: str
    pixels: List[float]  # 784 values in [0, 1]

class PredictResponse(BaseModel):
    predicted_class: int
    confidences: List[float]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    session = _load_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /init_model first.")
    if len(request.pixels) != 784:
        raise HTTPException(status_code=400, detail="pixels must have exactly 784 values.")

    network = session["network"]
    X = np.array(request.pixels, dtype=np.float32).reshape(1, 784)
    Y_hat = network.forward(X)
    confidences = Y_hat[0].tolist()
    predicted_class = int(np.argmax(Y_hat[0]))
    return PredictResponse(predicted_class=predicted_class, confidences=confidences)


# ------------------ Clear Session ------------------ #
@app.post("/clear_session")
def clear_session(session_id: str):
    session = _load_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    _delete_session(session_id)
    return {"message": "Session cleared successfully"}


# ------------------ Leaderboard ------------------ #
MAX_LEADERBOARD_ENTRIES = 10
LEADERBOARD_WRITE_RETRIES = 3


def _leaderboard_entries(raw_entries: list) -> List[LeaderboardEntry]:
    # Sanitize on read: rows stored before validation existed (or edited
    # directly in DynamoDB) get cleaned, and names that trip the profanity
    # filter are masked instead of shown.
    return [
        LeaderboardEntry(
            rank=i + 1,
            username=sanitize_username(e["username"]),
            score=float(e["score"]),
            epoch=int(e["epoch"]),
            submitted_at=int(e["submitted_at"]),
        )
        for i, e in enumerate(raw_entries)
    ]


@app.get("/leaderboard/{dataset}", response_model=LeaderboardResponse)
def get_leaderboard(dataset: str):
    if dataset not in LEADERBOARD_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")
    cfg = LEADERBOARD_CONFIG[dataset]
    raw_entries, _ = _leaderboard_read(dataset)
    entries = _leaderboard_entries(raw_entries)
    return LeaderboardResponse(
        dataset=dataset,
        metric_display=cfg["display"],
        higher_is_better=cfg["higher_is_better"],
        epoch_cap=cfg["epoch_cap"],
        entries=entries,
    )


@app.post("/leaderboard/submit", response_model=LeaderboardSubmitResponse)
def submit_leaderboard(request: LeaderboardSubmitRequest):
    if request.dataset not in LEADERBOARD_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset}")

    username = validate_username(request.username)

    # Retry on contention so concurrent submissions can't silently drop entries
    for _ in range(LEADERBOARD_WRITE_RETRIES):
        raw_entries, prev_updated_at = _leaderboard_read(request.dataset)

        insert_idx = _qualifies_for_top10(request.dataset, request.score, raw_entries)
        if insert_idx is None:
            entries = _leaderboard_entries(raw_entries)
            return LeaderboardSubmitResponse(accepted=False, rank=None, entries=entries)

        new_entry = {
            "username": username,
            "score": Decimal(str(request.score)),
            "epoch": request.epoch,
            "submitted_at": int(time.time()),
        }
        raw_entries.insert(insert_idx, new_entry)
        raw_entries = raw_entries[:MAX_LEADERBOARD_ENTRIES]

        if _leaderboard_write(request.dataset, raw_entries, prev_updated_at):
            entries = _leaderboard_entries(raw_entries)
            return LeaderboardSubmitResponse(accepted=True, rank=insert_idx + 1, entries=entries)

    raise HTTPException(status_code=503, detail="Leaderboard is busy, please retry.")


# ------------------ Assignments (classroom) ------------------ #
# Instructors create an assignment (dataset + metric target + epoch cap) and
# share the join code. Students train a model and submit their session; the
# SERVER evaluates the stored network on the held-out test set, so scores are
# authoritative and can't be fabricated client-side.

def _metric_info(output_activation: str) -> tuple[str, bool]:
    """Returns (metric_name, higher_is_better) for an output activation."""
    if output_activation == "linear":
        return "mae", False
    return "accuracy", True


def _official_metric(network: NeuralNetwork, dataset: str) -> tuple[str, float]:
    """Server-side evaluation of a session's network on the held-out test set."""
    _, X_test, _, Y_test, _, _, output_activation, _, y_mean, y_std = load_dataset(dataset)
    Y_hat = network.forward(X_test)
    value = calculate_metric(Y_hat, Y_test, output_activation)
    metric_name, _ = _metric_info(output_activation)
    if metric_name == "mae" and y_std is not None:
        # Report MAE in original units (targets are standardized for auto_mpg)
        value *= y_std
    return metric_name, float(value)


class AssignmentCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)
    dataset: str
    metric_target: float
    epoch_cap: int = Field(ge=1, le=100_000)


class AssignmentInfo(BaseModel):
    assignment_code: str
    title: str
    dataset: str
    metric_target: float
    epoch_cap: int
    metric_name: str
    higher_is_better: bool

class AssignmentCreatedResponse(AssignmentInfo):
    instructor_key: str  # secret — required to view submissions; shown once at creation

class AssignmentSubmitRequest(BaseModel):
    session_id: str
    student_name: str

class AssignmentSubmitResponse(BaseModel):
    metric: float
    metric_name: str
    epochs_used: int
    target_met: bool
    personal_best: bool

class SubmissionEntry(BaseModel):
    student_name: str
    metric: float
    epochs_used: int
    target_met: bool
    submitted_at: int


@app.post("/assignments/create", response_model=AssignmentCreatedResponse)
def create_assignment(request: AssignmentCreateRequest):
    if request.dataset not in LEADERBOARD_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {request.dataset}")

    assignments_table = _get_aws()["assignments"]
    if SESSION_BACKEND != "local":
        existing_codes = set()
        resp = assignments_table.scan(ProjectionExpression="assignment_code")
        existing_codes = {i["assignment_code"] for i in resp.get("Items", [])}

    for _ in range(10):  # collision retries
        code = "".join(secrets.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        if SESSION_BACKEND == "local" or code not in existing_codes:
            break
    else:
        raise HTTPException(status_code=503, detail="Could not allocate an assignment code, please retry.")

    instructor_key = uuid.uuid4().hex
    metric_name, higher_is_better = _metric_info(load_dataset(request.dataset)[6])
    item = {
        "assignment_code": code,
        "instructor_key": instructor_key,
        "title": request.title.strip(),
        "dataset": request.dataset,
        "metric_name": metric_name,
        "higher_is_better": higher_is_better,
        "metric_target": Decimal(str(request.metric_target)),
        "epoch_cap": request.epoch_cap,
        "created_at": int(time.time()),
    }
    if SESSION_BACKEND == "local":
        _local_sessions[f"assignment:{code}"] = item
    else:
        assignments_table.put_item(Item=item)

    return AssignmentCreatedResponse(**item)


def _get_assignment(code: str) -> Optional[dict]:
    if SESSION_BACKEND == "local":
        return _local_sessions.get(f"assignment:{code}")
    resp = _get_aws()["assignments"].get_item(Key={"assignment_code": code})
    return resp.get("Item")


@app.get("/assignments/{code}", response_model=AssignmentInfo)
def get_assignment(code: str):
    item = _get_assignment(code.upper())
    if item is None:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    return AssignmentInfo(**item)


@app.post("/assignments/{code}/submit", response_model=AssignmentSubmitResponse)
def submit_assignment(code: str, request: AssignmentSubmitRequest):
    code = code.upper()
    item = _get_assignment(code)
    if item is None:
        raise HTTPException(status_code=404, detail="Assignment not found.")

    session = _load_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Train a model first.")
    if session["dataset"] != item["dataset"]:
        raise HTTPException(status_code=400, detail=(
            f"This assignment uses the '{item['dataset']}' dataset, but your session was trained on '{session['dataset']}'."
        ))

    epochs_used = session.get("epochs_trained", 0)
    epoch_cap = int(item["epoch_cap"])
    if epochs_used > epoch_cap:
        raise HTTPException(status_code=400, detail=(
            f"Your session has used {epochs_used} epochs; this assignment allows {epoch_cap}. "
            "Re-initialize the model and try again."
        ))

    # Server-authoritative scoring: evaluate the STORED network on the test set.
    metric_name, metric_value = _official_metric(session["network"], item["dataset"])
    higher_is_better = bool(item.get("higher_is_better", metric_name == "accuracy"))
    target_met = (
        metric_value >= float(item["metric_target"]) if higher_is_better
        else metric_value <= float(item["metric_target"])
    )

    student_name = request.student_name.strip()
    if not student_name or len(student_name) > 32 or not re.fullmatch(r"[a-zA-Z0-9_-]+", student_name):
        raise HTTPException(status_code=400, detail="Student name must be 1–32 characters: letters, digits, _ or -")

    now = int(time.time())
    submitted_item = {
        "student_name": student_name,
        "metric": Decimal(str(round(metric_value, 4))),
        "epochs_used": epochs_used,
        "target_met": target_met,
        "submitted_at": now,
    }

    # Atomic best-score upsert: overwrite only when strictly better.
    comparison = "<" if higher_is_better else ">"  # new must beat old in the useful direction
    condition = f"attribute_not_exists(metric) OR metric {comparison} :m"
    if SESSION_BACKEND == "local":
        board = _local_sessions.setdefault(f"submissions:{code}", {})
        prev = board.get(student_name)
        personal_best = prev is None or (
            metric_value > float(prev["metric"]) if higher_is_better
            else metric_value < float(prev["metric"])
        )
        if personal_best:
            board[student_name] = submitted_item
    else:
        submissions_table = _get_aws()["submissions"]
        try:
            submissions_table.update_item(
                Key={"assignment_code": code, "student_name": student_name},
                UpdateExpression="SET metric = :m, epochs_used = :e, target_met = :t, submitted_at = :ts",
                ConditionExpression=condition,
                ExpressionAttributeValues={
                    ":m": submitted_item["metric"],
                    ":e": epochs_used,
                    ":t": target_met,
                    ":ts": now,
                },
            )
            personal_best = True
        except submissions_table.meta.client.exceptions.ConditionalCheckFailedException:
            personal_best = False

    return AssignmentSubmitResponse(
        metric=round(metric_value, 4),
        metric_name=metric_name,
        epochs_used=epochs_used,
        target_met=target_met,
        personal_best=personal_best,
    )


@app.get("/assignments/{code}/submissions", response_model=List[SubmissionEntry])
def list_submissions(code: str, instructor_key: str):
    code = code.upper()
    item = _get_assignment(code)
    if item is None:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    if not hmac.compare_digest(str(item["instructor_key"]), instructor_key):
        raise HTTPException(status_code=403, detail="Invalid instructor key.")

    if SESSION_BACKEND == "local":
        entries = list(_local_sessions.get(f"submissions:{code}", {}).values())
    else:
        from boto3.dynamodb.conditions import Key as DynamoKey
        resp = _get_aws()["submissions"].query(
            KeyConditionExpression=DynamoKey("assignment_code").eq(code),
        )
        entries = resp.get("Items", [])
    entries.sort(key=lambda e: (-float(e["metric"])) if item.get("higher_is_better", True)
                 else float(e["metric"]))
    return [SubmissionEntry(**e) for e in entries]


# AWS Lambda entrypoint (API Gateway / ALB compatible)
lambda_handler = Mangum(app, lifespan="off")
