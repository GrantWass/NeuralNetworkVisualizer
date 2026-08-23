from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import re
from decimal import Decimal
# import math  # re-enable with /attention endpoint (used for sqrt(HEAD_DIM))
from NeuralNetwork import NeuralNetwork
from datasets import load_dataset  # Assume a function to load the dataset
import uuid
import pickle
import base64
import time
from collections import OrderedDict
import boto3
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Union
from mangum import Mangum

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow specific domains (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods like GET, POST, OPTIONS
    allow_headers=["*"],  # Allow all headers
)

_dynamodb = boto3.resource("dynamodb", region_name="us-east-2")
_sessions_table = _dynamodb.Table("nn-sessions")
_leaderboard_table = _dynamodb.Table("nn-leaderboard")
_s3 = boto3.client("s3", region_name="us-east-2")
SESSION_BUCKET = "nn-sessions-data"
SESSION_TTL_SECONDS = 3 * 60 * 60  # 3 hours

LEADERBOARD_CONFIG = {
    "xor":      {"higher_is_better": False, "display": "Fewest epochs to 100%",     "epoch_cap": None},
    "iris":     {"higher_is_better": True,  "display": "Accuracy at epoch 100 (%)", "epoch_cap": 100},
    "auto_mpg": {"higher_is_better": False, "display": "MAE at epoch 200",           "epoch_cap": 200},
    "mnist":    {"higher_is_better": True,  "display": "Accuracy at epoch 300 (%)", "epoch_cap": 300},
}

# In-process LRU cache (L1) — avoids a DynamoDB round-trip when the same
# Lambda instance handles back-to-back requests for the same session.
_CACHE_MAX = 32
_session_cache: OrderedDict = OrderedDict()


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


def _leaderboard_entries(raw_entries: list) -> List[LeaderboardEntry]:
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


def _save_session(session_id: str, session: dict) -> None:
    _cache_put(session_id, session)
    s3_key = f"sessions/{session_id}.pkl"
    _s3.put_object(Bucket=SESSION_BUCKET, Key=s3_key, Body=pickle.dumps(session))
    _sessions_table.put_item(Item={
        "session_id": session_id,
        "s3_key": s3_key,
        "ttl": int(time.time()) + SESSION_TTL_SECONDS,
    })


def _load_session(session_id: str) -> Optional[dict]:
    cached = _cache_get(session_id)
    if cached is not None:
        return cached
    resp = _sessions_table.get_item(Key={"session_id": session_id})
    item = resp.get("Item")
    if item is None:
        return None
    s3_key = item.get("s3_key")
    if s3_key is None:
        return None
    obj = _s3.get_object(Bucket=SESSION_BUCKET, Key=s3_key)
    session = pickle.loads(obj["Body"].read())
    _cache_put(session_id, session)
    return session


def _delete_session(session_id: str) -> None:
    _cache_evict(session_id)
    _sessions_table.delete_item(Key={"session_id": session_id})

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
    learning_rate: float = 0.01
    epochs: int = 10

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
    layers: List[LayerDetail]

class TrainResponse(BaseModel):
    training_results: List[TrainResult]

@app.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    session = _load_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /init_model first.")

    network = session["network"]
    X_train, _, Y_train, _, _, _, _, _, _, _ = load_dataset(session["dataset"])

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

        training_results.append(TrainResult(
            epoch=epoch + 1,
            input=X_train[:30].tolist() if is_last else [],
            loss=result["loss"],
            name=metric_name,
            metric=metric_value,
            layers=layers
        ))

    _save_session(request.session_id, session)
    return TrainResponse(training_results=training_results)


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
            activation=l.activation,
            Z=l.Z[:30].astype(np.float32).tolist(),
            A=l.A[:30].astype(np.float32).tolist(),
            dW=l.dW.astype(np.float32).tolist() if l.dW is not None else [],
            db=l.db.astype(np.float32).tolist() if l.db is not None else [],
            dZ=l.dZ[:30].astype(np.float32).tolist() if l.dZ is not None else [],
            weights=l.weights.astype(np.float32).tolist(),
            biases=l.biases.astype(np.float32).tolist(),
        ) for l in network.layers
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
@app.get("/leaderboard/{dataset}", response_model=LeaderboardResponse)
def get_leaderboard(dataset: str):
    if dataset not in LEADERBOARD_CONFIG:
        raise HTTPException(status_code=400, detail=f"Unknown dataset: {dataset}")
    cfg = LEADERBOARD_CONFIG[dataset]
    resp = _leaderboard_table.get_item(Key={"dataset": dataset})
    raw_entries = resp.get("Item", {}).get("entries", [])
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

    resp = _leaderboard_table.get_item(Key={"dataset": request.dataset})
    raw_entries = resp.get("Item", {}).get("entries", [])

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
    raw_entries = raw_entries[:10]

    _leaderboard_table.put_item(Item={
        "dataset": request.dataset,
        "entries": raw_entries,
        "updated_at": int(time.time()),
    })

    entries = _leaderboard_entries(raw_entries)
    return LeaderboardSubmitResponse(accepted=True, rank=insert_idx + 1, entries=entries)


# AWS Lambda entrypoint (API Gateway / ALB compatible)
lambda_handler = Mangum(app, lifespan="off")
