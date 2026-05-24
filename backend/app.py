from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import re
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


# ── Leaderboard helpers ───────────────────────────────────────────────────────

def _is_better(dataset: str, new_score: float, existing_score: float) -> bool:
    return new_score > existing_score if LEADERBOARD_CONFIG[dataset]["higher_is_better"] else new_score < existing_score


def _qualifies_for_top10(dataset: str, score: float, entries: list) -> Optional[int]:
    for i, entry in enumerate(entries):
        if _is_better(dataset, score, entry["score"]):
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
    data = base64.b64encode(pickle.dumps(session)).decode("utf-8")
    _sessions_table.put_item(Item={
        "session_id": session_id,
        "data": data,
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
    session = pickle.loads(base64.b64decode(item["data"]))
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

    # Store the model and dataset in the user's session
    _save_session(session_id, {
        "network": network,
        "X_train": X_train,
        "Y_train": Y_train,
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
    X_train = session["X_train"]
    Y_train = session["Y_train"]

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
    X_train = session["X_train"]

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
    entries = [
        LeaderboardEntry(rank=i + 1, **e)
        for i, e in enumerate(raw_entries)
    ]
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

    username = request.username.strip()
    if not username or len(username) > 32 or not re.fullmatch(r"[a-zA-Z0-9_-]+", username):
        raise HTTPException(status_code=400, detail="Username must be 1–32 characters: letters, digits, _ or -")

    resp = _leaderboard_table.get_item(Key={"dataset": request.dataset})
    raw_entries = resp.get("Item", {}).get("entries", [])

    insert_idx = _qualifies_for_top10(request.dataset, request.score, raw_entries)
    if insert_idx is None:
        entries = [LeaderboardEntry(rank=i + 1, **e) for i, e in enumerate(raw_entries)]
        return LeaderboardSubmitResponse(accepted=False, rank=None, entries=entries)

    new_entry = {
        "username": username,
        "score": request.score,
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

    entries = [LeaderboardEntry(rank=i + 1, **e) for i, e in enumerate(raw_entries)]
    return LeaderboardSubmitResponse(accepted=True, rank=insert_idx + 1, entries=entries)


# AWS Lambda entrypoint (API Gateway / ALB compatible)
lambda_handler = Mangum(app)
