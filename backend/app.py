from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import math
from NeuralNetwork import NeuralNetwork
from datasets import load_dataset  # Assume a function to load the dataset
import uuid
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Union
from mangum import Mangum

NUM_HEADS = 12
HEAD_DIM = 64

# ── BERT model — lazy-loaded on first /attention call ─────────────────────────
# Loaded eagerly at module level caused every cold start (including /init_model,
# /train, etc.) to block for 5–15 s while BERT loaded, exceeding API Gateway's
# 29-second integration timeout and returning 503s for unrelated endpoints.
# Lazy-loading means non-BERT endpoints respond immediately on cold start.
# Warm /attention calls are unaffected (<1 s) since the module is already loaded.
_tokenizer = None
_model = None
_model_error: Optional[str] = None
_bert_loaded = False

def _load_bert():
    global _tokenizer, _model, _model_error, _bert_loaded
    if _bert_loaded:
        return
    _bert_loaded = True
    try:
        from transformers import BertTokenizer, BertModel  # type: ignore
        import torch as _torch  # type: ignore
        _tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        _model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
        _model.eval()
    except Exception as _e:
        _model_error = str(_e)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow specific domains (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods like GET, POST, OPTIONS
    allow_headers=["*"],  # Allow all headers
)
# NOTE: in-memory store — concurrent Lambda instances won't share state and
# sessions are lost on cold starts. To fix at scale, back this with DynamoDB:
# table with session_id partition key, TTL on a `ttl` attribute, serialize
# session dicts with pickle+base64. Free tier covers typical demo traffic
# (~1M req/month); on-demand beyond that runs ~$1.50 per million interactions.
user_sessions = {}

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

@app.post("/init_model", response_model=InitModelResponse)
def init_model(request: InitModelRequest):
    session_id = str(uuid.uuid4())  # Generate a unique session ID

    # Load dataset
    X_train, _, Y_train, _, input_size, output_size, output_activation, original_train_data = load_dataset(request.dataset)

    layers = [input_size] + request.layer_sizes + [output_size]

    # Ensure activations length matches hidden + output layers
    if len(request.activations) != len(request.layer_sizes):
        raise HTTPException(status_code=400, detail="Activations length must match number of layers.")

    activations = request.activations + [output_activation]
    network = NeuralNetwork(layers, activations, optimizer="batch")

    # Store the model and dataset in the user's session
    user_sessions[session_id] = {
        "network": network,
        "X_train": X_train,
        "Y_train": Y_train,
    }

    return InitModelResponse(
        message="Model initialized successfully",
        session_id=session_id,
        layer_sizes=layers,
        original_train_data=original_train_data[:30],
        network=network.to_dict()  # Return serialized network
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
    session = user_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /init_model first.")

    network = session["network"]
    X_train = session["X_train"]
    Y_train = session["Y_train"]

    training_results = []

    for epoch in range(request.epochs):
        result = network.train_step(X_train, Y_train, request.learning_rate)

        # Full detail for the last epoch
        if epoch == request.epochs - 1:
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
        else:
            # Only return weights & biases for earlier epochs
            layers = [
                LayerDetail(
                    weights=layer["weights"].astype(np.float32).tolist(),
                    biases=layer["biases"].astype(np.float32).tolist(),
                    Z=[],
                    A=[],
                    dW=[],
                    db=[],
                    dZ=[],
                    activation=layer["activation"]
                ) for layer in result["layers"]
            ]

        metric_name = "accuracy" if "accuracy" in result else "mae"
        metric_value = result.get("accuracy") if "accuracy" in result else result.get("mae")

        training_results.append(TrainResult(
            epoch=epoch + 1,
            input=X_train[:30].tolist() if epoch == request.epochs - 1 else [],
            loss=result["loss"],
            name=metric_name,
            metric=metric_value,
            layers=layers
        ))

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
    session = user_sessions.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /init_model first.")

    network = session["network"]
    X_train = session["X_train"]

    layer = network.layers[request.layer_index]
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

    return SetWeightResponse(layers=layers)


# ------------------ Clear Session ------------------ #
@app.post("/clear_session")
def clear_session(session_id: str):
    if session_id in user_sessions:
        del user_sessions[session_id]
        return {"message": "Session cleared successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found.")


# ── Attention endpoint ────────────────────────────────────────────────────────

class AttentionRequest(BaseModel):
    sentence: str
    layer: int = 6

class AttentionResponse(BaseModel):
    tokens: List[str]
    attentionMatrix: List[List[float]]
    multiHeadAttention: List[List[List[float]]]
    headLayer: int
    headIndices: List[int]
    rawScoresMatrix: List[List[float]]             # pre-softmax Q·K/√d averaged across all heads
    multiHeadRawScores: List[List[List[float]]]    # pre-softmax scores for each of the 4 displayed heads
    queryVectors: List[List[List[float]]]          # 4 heads × seq × 64
    keyVectors: List[List[List[float]]]            # 4 heads × seq × 64

# Smoke-test curl:
#   curl -X POST http://localhost:8000/attention \
#     -H "Content-Type: application/json" \
#     -d '{"sentence": "The cat sat on the mat"}'

@app.post("/attention", response_model=AttentionResponse)
def get_attention(request: AttentionRequest):
    """
    Return BERT self-attention weights for a given sentence, including pre-softmax
    Q·K/√d scores and the raw Q/K projection vectors for the 4 displayed heads.

    First call: 5–15 s (BERT loads from disk).  Subsequent calls: <1 s.
    Requires Lambda memory ≥ 2048 MB.
    """
    _load_bert()
    import torch  # type: ignore  # available after _load_bert()

    if _model_error:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {_model_error}")

    sentence = request.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="sentence must not be empty")

    inputs = _tokenizer(sentence, return_tensors="pt")
    token_count = inputs["input_ids"].shape[1]
    if token_count > 100:
        raise HTTPException(
            status_code=400,
            detail=f"Sentence is too long ({token_count} tokens). Maximum is 100."
        )

    tokens = _tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    layer = max(0, min(request.layer, 11))

    # Register forward hooks to capture Q and K projections from the target layer.
    # Each hook receives the linear layer output (batch, seq, hidden=768), reshapes to
    # (batch, seq, num_heads, head_dim) and permutes to (num_heads, seq, head_dim).
    store: dict = {}

    def q_hook(_, __, output):
        b, s, _ = output.shape
        store["q"] = output.detach().view(b, s, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)[0]

    def k_hook(_, __, output):
        b, s, _ = output.shape
        store["k"] = output.detach().view(b, s, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)[0]

    target_self = _model.encoder.layer[layer].attention.self
    q_handle = target_self.query.register_forward_hook(q_hook)
    k_handle = target_self.key.register_forward_hook(k_hook)

    try:
        with torch.no_grad():
            outputs = _model(**inputs)
    finally:
        q_handle.remove()
        k_handle.remove()

    # outputs.attentions: tuple[num_layers] of (batch=1, num_heads=12, seq, seq)
    attentions = outputs.attentions[layer][0]  # (12, seq, seq)
    main_attention = attentions.mean(dim=0).tolist()

    head_indices = [0, 3, 7, 11]
    multi_head = [attentions[h].tolist() for h in head_indices]

    # Pre-softmax scores: Q·Kᵀ / √d  →  (num_heads, seq, seq)
    q = store["q"]  # (12, seq, 64)
    k = store["k"]  # (12, seq, 64)
    raw_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(HEAD_DIM)  # (12, seq, seq)

    raw_scores_avg = raw_scores.mean(dim=0).tolist()
    multi_head_raw_scores = [raw_scores[h].tolist() for h in head_indices]

    query_vectors = [q[h].tolist() for h in head_indices]
    key_vectors = [k[h].tolist() for h in head_indices]

    return AttentionResponse(
        tokens=tokens,
        attentionMatrix=main_attention,
        multiHeadAttention=multi_head,
        headLayer=layer,
        headIndices=head_indices,
        rawScoresMatrix=raw_scores_avg,
        multiHeadRawScores=multi_head_raw_scores,
        queryVectors=query_vectors,
        keyVectors=key_vectors,
    )


# AWS Lambda entrypoint (API Gateway / ALB compatible)
lambda_handler = Mangum(app)
