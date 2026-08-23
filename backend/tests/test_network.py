import numpy as np
import pytest

from datasets import load_xor_dataset
from NeuralNetwork import NeuralNetwork


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        NeuralNetwork([2, 3, 1], ["sigmoid", "sigmoid"], "rmsprop")


def test_forward_shapes():
    nn = NeuralNetwork([4, 5, 2], ["relu", "softmax"], "batch")
    out = nn.forward(np.zeros((7, 4)))
    assert out.shape == (7, 2)


def test_xor_converges_with_batch():
    X_train, _, Y_train, _, *_ = load_xor_dataset()
    np.random.seed(0)
    nn = NeuralNetwork([2, 8, 8, 1], ["relu", "relu", "sigmoid"], "batch")
    for _ in range(500):
        result = nn.train_step(X_train, Y_train, learning_rate=0.5)
        if result["accuracy"] == 100.0:
            break
    preds = (nn.forward(X_train) > 0.5).astype(int)
    assert np.array_equal(preds, Y_train.astype(int))


def test_adam_state_roundtrip_preserves_predictions():
    X_train, _, Y_train, _, *_ = load_xor_dataset()
    np.random.seed(1)
    nn = NeuralNetwork([2, 8, 1], ["tanh", "sigmoid"], "adam")
    for _ in range(50):
        nn.train_step(X_train, Y_train, learning_rate=0.1)

    state = nn.to_state()
    assert isinstance(state, dict)

    restored = NeuralNetwork.from_state(state)
    assert np.allclose(nn.forward(X_train), restored.forward(X_train))

    # Adam moments survive serialization: continued training matches a non-serialized net
    for _ in range(50):
        nn.train_step(X_train, Y_train, learning_rate=0.1)
        restored.train_step(X_train, Y_train, learning_rate=0.1)
    assert np.allclose(nn.forward(X_train), restored.forward(X_train))


def test_batch_optimizer_has_no_state():
    nn = NeuralNetwork([2, 3, 1], ["sigmoid", "sigmoid"], "batch")
    assert nn.optimizer.state_dict(nn.layers) is None


def test_from_state_rejects_missing_keys():
    with pytest.raises(KeyError):
        NeuralNetwork.from_state({"optimizer": "batch"})
