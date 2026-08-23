import numpy as np
import pytest

from utils import (
    StandardScalerNP,
    activation_derivative,
    activation_function,
    calculate_metric,
    loss_function,
    one_hot_encode,
    train_test_split_np,
)


class TestActivations:
    def test_sigmoid_range(self):
        out = activation_function(np.array([-100.0, 0.0, 100.0]), "sigmoid")
        assert np.all(out >= 0) and np.all(out <= 1)
        assert out[1] == pytest.approx(0.5)

    def test_relu(self):
        out = activation_function(np.array([-2.0, 0.0, 3.0]), "relu")
        assert np.array_equal(out, [0.0, 0.0, 3.0])

    def test_softmax_sums_to_one(self):
        out = activation_function(np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]]), "softmax")
        assert np.allclose(out.sum(axis=1), 1.0)
        assert np.all(np.isfinite(out))  # numerically stable for large inputs

    def test_linear_identity(self):
        x = np.array([[-1.5, 2.0]])
        assert np.array_equal(activation_function(x, "linear"), x)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            activation_function(np.zeros(3), "gelu")

    def test_derivatives(self):
        z = np.array([[-1.0, 0.0, 2.0]])
        assert np.allclose(activation_derivative(z, "relu"), [[0, 0, 1]])
        assert np.allclose(activation_derivative(z, "linear"), np.ones_like(z))
        # sigmoid'(z) = s(z)(1 - s(z)), max value 0.25 at z=0
        d = activation_derivative(z, "sigmoid")
        assert d[0, 1] == pytest.approx(0.25)


class TestLoss:
    def test_mse_zero_for_identical(self):
        y = np.array([[1.0], [2.0]])
        assert loss_function(y, y, "mse") == pytest.approx(0.0)

    def test_cross_entropy_zero_for_perfect(self):
        y = one_hot_encode(np.array([0, 1]), 2)
        p = np.array([[0.9999999, 0.0000001], [0.0000001, 0.9999999]])
        assert loss_function(p, y, "cross-entropy") == pytest.approx(0.0, abs=1e-4)

    def test_binary_vs_categorical_shape_branch(self):
        y = np.array([[1.0], [0.0]])
        p = np.array([[0.8], [0.2]])
        bce = loss_function(p, y, "cross-entropy")
        assert bce > 0

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            loss_function(np.ones((1, 1)), np.ones((1, 1)), "huber")


class TestMetrics:
    def test_softmax_accuracy(self):
        preds = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        targets = one_hot_encode(np.array([1, 0, 2]), 3)
        assert calculate_metric(preds, targets, "softmax") == pytest.approx(200 / 3)

    def test_linear_mae(self):
        preds = np.array([[1.0], [3.0]])
        targets = np.array([[2.0], [3.0]])
        assert calculate_metric(preds, targets, "linear") == pytest.approx(0.5)

    def test_sigmoid_accuracy(self):
        preds = np.array([[0.9], [0.1]])
        targets = np.array([[1.0], [0.0]])
        assert calculate_metric(preds, targets, "sigmoid") == 100.0


class TestHelpers:
    def test_train_test_split_reproducible(self):
        X = np.arange(20).reshape(10, 2)
        a = train_test_split_np(X, None, test_size=0.2, random_state=42)
        b = train_test_split_np(X, None, test_size=0.2, random_state=42)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        assert len(a[0]) == 8 and len(a[1]) == 2

    def test_scaler_constant_column(self):
        scaler = StandardScalerNP().fit(np.array([[1.0, 5.0], [3.0, 5.0]]))
        out = scaler.transform(np.array([[1.0, 5.0]]))
        assert np.isfinite(out).all()  # zero-variance column doesn't divide by zero

    def test_one_hot(self):
        assert np.array_equal(one_hot_encode(np.array([0, 2]), 3),
                              np.array([[1, 0, 0], [0, 0, 1]]))


def test_numerical_gradient_matches_backward():
    """Analytical gradients from backward() should match numerical gradients."""
    from NeuralNetwork import NeuralNetwork

    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    Y = (rng.normal(size=(5, 1)) > 0).astype(np.float32)

    nn = NeuralNetwork([3, 4, 1], ["tanh", "sigmoid"], "batch")

    nn.train_step(X, Y, learning_rate=0.0)  # compute grads without updating
    analytic = nn.layers[0].dW.copy()

    eps = 1e-5
    W0 = nn.layers[0].weights.copy()
    numeric = np.zeros_like(W0)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            nn.layers[0].weights[i, j] = W0[i, j] + eps
            loss_plus = loss_function(nn.forward(X), Y, "cross-entropy")
            nn.layers[0].weights[i, j] = W0[i, j] - eps
            loss_minus = loss_function(nn.forward(X), Y, "cross-entropy")
            nn.layers[0].weights[i, j] = W0[i, j]
            numeric[i, j] = (loss_plus - loss_minus) / (2 * eps)

    assert np.allclose(analytic, numeric, atol=1e-4)


def test_multi_output_mse_gradient_matches_numerical():
    """Regression guard: MSE is averaged over m×k elements, so gradients for
    k>1 outputs must not be scaled by k (this bug shipped once)."""
    from NeuralNetwork import NeuralNetwork

    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 3))
    Y = rng.normal(size=(6, 3))
    nn = NeuralNetwork([3, 5, 3], ["tanh", "linear"], "batch")
    for layer in nn.layers:
        layer.weights = rng.normal(size=layer.weights.shape)

    nn.train_step(X, Y, learning_rate=0.0)
    analytic = nn.layers[0].dW.copy()

    eps = 1e-6
    W0 = nn.layers[0].weights.copy()
    numeric = np.zeros_like(W0)
    it = np.nditer(W0, flags=["multi_index"])
    while not it.finished:
        ix = it.multi_index
        nn.layers[0].weights[ix] = W0[ix] + eps
        lp = loss_function(nn.forward(X), Y, "mse")
        nn.layers[0].weights[ix] = W0[ix] - eps
        lm = loss_function(nn.forward(X), Y, "mse")
        nn.layers[0].weights[ix] = W0[ix]
        numeric[ix] = (lp - lm) / (2 * eps)
        it.iternext()

    rel = np.max(np.abs(analytic - numeric)) / (np.max(np.abs(numeric)) + 1e-12)
    assert rel < 1e-6


def test_mse_with_softmax_output_is_rejected():
    """Softmax backward passes dA through unchanged (valid only for the fused
    softmax+cross-entropy gradient); combining it with MSE would silently
    produce wrong gradients."""
    from NeuralNetwork import NeuralNetwork

    nn = NeuralNetwork([2, 3], ["relu", "softmax"], "batch")
    X = np.random.default_rng(0).normal(size=(4, 2))
    Y = np.eye(3)[np.random.default_rng(1).integers(0, 3, size=4)]
    with pytest.raises(ValueError):
        nn.train_step(X, Y, learning_rate=0.1, loss_type="mse")
