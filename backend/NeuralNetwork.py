import numpy as np

from NeuronLayer import NeuronLayer
from utils import calculate_metric, loss_function

OPTIMIZERS = {
    "batch": "Batch",
    "adam": "Adam",
}

class NeuralNetwork:
    def __init__(self, layer_sizes, activations, optimizer="batch"):
        if optimizer not in OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {optimizer}. Choose from {list(OPTIMIZERS)}")
        self.optimizer_name = optimizer
        # Initialize the neural network with a list of layer sizes and activation functions
        self.layers = []
        module = __import__("optimizer", fromlist=[OPTIMIZERS[optimizer]])
        self.optimizer = getattr(module, OPTIMIZERS[optimizer])()

        for i in range(len(layer_sizes) - 1):
            # Create each layer with the appropriate input and output sizes
            self.layers.append(NeuronLayer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    def forward(self, X):
        A = X  # Start with the input data
        for layer in self.layers:
            A = layer.forward(A)  # Pass the output of one layer as input to the next
        return A  # Return the final output of the network

    def backward(self, X, Y, Y_hat, loss_type="mse"):
        # Perform the backward pass through all layers

        # Compute the initial gradient based on loss type
        if loss_type == "mse":
            # Loss is averaged over all m×k elements; layer.backward() divides
            # by m, so fold the remaining 1/k (output dim) in here. For k=1
            # this is identical to the previous behaviour.
            dA = 2 * (Y_hat - Y) / Y.shape[1]
        elif loss_type == "cross-entropy":
            if self.layers[-1].activation == "softmax":
                # Combined dL/dZ for softmax+cross-entropy; NeuronLayer passes it through unchanged
                dA = Y_hat - Y
            else:
                # True dL/dA for sigmoid+cross-entropy so NeuronLayer can chain-rule correctly:
                # dZ = dA * sigmoid'(Z) = [-(Y/A - (1-Y)/(1-A))] * A*(1-A) = A - Y
                epsilon = 1e-9
                clipped = np.clip(Y_hat, epsilon, 1 - epsilon)
                dA = -(Y / clipped - (1 - Y) / (1 - clipped))
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        # Iterate through layers in reverse order
        for i in reversed(range(len(self.layers))):
            prev_A = X if i == 0 else self.layers[i-1].A  # Input to the current layer

            # Compute gradients for the current layer
            dA = self.layers[i].backward(dA, prev_A)

    def update_parameters(self, learning_rate):
        # Update the weights and biases of each layer using gradient descent
        for layer in self.layers:
            self.optimizer.step(layer, learning_rate)


    def train_step(self, X, Y, learning_rate, loss_type=None):
        # Determine loss type based on the output activation
        if loss_type is None:
            loss_type = "mse" if self.layers[-1].activation == "linear" else "cross-entropy"
        elif loss_type == "mse" and self.layers[-1].activation == "softmax":
            raise ValueError(
                "MSE with a softmax output is not supported: softmax backward "
                "assumes the fused cross-entropy gradient."
            )

        Y_hat = self.forward(X)
        loss = loss_function(Y_hat, Y, loss_type)
        self.backward(X, Y, Y_hat, loss_type)
        self.update_parameters(learning_rate)

        metric_value = calculate_metric(Y_hat, Y, self.layers[-1].activation)
        metric_name = "Accuracy" if self.layers[-1].activation in ["softmax", "sigmoid"] else "MAE"

        layer_details = []
        for layer in self.layers:
            layer_details.append({
                "weights": layer.weights,
                "biases": layer.biases,
                "Z": layer.Z,
                "A": layer.A,
                "dW": layer.dW,
                "db": layer.db,
                "dZ": layer.dZ,
                "activation": layer.activation
            })

        return {
            "loss": loss,
            metric_name.lower(): metric_value,
            "layers": layer_details
        }

    def to_dict(self):
        return {
            "layers": [layer.to_dict() for layer in self.layers]
        }

    def to_state(self):
        # JSON-serializable state for persistence (weights, biases, optimizer state)
        return {
            "optimizer": self.optimizer_name,
            "layers": [
                {
                    "input_size": layer.input_size,
                    "output_size": layer.output_size,
                    "activation": layer.activation,
                    "weights": layer.weights.tolist(),
                    "biases": layer.biases.tolist(),
                }
                for layer in self.layers
            ],
            "optimizer_state": self.optimizer.state_dict(self.layers),
        }

    @classmethod
    def from_state(cls, state):
        layer_sizes = [state["layers"][0]["input_size"]] + [ls["output_size"] for ls in state["layers"]]
        activations = [ls["activation"] for ls in state["layers"]]
        nn = cls(layer_sizes, activations, state["optimizer"])
        for layer, layer_state in zip(nn.layers, state["layers"]):
            layer.weights = np.array(layer_state["weights"], dtype=layer.weights.dtype)
            layer.biases = np.array(layer_state["biases"], dtype=layer.biases.dtype)
        if state.get("optimizer_state") is not None:
            nn.optimizer.load_state_dict(nn.layers, state["optimizer_state"])
        return nn

