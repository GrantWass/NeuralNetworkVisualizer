import numpy as np
# import matplotlib.pyplot as plt


def activation_function(x, activation="sigmoid"):
    """
    Apply an activation function to the input.
    
    Args:
        x (numpy.ndarray): Input array.
        activation (str): Type of activation function ("relu", "sigmoid", "tanh", "softmax").
    
    Returns:
        numpy.ndarray: Activated output.
    """
    if activation == "sigmoid":
        return 1 / (1 + np.exp(-x))
    elif activation == "relu":
        return np.maximum(0, x)
    elif activation == "tanh":
        return np.tanh(x)
    elif activation == "softmax":
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    elif activation == "linear":
        return x
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def activation_derivative(Z, activation, Y=None):
    if activation == "sigmoid":
        sigmoid = 1 / (1 + np.exp(-Z))
        return sigmoid * (1 - sigmoid)
    elif activation == "relu":
        return np.where(Z > 0, 1, 0)
    elif activation == "tanh":
        return 1 - np.tanh(Z) ** 2
    elif activation == "softmax":
        if Y is not None:
            softmax = activation_function(Z, activation="softmax")
            return softmax - Y # Derivative of softmax with respect to cross-entropy loss
        raise ValueError("Softmax derivative requires Y (one-hot labels).")
    elif activation == "linear": 
        return np.ones_like(Z)  # Derivative of linear function is always 1
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def loss_function(predictions, targets, type= "cross-entropy"):
    """
    Calculate the loss between predictions and actual targets.
    
    Args:
        predictions (numpy.ndarray): Model predictions .
        targets (numpy.ndarray): Ground truth labels .
        type (string): Type of loss function.
    
    Returns:
        float: Loss value.
    """
    if type == "cross-entropy":
        epsilon = 1e-9
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        if predictions.shape[1] == 1:
            # Binary cross-entropy: both y·log(p) and (1-y)·log(1-p) terms
            return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        else:
            # Categorical cross-entropy with one-hot targets
            return -np.sum(targets * np.log(predictions)) / len(targets)
    elif type == "mse":
        errors = predictions - targets
        squared_errors = errors ** 2
        return np.mean(squared_errors)
    else:
        raise ValueError("Unsupported loss function type.")
    
    
def calculate_metric(predictions, targets, activation):
    """
    Calculate an appropriate evaluation metric based on the output activation.
    
    Args:
        predictions (numpy.ndarray): Predicted outputs.
        targets (numpy.ndarray): Actual target outputs.
        activation (str): Activation function of the output layer.
    
    Returns:
        float: Accuracy for classification, MAE for regression.
    """
    if activation == "softmax":  # Multi-class classification case
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(targets, axis=1)
        return np.mean(y_pred == y_true) * 100  # Accuracy in percentage
    
    elif activation == "sigmoid":  # Binary classification case
        y_pred = (predictions > 0.5).astype(int)
        y_true = targets.astype(int)
        return np.mean(y_pred == y_true) * 100  # Accuracy in percentage
    
    elif activation == "linear":  # Regression case
        return np.mean(np.abs(predictions - targets))  # Mean Absolute Error (MAE)
    
    else:
        raise ValueError(f"Unsupported output activation: {activation}")


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def generate_wt(input, output, activation="sigmoid"):
    if activation == "relu":
        return np.random.randn(input, output) * np.sqrt(2 / input)  # He initialization
    elif activation in ["linear", "tanh", "sigmoid", "softmax"]:
        return np.random.randn(input, output) * np.sqrt(1 / input)  # Xavier initialization
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

def train_test_split_np(X, y, test_size=0.2, random_state=42, shuffle=True):
    rng = np.random.default_rng(seed=random_state)
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        rng.shuffle(indices)
    split = int(n * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    if y is None:
        return X_train, X_test
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

class StandardScalerNP:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        # Prevent division by zero
        self.scale_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("StandardScalerNP must be fitted before transform.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    
# def plot_metrics(losses, accuracies, epochs):
#     plt.figure(figsize=(12, 5))

#     # Loss Plot
#     plt.subplot(1, 2, 1)
#     plt.plot(range(epochs), losses, label="Loss", color='red')
#     plt.xlabel("Epochs")
#     plt.ylabel("Loss")
#     plt.title("Training Loss")
#     plt.legend()

#     # Accuracy Plot
#     plt.subplot(1, 2, 2)
#     plt.plot(range(0, epochs, 2), accuracies, label="Accuracy", color='blue')
#     plt.xlabel("Epochs")
#     plt.ylabel("Accuracy (%)")
#     plt.title("Training Accuracy")
#     plt.legend()

#     plt.show()
