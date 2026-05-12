import numpy as np
from utils import generate_wt, activation_function, activation_derivative

class NeuronLayer:
    def __init__(self, input_size, output_size, activation="sigmoid"):
        # Initialize the layer with input size, output size, and activation function
        self.input_size = input_size  # Number of input features
        self.output_size = output_size  # Number of neurons in this layer
        self.activation = activation  # Activation function to be used (e.g., sigmoid, softmax)
        
        # Initialize weights and biases
        self.weights = generate_wt(input_size, output_size, activation)  # Weight matrix
        self.biases = np.zeros((1, output_size))  # Bias vector
        
        # Placeholders for forward and backward pass values
        self.Z = None  # Linear output (W.X + b)
        self.A = None  # Activated output (activation(Z))
        self.dW = None  # Gradient of weights
        self.db = None  # Gradient of biases
        self.dZ = None  # Gradient of Z
        self.dA = None 

    def forward(self, X):
        # Perform the forward pass: compute linear output Z and activated output A
        self.Z = np.dot(X, self.weights) + self.biases  # Linear transformation
        self.A = activation_function(self.Z, self.activation)  # Apply activation function
        return self.A  # Return the activated output

    def backward(self, dA, prev_A):
        # Perform the backward pass: compute gradients of weights, biases, and input

        # Compute dZ depending on activation
        if self.activation == "softmax":
            self.dZ = dA  # Already handled by loss function
        else:
            self.dZ = dA * activation_derivative(self.Z, self.activation)
        
        # Compute gradients of weights and biases
        m = prev_A.shape[0]  # Number of samples in the batch
        self.dW = np.dot(prev_A.T, self.dZ) / m  # Gradient of weights
        self.db = np.sum(self.dZ, axis=0, keepdims=True) / m  # Gradient of biases
        
        # Compute the gradient of the input to this layer (to be passed to the previous layer)
        dA_prev = np.dot(self.dZ, self.weights.T)
        return dA_prev
    
    def to_dict(self):
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "activation": self.activation,
            "weights": self.weights.tolist(),
            "biases": self.biases.tolist(),
            "A": self.A.tolist() if self.A is not None else "",
            "Z": self.Z.tolist() if self.Z is not None else "",
            "dW": self.dW.tolist() if self.dW is not None else "",
            "db": self.db.tolist() if self.db is not None else "",
            "dZ": self.dZ.tolist() if self.dZ is not None else "",
            "dA": self.dA.tolist() if self.dA is not None else "",
        }