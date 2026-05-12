import numpy as np

class Optimizer:
    def step(self, layer, learning_rate):
        raise NotImplementedError("Must be implemented by subclass")


class Batch(Optimizer):

    def step(self, layer, learning_rate):
        layer.weights -= learning_rate * layer.dW
        layer.biases -= learning_rate * layer.db

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layer, learning_rate):
        if layer not in self.m:
            self.m[layer] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}
            self.v[layer] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}

        self.t += 1
        m, v = self.m[layer], self.v[layer]

        # Update biased moments
        m['w'] = self.beta1 * m['w'] + (1 - self.beta1) * layer.dW
        m['b'] = self.beta1 * m['b'] + (1 - self.beta1) * layer.db
        v['w'] = self.beta2 * v['w'] + (1 - self.beta2) * (layer.dW ** 2)
        v['b'] = self.beta2 * v['b'] + (1 - self.beta2) * (layer.db ** 2)

        # Correct bias
        m_hat_w = m['w'] / (1 - self.beta1 ** self.t)
        m_hat_b = m['b'] / (1 - self.beta1 ** self.t)
        v_hat_w = v['w'] / (1 - self.beta2 ** self.t)
        v_hat_b = v['b'] / (1 - self.beta2 ** self.t)

        # Update weights
        layer.weights -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        layer.biases -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

class RMSprop(Optimizer):
    def __init__(self, beta=0.9, epsilon=1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.v = {}

    def step(self, layer, learning_rate):
        if layer not in self.v:
            self.v[layer] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}

        self.v[layer]['w'] = self.beta * self.v[layer]['w'] + (1 - self.beta) * (layer.dW ** 2)
        self.v[layer]['b'] = self.beta * self.v[layer]['b'] + (1 - self.beta) * (layer.db ** 2)

        layer.weights -= learning_rate * layer.dW / (np.sqrt(self.v[layer]['w']) + self.epsilon)
        layer.biases -= learning_rate * layer.db / (np.sqrt(self.v[layer]['b']) + self.epsilon)


class Adagrad(Optimizer):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.g = {}

    def step(self, layer, learning_rate):
        if layer not in self.g:
            self.g[layer] = {'w': np.zeros_like(layer.weights), 'b': np.zeros_like(layer.biases)}

        self.g[layer]['w'] += layer.dW ** 2
        self.g[layer]['b'] += layer.db ** 2

        layer.weights -= learning_rate * layer.dW / (np.sqrt(self.g[layer]['w']) + self.epsilon)
        layer.biases -= learning_rate * layer.db / (np.sqrt(self.g[layer]['b']) + self.epsilon)
