import numpy as np


class HebbianEngine:
    """Production-ready Hebbian and Oja's Learning Engine using pure NumPy."""

    def __init__(self, input_dim: int, output_dim: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_dim, output_dim) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.weights)

    def train_hebbian(self, x: np.ndarray) -> np.ndarray:
        y = self.forward(x)
        delta_w = self.learning_rate * np.dot(x.T, y)
        self.weights += delta_w
        return self.weights

    def train_ojas_rule(self, x: np.ndarray) -> np.ndarray:
        y = self.forward(x)
        for i in range(x.shape[0]):
            x_i = x[i : i + 1]
            y_i = y[i : i + 1]
            delta_w = self.learning_rate * (
                np.dot(x_i.T, y_i) - np.dot(y_i, y_i.T) * self.weights
            )
            self.weights += delta_w
        return self.weights

    def recall(self, noisy_input: np.ndarray) -> np.ndarray:
        return self.forward(noisy_input)
    