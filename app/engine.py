import os
import numpy as np


class HebbianEngine:
    """Production-ready Hebbian and Oja's Learning Engine with State Persistence."""

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

    def save_weights(self, filepath: str = "weights.npy") -> str:
        """Serializes current weight matrix to binary file."""
        np.save(filepath, self.weights)
        return filepath

    def load_weights(self, filepath: str = "weights.npy") -> str:
        """Deserializes weights from binary file into active memory."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight file '{filepath}' does not exist.")
        loaded_weights = np.load(filepath)
        if loaded_weights.shape != self.weights.shape:
            raise ValueError(
                f"Shape mismatch: file has {loaded_weights.shape}, expected {self.weights.shape}"
            )
        self.weights = loaded_weights
        return filepath
    @app.post("/save", status_code=status.HTTP_200_OK)
def save_model_state(filename: str = "model_weights.npy"):
    """Persists current network weights to disk."""
    try:
        saved_path = engine.save_weights(filepath=filename)
        return {
            "status": "success",
            "message": f"Weights saved successfully to {saved_path}",
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/load", status_code=status.HTTP_200_OK)
def load_model_state(filename: str = "model_weights.npy"):
    """Loads previously saved network weights into active memory."""
    try:
        engine.load_weights(filepath=filename)
        return {
            "status": "success",
            "message": f"Weights successfully loaded from {filename}",
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    