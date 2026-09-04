from typing import List
from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

from app.engine import HebbianEngine

app = FastAPI(
    title="HebbiDB Neural Memory API",
    description="A high-performance backend memory engine driven by Hebbian Learning.",
    version="0.1.0",
)

# Global memory engine instance (Initialized with 4 inputs, 2 outputs)
engine = HebbianEngine(input_dim=4, output_dim=2, learning_rate=0.01)


# Request schema validation
class TrainRequest(BaseModel):
    data: List[List[float]]
    use_ojas: bool = True


class RecallRequest(BaseModel):
    pattern: List[float]


@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "online", "engine": "HebbiDB Core v0.1.0"}


@app.post("/train")
def train_network(payload: TrainRequest):
    """Train the Hebbian engine on an incoming batch of matrix data."""
    try:
        matrix_input = np.array(payload.data)

        if matrix_input.shape[1] != engine.input_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Expected input dimension of {engine.input_dim}, got {matrix_input.shape[1]}",
            )

        if payload.use_ojas:
            updated_weights = engine.train_ojas_rule(matrix_input)
        else:
            updated_weights = engine.train_hebbian(matrix_input)

        return {
            "message": "Weights successfully updated",
            "weights": updated_weights.tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recall")
def recall_pattern(payload: RecallRequest):
    """Recall or extract feature activations from a given input vector."""
    try:
        vector_input = np.array([payload.pattern])

        if vector_input.shape[1] != engine.input_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Expected pattern length of {engine.input_dim}, got {vector_input.shape[1]}",
            )

        activation = engine.recall(vector_input)
        return {"output_activation": activation.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))