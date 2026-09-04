import numpy as np
from app.engine import HebbianEngine

# 1. Initialize the engine with 4 inputs and 2 outputs
engine = HebbianEngine(input_dim=4, output_dim=2, learning_rate=0.1)

# 2. Create sample input data (2 data points, 4 features each)
sample_data = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0]
])

# 3. Pass the data through Oja's training rule
updated_weights = engine.train_ojas_rule(sample_data)

# 4. Print results to confirm it works
print("Engine initialized successfully!")
print("Weights Matrix Shape:", updated_weights.shape)