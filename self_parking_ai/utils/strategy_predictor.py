# utils/strategy_predictor.py

import os
import numpy as np
import torch
import torch.nn as nn
import warnings
try:
    # scikit-learn may warn if the saved scaler was from a different minor version
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.simplefilter("ignore", InconsistentVersionWarning)
except Exception:
    pass

from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# ---------- Model definition (MUST match training) ----------
# If you trained with this exact architecture, keep it.
# If you trained a custom MLP class, replace this with that class definition.
model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

# ---------- Resolve paths relative to this file ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # -> .../self_parking_ai
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mlp_strategy.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


# ---------- Load weights (CPU-safe) ----------
try:
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
except FileNotFoundError as e:
    raise FileNotFoundError(
        f"Could not find model weights at {MODEL_PATH}. "
        "Ensure mlp_strategy.pt is in self_parking_ai/models/."
    ) from e

# ---------- Optional: load scaler ----------
try:
    import joblib
    scaler = joblib.load(SCALER_PATH)
except Exception:
    scaler = None
    print("⚠️  No scaler loaded; continuing without feature scaling.")

# ---------- Inference helper ----------
def predict_strategy(car_L, car_W, spot_L, spot_W, dist, verbose: bool = False):
    """
    Inputs: floats (meters)
    Returns:
      strategy: str
      confidence: float in [0,1]
      scores: dict[str -> float] of class probabilities
    """
    x = np.array([[car_L, car_W, spot_L, spot_W, dist]], dtype=float)
    if scaler is not None:
        x = scaler.transform(x)

    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()[0]

    classes = ["Forward Perpendicular", "Reverse Parallel", "Cannot Park"]
    idx = int(probs.argmax())
    if verbose:
        print("probs:", dict(zip(classes, map(float, probs))))
    return classes[idx], float(probs[idx]), dict(zip(classes, map(float, probs)))
