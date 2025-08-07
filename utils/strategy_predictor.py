import torch
import numpy as np
import joblib

# Define the model structure (same as in training)
model = torch.nn.Sequential(
    torch.nn.Linear(5, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 3)
)

# Load trained weights
model.load_state_dict(torch.load('mlp_strategy.pt'))
model.eval()

# Try loading the scaler
try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    scaler = None
    print("⚠️ No scaler found — using raw inputs")

# Clean, reusable prediction function
def predict_strategy(car_length, car_width, spot_length, spot_width, distance, verbose=False):
    """
    Predict parking strategy based on car and spot dimensions.

    Inputs:
        - car_length, car_width, spot_length, spot_width, distance: float
        - verbose: bool (print confidence scores)

    Returns:
        - recommendation (str): best strategy label
        - confidence (float): probability score of top strategy
        - all_probs (dict): confidence for all strategy classes
    """

    sample_input = np.array([[car_length, car_width, spot_length, spot_width, distance]])

    if scaler:
        sample_input = scaler.transform(sample_input)

    input_tensor = torch.tensor(sample_input, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        predicted_class = torch.argmax(output, dim=1).item()

    strategy_labels = ['Reverse Parallel', 'Forward Perpendicular', 'Cannot Park']
    recommendation = strategy_labels[predicted_class]
    confidence = probabilities[predicted_class]
    all_probs = {label: float(f"{prob:.4f}") for label, prob in zip(strategy_labels, probabilities)}

    if verbose:
        print("📊 Strategy Confidence Scores:")
        for label, prob in all_probs.items():
            print(f"{label}: {prob*100:.2f}%")
        print(f"\n🚘 Recommended Strategy: {recommendation} ({confidence:.2%} confidence)")

    return recommendation, confidence, all_probs
