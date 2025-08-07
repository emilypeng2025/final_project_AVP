# train_mlp_classifier.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load or generate sample data
# Format: [car_length, car_width, spot_length, spot_width, distance_to_spot], label
data = pd.read_csv("data/parking_data.csv")  # or generate manually
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 2. Normalize input
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, "models/scaler.pkl")

# 3. Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# 4. Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

# 5. Define model
model = nn.Sequential(
    nn.Linear(5, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

# 6. Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 7. Save model
torch.save(model.state_dict(), "models/mlp_strategy.pt")
print("✅ Model saved to models/mlp_strategy.pt")