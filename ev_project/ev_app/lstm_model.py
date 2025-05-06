import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load your usage data
df = pd.read_csv("usage_data.csv", parse_dates=["date"])
df = df.sort_values("date")

# Normalize usage
scaler = MinMaxScaler()
df["scaled_usage"] = scaler.fit_transform(df["usage_count"].values.reshape(-1, 1))

# Prepare sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_len = 30
X, y = create_sequences(df["scaled_usage"].values, seq_len)

X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # (batch, seq_len, 1)
y_tensor = torch.FloatTensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])  # last timestep
        return output

model = LSTMModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 50
for epoch in range(epochs):
    for xb, yb in loader:
        pred = model(xb).squeeze()
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

# Forecast next N days
model.eval()
with torch.no_grad():
    recent_seq = torch.FloatTensor(df["scaled_usage"].values[-seq_len:]).unsqueeze(0).unsqueeze(-1)
    predictions = []
    for _ in range(10):  # Predict next 10 days
        next_val = model(recent_seq).item()
        predictions.append(next_val)
        next_input = torch.cat((recent_seq.squeeze(0)[1:], torch.tensor([[next_val]])), 0)
        recent_seq = next_input.unsqueeze(0)

# Inverse scale and show predictions
predicted_usage = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
print("Predicted next 10 days usage:\n", predicted_usage.flatten())
