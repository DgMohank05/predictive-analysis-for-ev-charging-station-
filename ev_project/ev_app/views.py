from django.http import JsonResponse
from .fetch_data import fetch_station_data
from .clustering import cluster_stations
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ---- Helper Function: Load and Scale Data ----
def load_and_scale_data(filepath):
    """
    Loads and scales the charging station usage data.
    """
    try:
        df = pd.read_csv(filepath).sort_values("ID")
        scaler = MinMaxScaler()
        df["scaled_usage"] = scaler.fit_transform(df["usage_count"].values.reshape(-1, 1))
        return df, scaler
    except Exception as e:
        raise Exception(f"Error loading or scaling data: {e}")

# ---- Helper Function: Create Sequences for LSTM ----
def create_sequences(data, seq_length=30):
    """
    Creates sequences of data for training the LSTM.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

# ---- LSTM Model Definition ----
class LSTMModel(nn.Module):
    def __init__(self):
        """
        Define the LSTM model architecture.
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ---- Helper Function: Train the LSTM Model ----
def train_lstm_model(df, seq_len=30, epochs=30, batch_size=16):
    """
    Trains the LSTM model using the provided data.
    """
    try:
        # Prepare data sequences
        X, y = create_sequences(df["scaled_usage"].values, seq_len)
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y)

        # Prepare DataLoader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, loss function, and optimizer
        model = LSTMModel()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                pred = model(xb).squeeze()
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save model after training
        torch.save(model.state_dict(), "model.pth")
        return model, loss_fn, optimizer
    except Exception as e:
        raise Exception(f"Error training the LSTM model: {e}")

# ---- Helper Function: Generate Predictions Using LSTM ----
def generate_predictions(model, scaler, seq_len=30, forecast_days=10):
    """
    Generates future predictions for the next 'forecast_days' using the trained model.
    """
    try:
        # Use the most recent data
        df = pd.read_csv(r"D:\3rd YEAR\24-25 EVEN SEM\TERM PAPER\ev_charging_patterns_final.csv")
        df = df.sort_values("ID")

        # Get the most recent sequence for prediction
        recent_seq = torch.FloatTensor(df["scaled_usage"].values[-seq_len:]).unsqueeze(0).unsqueeze(-1)

        # Predict the next 'forecast_days' values
        predictions = []
        for _ in range(forecast_days):
            next_val = model(recent_seq).item()
            predictions.append(next_val)
            # Update sequence for next prediction
            next_input = torch.cat((recent_seq.squeeze(0)[1:], torch.tensor([[next_val]])), 0)
            recent_seq = next_input.unsqueeze(0)

        # Inverse scale the predictions
        predicted_usage = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return predicted_usage.tolist()
    except Exception as e:
        raise Exception(f"Error generating predictions: {e}")

# ---- Django View: Predict Usage Only ----
def predict_usage(request):
    """
    Django view to predict usage for the next 10 days.
    """
    try:
        if not os.path.exists("model.pth"):
            # Load and scale the data
            df, scaler = load_and_scale_data(r"D:\3rd YEAR\24-25 EVEN SEM\TERM PAPER\ev_charging_patterns_final.csv")
            # Train the model (only if not already trained)
            model, _, _ = train_lstm_model(df)
        else:
            # Load the trained model
            model = LSTMModel()
            model.load_state_dict(torch.load("model.pth"))
            model.eval()

            # Load and scale the data
            df, scaler = load_and_scale_data(r"D:\3rd YEAR\24-25 EVEN SEM\TERM PAPER\ev_charging_patterns_final.csv")

        # Generate predictions
        predictions = generate_predictions(model, scaler)
        return JsonResponse({
            "predicted_usage_next_10_days": predictions
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

# ---- Django View: Cluster Stations and Annotate with Predicted Usage ----
def fetch_stations(request):
    """
    Django view to fetch and cluster stations, then annotate with predicted usage.
    """
    try:
        # Step 1: Fetch station data
        station_data = fetch_station_data(lat=17.385044, lon=78.486671, results=100)

        # Step 2: Cluster stations
        clustered_data = cluster_stations(station_data)

        # Step 3: Predict usage (single value used for all stations)
        if not os.path.exists("model.pth"):
            # Load and scale the data
            df, scaler = load_and_scale_data(r"D:\3rd YEAR\24-25 EVEN SEM\TERM PAPER\ev_charging_patterns_final.csv")
            # Train the model (only if not already trained)
            model, _, _ = train_lstm_model(df)
        else:
            # Load the trained model
            model = LSTMModel()
            model.load_state_dict(torch.load("model.pth"))
            model.eval()

            # Load and scale the data
            df, scaler = load_and_scale_data(r"D:\3rd YEAR\24-25 EVEN SEM\TERM PAPER\ev_charging_patterns_final.csv")

        # Get predicted usage for day 1
        predicted_usage = generate_predictions(model, scaler)
        clustered_data['predicted_usage_day1'] = predicted_usage[0] if predicted_usage else None

        # Step 4: Return JSON response
        return JsonResponse(clustered_data.to_dict(orient='records'), safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
