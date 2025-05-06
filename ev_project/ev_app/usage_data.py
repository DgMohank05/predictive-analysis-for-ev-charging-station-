# ev_app/usage_data.py
import pandas as pd

def load_usage_data(file_path="D:\3rd YEAR\24-25 EVEN SEM\TERM PAPER\ev_charging_patterns_final.csv"):
    """
    Load and prepare usage data.
    """
    # Load the data from CSV
    data = pd.read_csv(file_path, parse_dates=['date'])
    data.set_index('date', inplace=True)

    # Optionally resample to daily data (if your data isn't daily)
    data = data.resample('D').sum()

    return data
# ev_app/usage_data.py (continued)
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_usage_data(data, sequence_length=30):
    """
    Preprocess usage data for LSTM: normalize and create sequences.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)

    # Reshape X to be in the shape (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler
