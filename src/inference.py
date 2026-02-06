import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import time
import os

# --- 1. DYNAMIC PATH CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

MODEL_PATH = os.path.join(MODELS_DIR, 'fraud_detection_model.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Threshold derived from training loss analysis (update this based on your notebook plots)
THRESHOLD = 0.1949


class FraudDetector:
    def __init__(self):
        print("\n--- Initializing Blockchain Fraud System ---")

        # Check Files
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model missing at: {MODEL_PATH}\n--> Please run the training notebook first.")
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler missing at: {SCALER_PATH}")

        # Load Model & Scaler
        print("Loading AI Model...")
        # compile=False prevents the crash. We don't need the loss function just to predict.
        self.model = load_model(MODEL_PATH, compile=False)

        print("Loading Data Scaler...")
        self.scaler = joblib.load(SCALER_PATH)

        # Get expected features from scaler
        self.expected_features = self.scaler.n_features_in_
        print("System Ready.")

    def preprocess(self, transaction_data):
        """
        Scales raw transaction data to match the training distribution.
        """
        # Ensure input is 2D
        if transaction_data.ndim == 1:
            transaction_data = transaction_data.reshape(1, -1)
        return self.scaler.transform(transaction_data)

    def predict(self, sequence_data):
        """
        Input: Sequence of transactions (Batch, TimeSteps, Features)
        Output: Fraud Boolean, Anomaly Score, Latency
        """
        start_time = time.time()

        # 1. Reconstruct Sequence
        reconstruction = self.model.predict(sequence_data, verbose=0)

        # 2. Calculate MSE Loss (Anomaly Score)
        loss = np.mean(np.abs(reconstruction - sequence_data), axis=1)
        mean_loss = np.mean(loss)

        # 3. Decision Logic
        is_fraud = mean_loss > THRESHOLD

        latency_ms = (time.time() - start_time) * 1000
        return is_fraud, mean_loss, latency_ms


if __name__ == "__main__":
    detector = FraudDetector()

    print("\n---Processing Live Transaction Batch ---")

    # Simulate a batch of 20 transactions
    # Note: We use detector.expected_features to match the scaler (likely 8 or 9 columns)
    n_features = detector.expected_features
    dummy_sequence = np.random.rand(1, 20, n_features)

    is_fraud, score, latency = detector.predict(dummy_sequence)

    print(f"Latency: {latency:.2f} ms")
    print(f"Anomaly Score: {score:.4f} (Threshold: {THRESHOLD})")

    if is_fraud:
        print("RESULT: FRAUD DETECTED")
    else:
        print("RESULT: BENIGN TRANSACTION")
