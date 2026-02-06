Deep Learning Blockchain Fraud DetectionA real-time anomaly detection engine for Ethereum (ERC-20) token transactions. This system leverages a Recurrent Autoencoder (LSTM) to analyze temporal transaction sequences, identifying complex fraud patterns (like wallet draining and rapid-exit schemes) that traditional rule-based systems miss.
Key Results & Impact
Unlike traditional classifiers that struggle with unseen attacks, this semi-supervised model learns the "shape" of legitimate behavior and flags deviations.
    1. 93.6% Precision: High confidence in flagged alerts, significantly reducing false positives for compliance teams.
    2. Dynamic Thresholding: Utilized statistical analysis (95th percentile of reconstruction error) to set a robust cutoff at 0.1949, replacing brittle hardcoded values.
    3. <50ms Latency: Optimized inference pipeline suitable for real-time blockchain monitoring.

Performance Visualizations
1. The Separation (Fraud vs. Benign)The model successfully separates normal traffic (green) from malicious attacks (red) based on Reconstruction Error.(Note: Distinct separation validates the model's ability to distinguish behavioral patterns.)
2. Confusion MatrixValidation on a mixed test set reveals strong precision with minimal false alarms.

System Architecture
The core model is an LSTM Autoencoder designed to reconstruct sequences of 20 consecutive transactions.
    1. Input: Sequence of 20 transactions $\times$ 9 features (Amount, Balance State, Approval Limit, etc.).
    2. Encoder: Compresses the sequence into a lower-dimensional latent vector, capturing the "context" of the financial activity.
    3. Decoder: Attempts to reconstruct the original sequence from the latent vector.
    4. Anomaly Logic:
        Low Error: The pattern matches learned "normal" behavior $\rightarrow$ Benign.
        High Error: The pattern is unfamiliar/erratic $\rightarrow$ Fraud.

Project Structure
├── data/ 
│   └──  processed/            # Processed transaction datasets (AG vs MAL)
│   └──  schemas/              #Defines Pydantic models for Ethereum transaction validation.Acts as a data contract to prevent malformed data from crashing the inference engine.
├── images/                # Performance plots and architecture diagrams
├── models/                # Serialized Model (.h5) and Scaler (.pkl) artifacts
├── notebooks/             # Primary Training & Analysis Notebook
│   └──  test_notebooks/        # Deprecated experiments and initial EDA (Reference)
├── src/                   # Production-grade inference scripts
│   └── inference.py       # Main entry point for real-time detection
├── .gitignore             # Configuration for ignored files
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
Installation & Usage
1. Prerequisites
    Python 3.10+Git2. SetupBash# Clone the repository
git clone https://github.com/YOUR_USERNAME/Blockchain-Fraud-Detection.git
cd Blockchain-Fraud-Detection

# Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
3. Run InferenceTo simulate a live batch of transactions and detect fraud:Bashpython src/inference.py
Expected Output:Plaintext--- 📡 Processing Live Transaction Batch ---
⏱️  Latency: 42.15 ms
📉  Anomaly Score: 0.3752 (Threshold: 0.1949)
🚨  RESULT: FRAUD DETECTED
📝 MethodologyData Preprocessing:Features were scaled using MinMaxScaler to ensure model stability.Sliding Window approach created sequences of T=20 to capture temporal dependencies.Training:Trained exclusively on Benign (AG) data.Loss Function: Mean Absolute Error (MAE).Optimizer: Adam.Evaluation:Tested against a mixed dataset containing known Malicious (MAL) traces (e.g., Poly Network hack patterns).Threshold determined via the 95th percentile of training loss to ensure 95% specificity.🔮 Future ImprovementsGraph Neural Networks (GNN): Incorporate wallet interaction graphs to detect ring-fencing fraud.API Deployment: Wrap inference.py in a FastAPI container for a live REST endpoint.Feedback Loop: Implement active learning to update the model with confirmed fraud cases.📜 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.Built by Bhargav Madhav | LinkedIn | GitHub