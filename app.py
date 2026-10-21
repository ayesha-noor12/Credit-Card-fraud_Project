import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────
# Page configuration — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS — refined dark professional theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

    /* Base */
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .stApp {
        background-color: #0d0f12;
        color: #e2e8f0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111318;
        border-right: 1px solid #1e2330;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
        font-size: 0.82rem;
    }

    /* Main content area */
    .main .block-container {
        padding: 2rem 2.5rem;
        max-width: 1200px;
    }

    /* Header */
    .app-header {
        border-bottom: 1px solid #1e2330;
        padding-bottom: 1.5rem;
        margin-bottom: 2rem;
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 0.3rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Cards */
    .card {
        background: #111318;
        border: 1px solid #1e2330;
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #475569;
        margin-bottom: 1rem;
    }

    /* Result cards */
    .result-fraud {
        background: #1a0a0a;
        border: 1px solid #7f1d1d;
        border-radius: 8px;
        padding: 1.8rem 2rem;
    }
    .result-legit {
        background: #071610;
        border: 1px solid #14532d;
        border-radius: 8px;
        padding: 1.8rem 2rem;
    }
    .result-label {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .result-fraud .result-label { color: #fca5a5; }
    .result-legit .result-label { color: #86efac; }
    .result-prob {
        font-size: 2.4rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
        margin: 0.5rem 0;
    }
    .result-fraud .result-prob { color: #ef4444; }
    .result-legit .result-prob { color: #22c55e; }
    .result-desc {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.4rem;
    }

    /* Metric row */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1.2rem;
    }
    .metric-box {
        flex: 1;
        background: #0d0f12;
        border: 1px solid #1e2330;
        border-radius: 6px;
        padding: 0.9rem 1rem;
    }
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: 500;
        font-family: 'IBM Plex Mono', monospace;
        color: #e2e8f0;
    }

    /* Inputs */
    .stNumberInput input, .stTextInput input {
        background: #0d0f12 !important;
        border: 1px solid #1e2330 !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.9rem !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
    }
    .stSlider {
        padding: 0.2rem 0;
    }
    .stSlider [data-testid="stSlider"] {
        padding: 0;
    }

    /* Button */
    .stButton > button {
        background: #1d4ed8 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.5rem !important;
        letter-spacing: 0.02em !important;
        transition: background 0.15s !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #2563eb !important;
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #475569;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* Error / warning */
    .stAlert {
        border-radius: 6px !important;
        font-size: 0.85rem !important;
    }

    /* Progress bar */
    .stProgress > div > div {
        background: #1e2330;
        border-radius: 4px;
    }
    .stProgress > div > div > div {
        border-radius: 4px;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #111318 !important;
        border: 1px solid #1e2330 !important;
        border-radius: 6px !important;
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
    }

    /* Divider */
    hr {
        border-color: #1e2330 !important;
        margin: 1.5rem 0 !important;
    }

    /* Hide Streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load model and scaler
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load trained model and scaler from disk. Cached so they load only once."""
    try:
        model = joblib.load('fraud_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, None
    except FileNotFoundError as e:
        return None, None, str(e)

model, scaler, load_error = load_artifacts()


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Fraud Detection System")
    st.markdown("---")
    
    if load_error:
        st.error("Model files not found.")
        st.markdown("""
        **Setup required:**
        1. Run `fraud_detection.ipynb` completely
        2. Confirm `fraud_model.pkl` and `scaler.pkl` are created
        3. Restart this app
        """)
    else:
        st.markdown("**Status**")
        st.success("Model loaded")
        st.markdown("---")

    st.markdown("**About**")
    st.markdown("""
    This system uses a trained machine learning model to classify credit card transactions
    as legitimate or fraudulent in real time.
    """)
    
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("""
    - Source: Kaggle (mlg-ulb)
    - Records: 284,807
    - Features: 30
    - Fraud rate: 0.17%
    """)
    
    st.markdown("---")
    st.markdown("**Key metrics**")
    st.markdown("""
    On imbalanced fraud data, **Recall** and **F1-Score** are more meaningful
    than Accuracy. A model predicting everything as legitimate achieves 99.83%
    accuracy yet catches zero fraud.
    """)

    st.markdown("---")
    st.markdown("**How to use**")
    st.markdown("""
    1. Enter the transaction Amount and Time
    2. Optionally adjust V1–V28 PCA features
    3. Click **Run Prediction**
    4. Review the result and confidence score
    """)


# ─────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <p class="app-title">Credit Card Fraud Detection</p>
    <p class="app-subtitle">Real-time transaction classification using ensemble machine learning</p>
</div>
""", unsafe_allow_html=True)

if load_error:
    st.error(
        f"Could not load model files. Please run the Jupyter notebook first "
        f"to train and save `fraud_model.pkl` and `scaler.pkl`."
    )
    st.stop()


# ─────────────────────────────────────────────
# Input section
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">Transaction Details</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card"><p class="card-title">Primary Inputs</p>', unsafe_allow_html=True)
    
    amount = st.number_input(
        "Transaction Amount (USD)",
        min_value=0.01,
        max_value=50000.0,
        value=149.62,
        step=0.01,
        format="%.2f",
        help="The monetary value of the transaction in US dollars."
    )
    
    time_val = st.number_input(
        "Time (seconds since dataset start)",
        min_value=0.0,
        max_value=200000.0,
        value=50000.0,
        step=1.0,
        format="%.0f",
        help="Number of seconds elapsed between this transaction and the first transaction in the dataset."
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <p class="card-title">Feature Guide</p>
        <p style="font-size: 0.82rem; color: #64748b; line-height: 1.7;">
            <strong style="color: #94a3b8;">Amount</strong> — Transaction value in USD. 
            Most fraudulent transactions occur at smaller amounts.<br><br>
            <strong style="color: #94a3b8;">Time</strong> — Seconds elapsed since the 
            first transaction. Used to capture temporal patterns.<br><br>
            <strong style="color: #94a3b8;">V1–V28</strong> — Anonymized PCA-transformed 
            features from the original dataset. Default 0.0 represents a 
            typical transaction profile. Adjust only if you have actual 
            transaction feature values.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PCA features — collapsible
# ─────────────────────────────────────────────
st.markdown('<p class="section-label">PCA Features</p>', unsafe_allow_html=True)

with st.expander("V1 – V28  (click to expand — default 0.0 for typical transaction)"):
    v_vals = {}
    
    # Row 1: V1–V7
    st.markdown("**V1 – V7**")
    cols_a = st.columns(7)
    for i, col in enumerate(cols_a, start=1):
        with col:
            v_vals[f'V{i}'] = st.number_input(
                f"V{i}", value=0.0, min_value=-30.0, max_value=30.0,
                step=0.01, format="%.3f", key=f"v{i}", label_visibility="visible"
            )
    
    # Row 2: V8–V14
    st.markdown("**V8 – V14**")
    cols_b = st.columns(7)
    for i, col in enumerate(cols_b, start=8):
        with col:
            v_vals[f'V{i}'] = st.number_input(
                f"V{i}", value=0.0, min_value=-30.0, max_value=30.0,
                step=0.01, format="%.3f", key=f"v{i}", label_visibility="visible"
            )
    
    # Row 3: V15–V21
    st.markdown("**V15 – V21**")
    cols_c = st.columns(7)
    for i, col in enumerate(cols_c, start=15):
        with col:
            v_vals[f'V{i}'] = st.number_input(
                f"V{i}", value=0.0, min_value=-30.0, max_value=30.0,
                step=0.01, format="%.3f", key=f"v{i}", label_visibility="visible"
            )
    
    # Row 4: V22–V28
    st.markdown("**V22 – V28**")
    cols_d = st.columns(7)
    for i, col in enumerate(cols_d, start=22):
        with col:
            v_vals[f'V{i}'] = st.number_input(
                f"V{i}", value=0.0, min_value=-30.0, max_value=30.0,
                step=0.01, format="%.3f", key=f"v{i}", label_visibility="visible"
            )


# ─────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────
def validate_inputs(amount, time_val):
    errors = []
    if amount <= 0:
        errors.append("Transaction amount must be greater than 0.")
    if time_val < 0:
        errors.append("Time value cannot be negative.")
    if amount > 30000:
        st.warning(
            "Amount exceeds $30,000. This is unusual but will be processed. "
            "Verify this is intentional."
        )
    return errors


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
st.markdown("---")
predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

if predict_clicked:
    errors = validate_inputs(amount, time_val)
    
    if errors:
        for error in errors:
            st.error(error)
    else:
        # Build feature vector in exact training order: Time, V1–V28, Amount
        feature_values = (
            [time_val] +
            [v_vals[f'V{i}'] for i in range(1, 29)] +
            [amount]
        )
        
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_df = pd.DataFrame([feature_values], columns=feature_names)
        
        # Scale Time and Amount using the saved scaler
        input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])
        
        # Rename to match training column names
        input_df.rename(columns={'Time': 'Time_scaled', 'Amount': 'Amount_scaled'}, inplace=True)
        
        # Reorder to match training feature order
        model_features = ['Time_scaled'] + [f'V{i}' for i in range(1, 29)] + ['Amount_scaled']
        input_df = input_df[model_features]
        
        # Predict
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        fraud_prob = probabilities[1]
        legit_prob = probabilities[0]
        
        # ── Result display ──────────────────────
        st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)
        
        res_col1, res_col2 = st.columns([1.2, 1], gap="large")
        
        with res_col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    <p class="result-label">Fraudulent Transaction</p>
                    <p class="result-prob">{fraud_prob * 100:.1f}%</p>
                    <p class="result-desc">Fraud probability score</p>
                    <div class="metric-row">
                        <div class="metric-box">
                            <p class="metric-label">Fraud prob.</p>
                            <p class="metric-value" style="color:#ef4444">{fraud_prob:.4f}</p>
                        </div>
                        <div class="metric-box">
                            <p class="metric-label">Legit prob.</p>
                            <p class="metric-value">{legit_prob:.4f}</p>
                        </div>
                        <div class="metric-box">
                            <p class="metric-label">Decision</p>
                            <p class="metric-value" style="color:#ef4444">Flag</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    <p class="result-label">Legitimate Transaction</p>
                    <p class="result-prob">{legit_prob * 100:.1f}%</p>
                    <p class="result-desc">Legitimacy confidence score</p>
                    <div class="metric-row">
                        <div class="metric-box">
                            <p class="metric-label">Legit prob.</p>
                            <p class="metric-value" style="color:#22c55e">{legit_prob:.4f}</p>
                        </div>
                        <div class="metric-box">
                            <p class="metric-label">Fraud prob.</p>
                            <p class="metric-value">{fraud_prob:.4f}</p>
                        </div>
                        <div class="metric-box">
                            <p class="metric-label">Decision</p>
                            <p class="metric-value" style="color:#22c55e">Approve</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with res_col2:
            st.markdown('<div class="card"><p class="card-title">Confidence Breakdown</p>', unsafe_allow_html=True)
            
            st.markdown("**Fraud probability**")
            st.progress(float(fraud_prob))
            st.markdown(f"`{fraud_prob * 100:.2f}%`")
            
            st.markdown("**Legitimate probability**")
            st.progress(float(legit_prob))
            st.markdown(f"`{legit_prob * 100:.2f}%`")
            
            st.markdown("---")
            if prediction == 1:
                st.markdown("""
                **Recommended action:**  
                Hold the transaction and initiate a verification call to the cardholder.
                Do not decline immediately — verify identity first.
                """)
            else:
                st.markdown("""
                **Recommended action:**  
                Transaction cleared. No further review required.
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # ── Input summary ──────────────────────
        with st.expander("Input summary"):
            summary = pd.DataFrame({
                'Feature': ['Amount', 'Time'] + [f'V{i}' for i in range(1, 29)],
                'Value': [amount, time_val] + [v_vals[f'V{i}'] for i in range(1, 29)]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
