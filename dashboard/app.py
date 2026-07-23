import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ---------- Fix Paths ----------
# Get the root directory (parent of the 'dashboard' folder)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add root to sys.path so we can import f1_features if needed (optional)
sys.path.append(ROOT_DIR)

# ---------- Page Config ----------
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .stApp { background-color: #111111; color: #EEEEEE; }
    [data-testid="stSidebar"] { background-color: #1E1E1E; }
    h1, h2, h3, h4, h5, h6 { color: #E10600; }
    .stButton>button { background-color: #E10600; color: white; border-radius: 5px; }
    [data-testid="stMetric"] { background-color: #2A2A2A; border: 1px solid #E10600; border-radius: 10px; padding: 15px; }
    input[type=number] { background-color: #2A2A2A; color: white; }
</style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_path = os.path.join(ROOT_DIR, 'f1_model.pkl')
    features_path = os.path.join(ROOT_DIR, 'feature_names.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        st.error("❌ Model files not found! Please run `python src/train_and_save.py` first to generate the model.")
        return None, None
        
    try:
        loaded_model = joblib.load(model_path)
        loaded_feature_names = joblib.load(features_path)
        return loaded_model, loaded_feature_names
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

model, feature_names = load_model()
if model is None:
    st.stop()

# ---------- UI Logic (rest of your code) ----------
drivers = ['VER', 'PER', 'LEC', 'SAI', 'HAM', 'RUS', 'ALO', 'STR', 'NOR', 'PIA',
           'GAS', 'OCO', 'TSU', 'DEV', 'BOT', 'ZHO', 'MAG', 'HUL', 'ALB', 'SAR']

circuits = {
    'Bahrain': 0, 'Jeddah': 1, 'Melbourne': 1, 'Baku': 1,
    'Miami': 1, 'Monaco': 1, 'Montreal': 0, 'Silverstone': 0,
    'Hungaroring': 0, 'Spa': 0, 'Monza': 0, 'Singapore': 1,
    'Suzuka': 0, 'Losail': 0, 'Austin': 0, 'Mexico City': 0,
    'Interlagos': 0, 'Las Vegas': 1, 'Abu Dhabi': 0
}
circuit_names = list(circuits.keys())

with st.sidebar:
    st.title("🏁 Setup")
    selected_driver = st.selectbox("Driver", drivers)
    selected_circuit = st.selectbox("Circuit", circuit_names)
    grid_pos = st.slider("Grid Position", 1, 20, 5)
    form = st.slider("Driver Form (avg last 3 finishes)", 1.0, 20.0, 5.0)
    constructor_avg = st.slider("Constructor Avg Grid", 1.0, 20.0, 6.0)
    is_street = circuits[selected_circuit]
    st.caption(f"Street circuit: {'Yes' if is_street else 'No'}")

# ---------- Main panel ----------
st.title("🏎️ F1 Race Finish Predictor")
st.markdown("### Predicted final position based on race conditions")

# Build input
input_data = pd.DataFrame(
    [[grid_pos, form, constructor_avg, is_street]],
    columns=feature_names
)

try:
    raw_prediction = float(model.predict(input_data)[0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

prediction = float(np.clip(raw_prediction, 1, 20))

col1, col2 = st.columns([1, 2])

with col1:
    st.metric(
        label="Predicted Finish",
        value=f"P{prediction:.0f}",
        help=f"Raw model output: {raw_prediction:.2f}"
    )

    delta = grid_pos - prediction
    if delta > 0.05:
        st.markdown(f"<span style='color:#2ECC71; font-weight:600;'>▲ {delta:.1f} positions gained</span>", unsafe_allow_html=True)
    elif delta < -0.05:
        st.markdown(f"<span style='color:#E10600; font-weight:600;'>▼ {abs(delta):.1f} positions lost</span>", unsafe_allow_html=True)
    else:
        st.caption("No change from grid")

with col2:
    st.subheader("Feature Influence")
    st.caption("How each input affects the prediction")

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
        is_signed = False
    elif hasattr(model, "coef_"):
        importances = np.asarray(model.coef_).ravel()
        is_signed = True
    else:
        importances = None
        is_signed = False

    if importances is None:
        st.info("This model type doesn't expose feature importances.")
    else:
        order = np.argsort(np.abs(importances))
        sorted_features = [feature_names[i] for i in order]
        sorted_importances = importances[order]

        if is_signed:
            colors = ['#E10600' if v > 0 else '#4C9AFF' for v in sorted_importances]
            legend = "🔴 pushes the predicted position higher (worse)  🔵 pushes it lower (better)"
        else:
            colors = ['#E10600'] * len(sorted_importances)
            legend = "Longer bars = bigger influence on the prediction"

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(sorted_features, sorted_importances, color=colors)
        ax.set_xlabel('Importance')
        ax.set_title('Model Feature Importances')
        ax.set_facecolor('#1E1E1E')
        fig.patch.set_facecolor('#1E1E1E')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
        st.pyplot(fig)
        plt.close(fig)
        st.caption(legend)

with st.expander("Show input details"):
    st.write("Input values:")
    st.dataframe(input_data)
    st.write("Feature names (order matters):")
    st.write(list(feature_names))
    st.write(f"Model type: {type(model).__name__}")

st.caption("Model trained on 2023 season data · Predictions are estimates for illustration, not betting advice.")
