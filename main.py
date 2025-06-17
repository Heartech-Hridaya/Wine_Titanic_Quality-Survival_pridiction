# main.py
import math
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
#  0️⃣  Page‑level settings
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ML Playground – Wine & Titanic",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Machine‑Learning Playground")
st.caption("Predict wine quality **or** Titanic passenger survival with one click.")

# ---------------------------------------------------------------------------
# 1️⃣  Sidebar – App selector + model selector
# ---------------------------------------------------------------------------
APP_CHOICES = {
    "Wine Quality": {
        "baseline": "svr_baseline_model.pkl",
        "tuned":    "svr_tuned_model.pkl",
        "emoji": "🍷",
    },
    "Titanic Survival": {
        "baseline": "test_baseline_model.pkl",
        "tuned":    "test_tuned_model.pkl",
        "emoji": "🚢",
    },
}

app_type = st.sidebar.radio("Choose a predictor", list(APP_CHOICES.keys()))
model_flavour = st.sidebar.selectbox("Choose model variant",
                                     ["Baseline Model", "Tuned Model"],
                                     index=0)

model_key  = "baseline" if model_flavour.startswith("Baseline") else "tuned"
model_file = APP_CHOICES[app_type][model_key]
MODEL_PATH = Path(__file__).parent / model_file

# ---------------------------------------------------------------------------
# 2️⃣  Utility – load model only once
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# 3️⃣  Wine Quality inputs & prediction
# ---------------------------------------------------------------------------
if app_type == "Wine Quality":
    st.header(f"{APP_CHOICES[app_type]['emoji']} Wine Physicochemical Properties")

    fixed_acidity  = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0, 0.1)
    volatile_acid  = st.number_input("Volatile Acidity", 0.0, 5.0, 0.7, 0.01)
    citric_acid    = st.number_input("Citric Acid", 0.0, 2.0, 0.3, 0.01)
    residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 2.5, 0.1)
    chlorides      = st.number_input("Chlorides", 0.0, 0.5, 0.08, 0.001)
    free_SO2       = st.number_input("Free Sulfur Dioxide", 0, 100, 15, 1)
    total_SO2      = st.number_input("Total Sulfur Dioxide", 0, 300, 45, 1)
    density        = st.number_input("Density", 0.9900, 1.0100, 0.9950, 0.0001, format="%.4f")
    ph_val         = st.number_input("pH", 2.0, 5.0, 3.3, 0.01)
    sulphates      = st.number_input("Sulphates", 0.0, 2.0, 0.6, 0.01)
    alcohol        = st.number_input("Alcohol (%)", 0.0, 20.0, 10.0, 0.1)

    st.subheader("Engineered Features")
    acid_ratio   = st.number_input("Acid Ratio", 0.0, 10.0, 1.0, 0.01)
    sulfur_ratio = st.number_input("Sulfur Ratio", 0.0, 10.0, 1.0, 0.01)
    sugar_alc    = st.number_input("Sugar‑Alcohol Ratio", 0.0, 10.0, 0.2, 0.01)
    total_acid   = st.number_input("Total Acidity", 0.0, 50.0, 7.5, 0.1)
    alc_sul_int  = st.number_input("Alcohol‑Sulphates Interaction", 0.0, 20.0, 6.0, 0.01)

    if st.button("Predict Wine Quality"):
        wine_feats = np.array([[
            fixed_acidity, volatile_acid, citric_acid, residual_sugar, chlorides,
            free_SO2, total_SO2, density, ph_val, sulphates, alcohol,
            acid_ratio, sulfur_ratio, sugar_alc, total_acid, alc_sul_int
        ]])
        score = model.predict(wine_feats)[0]
        st.success(f"**Predicted Quality Score:** {score:.2f}")

# ---------------------------------------------------------------------------
# 4️⃣  Titanic inputs & prediction
# ---------------------------------------------------------------------------
else:
    st.header(f"{APP_CHOICES[app_type]['emoji']} Titanic Passenger Features")

    # Basic inputs
    pclass = st.selectbox("Passenger Class (1st/2nd/3rd)", [1, 2, 3], index=2)
    age    = st.number_input("Age", 0.0, 80.0, 32.0, 0.5)
    sibsp  = st.number_input("Siblings/Spouses Aboard", 0, 8, 0)
    parch  = st.number_input("Parents/Children Aboard", 0, 6, 0)
    fare   = st.number_input("Ticket Fare (£)", 0.0, 600.0, 32.0)
    cabin  = st.selectbox("Cabin information available?", ["No", "Yes"])
    cabin_flag = 1 if cabin == "Yes" else 0

    # Extra inputs (only used if model needs them)
    sex     = st.selectbox("Sex", ["male", "female"])
    sex_enc = 1 if sex == "male" else 0
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    embarked_enc = {"S": 0, "C": 1, "Q": 2}[embarked]

    # Derived
    family_size = sibsp + parch + 1
    fare_log    = math.log(fare + 1)
    fare_cap    = min(fare, 300.0)

    st.write(f"ℹ️ **FamilySize:** {family_size} | **log(Fare+1):** {fare_log:.3f} | **Fare Capped:** {fare_cap:.2f}")

    # Build feature vector dynamically
    n_features = getattr(model, "n_features_in_", 11)
    if n_features == 9:        # baseline
        titanic_feats = np.array([[pclass, age, sibsp, parch, fare,
                                   family_size, fare_log, fare_cap, cabin_flag]])
    elif n_features == 11:     # tuned
        titanic_feats = np.array([[pclass, age, sibsp, parch, fare,
                                   family_size, fare_log, fare_cap, cabin_flag,
                                   sex_enc, embarked_enc]])
    else:
        st.error(f"Model expects {n_features} features – please update form.")
        st.stop()

    if st.button("Predict Survival"):
        # Probabilities if available, else plain prediction
        survived_prob = (
            model.predict_proba(titanic_feats)[0][1]
            if hasattr(model, "predict_proba")
            else model.predict(titanic_feats)[0]
        )
        outcome = "🎉 Survived!" if survived_prob >= 0.5 else "💧 Did not survive"
        st.markdown(f"### {outcome}\n**Probability of survival:** {survived_prob:.2%}")

# ---------------------------------------------------------------------------
# 5️⃣  Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ by Hridaya Manandhar & Pramisha Giri</p>",
    unsafe_allow_html=True,
)
