"""
app.py
------
Streamlit front-end for the Pima Indians Diabetes prediction model.

Requirements:
    pip install streamlit joblib scikit-learn numpy pandas

Run:
    streamlit run app.py

Expects 'diabetes_model.pkl' and 'scaler.pkl' in the same directory.
If scaler.pkl is absent the raw feature vector is sent to the model directly.
"""

import os
import joblib
import numpy as np
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "Diabetes Risk Predictor",
    page_icon  = "🩺",
    layout     = "centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Tokens ── */
:root {
    --bg:          #F7F9FC;
    --surface:     #FFFFFF;
    --border:      #DDE3ED;
    --text-primary:#1A2235;
    --text-muted:  #6B7A99;
    --accent:      #2563EB;
    --accent-soft: #EEF3FF;
    --danger:      #DC2626;
    --danger-soft: #FEF2F2;
    --safe:        #16A34A;
    --safe-soft:   #F0FDF4;
    --radius:      10px;
    --mono:        'JetBrains Mono', 'Fira Mono', monospace;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: var(--text-primary);
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-icon {
    font-size: 2.8rem;
    line-height: 1;
    margin-bottom: 0.6rem;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0 0 0.4rem;
}
.hero p {
    font-size: 0.95rem;
    color: var(--text-muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 1.1rem;
}

/* ── Input labels & helpers ── */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}
.helper {
    font-size: 0.73rem;
    color: var(--text-muted);
    margin-top: -0.55rem;
    margin-bottom: 0.5rem;
    line-height: 1.4;
}

/* ── Number inputs ── */
input[type=number] {
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
    font-family: var(--mono) !important;
    font-size: 0.9rem !important;
}
input[type=number]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    padding: 0.65rem 1rem !important;
    margin-top: 0.4rem;
    transition: background 0.15s, transform 0.1s;
}
div[data-testid="stButton"] > button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0);
}

/* ── Result banners ── */
.result-box {
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-top: 1.2rem;
    border: 1.5px solid;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
}
.result-box.positive {
    background: var(--danger-soft);
    border-color: var(--danger);
}
.result-box.negative {
    background: var(--safe-soft);
    border-color: var(--safe);
}
.result-icon { font-size: 2.2rem; line-height: 1; flex-shrink: 0; }
.result-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.result-label.pos { color: var(--danger); }
.result-label.neg { color: var(--safe); }
.result-headline {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.result-sub {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.5;
}

/* ── Confidence bar ── */
.conf-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-top: 1rem;
    margin-bottom: 0.3rem;
}
.conf-track {
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.6s ease;
}

/* ── Reference ranges ── */
.ref-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem 1.5rem;
    margin-top: 0.4rem;
}
.ref-item {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--border);
    color: var(--text-muted);
}
.ref-item span:first-child { color: var(--text-primary); font-weight: 500; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.72rem;
    color: var(--text-muted);
    padding: 2rem 0 1rem;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ── Load model & scaler ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path  = "diabetes_model.pkl"
    scaler_path = "scaler.pkl"

    if not os.path.exists(model_path):
        st.error(
            f"Model file **'{model_path}'** not found. "
            "Run `train_model.py` first to generate it."
        )
        st.stop()

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler


model, scaler = load_artifacts()


# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-icon">🩺</div>
    <h1>Diabetes Risk Predictor</h1>
    <p>Enter the patient's clinical measurements below.
       The model will assess the likelihood of a positive diabetes diagnosis.</p>
</div>
""", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────

st.markdown('<div class="card"><div class="card-title">Reproductive &amp; Metabolic</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0, max_value=20, value=1, step=1,
        help="Number of times pregnant"
    )
    st.markdown('<p class="helper">Times pregnant (0 – 20)</p>', unsafe_allow_html=True)

    glucose = st.number_input(
        "Glucose  (mg/dL)",
        min_value=0, max_value=300, value=110, step=1,
        help="Plasma glucose concentration (2-hour oral glucose tolerance test)"
    )
    st.markdown('<p class="helper">2-hr plasma glucose • Normal: 70–140</p>', unsafe_allow_html=True)

    blood_pressure = st.number_input(
        "Blood Pressure  (mm Hg)",
        min_value=0, max_value=140, value=72, step=1,
        help="Diastolic blood pressure"
    )
    st.markdown('<p class="helper">Diastolic BP • Normal: 60–80</p>', unsafe_allow_html=True)

    skin_thickness = st.number_input(
        "Skin Thickness  (mm)",
        min_value=0, max_value=100, value=20, step=1,
        help="Triceps skinfold thickness"
    )
    st.markdown('<p class="helper">Triceps skinfold • Normal: 10–40</p>', unsafe_allow_html=True)

with col2:
    insulin = st.number_input(
        "Insulin  (μU/mL)",
        min_value=0, max_value=900, value=80, step=1,
        help="2-hour serum insulin"
    )
    st.markdown('<p class="helper">2-hr serum insulin • Normal: 16–166</p>', unsafe_allow_html=True)

    bmi = st.number_input(
        "BMI  (kg/m²)",
        min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        format="%.1f",
        help="Body Mass Index"
    )
    st.markdown('<p class="helper">Body Mass Index • Normal: 18.5–24.9</p>', unsafe_allow_html=True)

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.000, max_value=3.000, value=0.350, step=0.001,
        format="%.3f",
        help="Genetic influence score based on family history"
    )
    st.markdown('<p class="helper">Family history score • Typical: 0.08–2.42</p>', unsafe_allow_html=True)

    age = st.number_input(
        "Age  (years)",
        min_value=1, max_value=120, value=30, step=1,
        help="Age in years"
    )
    st.markdown('<p class="helper">Patient age</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close card


# ── Predict button ────────────────────────────────────────────────────────────

predict_clicked = st.button("🔍  Run Prediction", width=True)


# ── Prediction logic & result display ────────────────────────────────────────

if predict_clicked:
    features = np.array([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], dtype=float)

    # Scale if scaler is available
    if scaler is not None:
        features_input = scaler.transform(features)
    else:
        features_input = features

    prediction   = model.predict(features_input)[0]
    has_proba    = hasattr(model, "predict_proba")
    confidence   = float(model.predict_proba(features_input)[0][prediction]) if has_proba else None

    if prediction == 1:
        conf_pct  = f"{confidence * 100:.1f}%" if confidence else "N/A"
        fill_col  = "#DC2626"
        fill_w    = f"{confidence * 100:.1f}%" if confidence else "0%"

        st.markdown(f"""
        <div class="result-box positive">
            <div class="result-icon">⚠️</div>
            <div>
                <div class="result-label pos">Prediction result</div>
                <div class="result-headline">Positive for Diabetes Risk</div>
                <div class="result-sub">
                    The model predicts a <strong>high likelihood</strong> of diabetes
                    based on the provided measurements.
                    Please consult a healthcare professional for further evaluation.
                </div>
                {"" if not confidence else f'''
                <div class="conf-row">
                    <span>Model confidence</span><span>{conf_pct}</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{fill_w};background:{fill_col};"></div>
                </div>
                '''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        conf_pct  = f"{confidence * 100:.1f}%" if confidence else "N/A"
        fill_col  = "#16A34A"
        fill_w    = f"{confidence * 100:.1f}%" if confidence else "0%"

        st.markdown(f"""
        <div class="result-box negative">
            <div class="result-icon">✅</div>
            <div>
                <div class="result-label neg">Prediction result</div>
                <div class="result-headline">Low Diabetes Risk Detected</div>
                <div class="result-sub">
                    The model predicts a <strong>low likelihood</strong> of diabetes
                    based on the provided measurements.
                    Maintain a healthy lifestyle and schedule regular check-ups.
                </div>
                {"" if not confidence else f'''
                <div class="conf-row">
                    <span>Model confidence</span><span>{conf_pct}</span>
                </div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{fill_w};background:{fill_col};"></div>
                </div>
                '''}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input summary ──────────────────────────────────────────────────────
    with st.expander("View input summary", expanded=False):
        import pandas as pd
        summary = pd.DataFrame({
            "Feature": [
                "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
                "Insulin", "BMI", "Diabetes Pedigree Function", "Age"
            ],
            "Value entered": [
                pregnancies, f"{glucose} mg/dL", f"{blood_pressure} mm Hg",
                f"{skin_thickness} mm", f"{insulin} μU/mL",
                f"{bmi:.1f} kg/m²", f"{dpf:.3f}", f"{age} yrs"
            ]
        })
        st.dataframe(summary, width=True, hide_index=True)


# ── Reference ranges ──────────────────────────────────────────────────────────

st.markdown("""
<div class="card" style="margin-top:1.4rem;">
    <div class="card-title">Clinical Reference Ranges</div>
    <div class="ref-grid">
        <div class="ref-item"><span>Glucose</span><span>70 – 140 mg/dL</span></div>
        <div class="ref-item"><span>Insulin</span><span>16 – 166 μU/mL</span></div>
        <div class="ref-item"><span>Blood Pressure</span><span>60 – 80 mm Hg</span></div>
        <div class="ref-item"><span>BMI</span><span>18.5 – 24.9 kg/m²</span></div>
        <div class="ref-item"><span>Skin Thickness</span><span>10 – 40 mm</span></div>
        <div class="ref-item"><span>Pedigree Function</span><span>0.08 – 2.42</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    Trained on the Pima Indians Diabetes Dataset (NIDDK) &nbsp;·&nbsp;
    For research and educational use only<br>
    <strong>Not a substitute for professional medical advice.</strong>
</div>
""", unsafe_allow_html=True)
