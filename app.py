
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Professional high-contrast UI
# -----------------------------
st.markdown("""
<style>
:root {
    --navy: #102a43;
    --blue: #2563eb;
    --blue-dark: #1d4ed8;
    --text: #172033;
    --muted: #64748b;
    --border: #dbe3ec;
    --bg: #f4f7fb;
    --card: #ffffff;
}

.stApp {
    background: var(--bg);
}

/* Global text contrast */
html, body, [class*="css"], p, span, label, div {
    color: var(--text);
}

h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: var(--navy) !important;
}

/* Hero */
.hero {
    padding: 2.1rem 2.3rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f2740 0%, #174f86 55%, #2563eb 100%);
    color: white !important;
    margin-bottom: 1.5rem;
    box-shadow: 0 14px 35px rgba(16,42,67,.16);
}

.hero h1 {
    color: white !important;
    margin: 0;
    font-size: 2.45rem;
    font-weight: 800;
}

.hero p {
    color: #e8f1fb !important;
    margin: .5rem 0 0;
    font-size: 1.02rem;
}

/* Metric cards */
.metric-card {
    background: var(--card);
    padding: 1.15rem 1.25rem;
    border-radius: 16px;
    border: 1px solid var(--border);
    box-shadow: 0 5px 18px rgba(15,23,42,.05);
    min-height: 92px;
}

.metric-title {
    color: var(--muted) !important;
    font-size: .82rem;
    font-weight: 600;
}

.metric-value {
    color: var(--navy) !important;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: .2rem;
}

/* Section headers */
.section-header {
    margin: 1.4rem 0 .65rem;
    padding: .75rem 1rem;
    background: #eaf2ff;
    border-left: 5px solid var(--blue);
    border-radius: 10px;
}

.section-header h3 {
    color: var(--navy) !important;
    margin: 0;
    font-size: 1.05rem;
}

/* Cards/forms */
div[data-testid="stForm"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 8px 25px rgba(15,23,42,.06);
}

/* Widget labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stSelectbox label,
.stNumberInput label {
    color: var(--text) !important;
    font-weight: 650 !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input {
    color: var(--text) !important;
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(37,99,235,.14) !important;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] * {
    color: var(--text) !important;
}

/* Captions/help text */
[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
}

/* Main button */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: .75rem 1rem !important;
    font-weight: 750 !important;
    font-size: 1rem !important;
    box-shadow: 0 7px 18px rgba(37,99,235,.22);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: var(--text) !important;
}

/* Result */
.result-card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1.6rem;
    margin-top: 1.3rem;
    box-shadow: 0 10px 30px rgba(15,23,42,.08);
}

.result-label {
    color: var(--muted) !important;
    font-size: .9rem;
    font-weight: 650;
}

.result-value {
    color: var(--navy) !important;
    font-size: 2.25rem;
    font-weight: 850;
    margin-top: .2rem;
}

.result-note {
    color: var(--muted) !important;
    font-size: .84rem;
    margin-top: .3rem;
}

/* Info/warnings */
[data-testid="stAlert"] p {
    color: var(--text) !important;
}

/* Divider */
hr {
    border-color: var(--border) !important;
}

/* Mobile */
@media (max-width: 900px) {
    .hero h1 { font-size: 1.85rem; }
    .hero { padding: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join("model", "student_performance_xgboost.json")

FEATURES = [
    "Application mode",
    "Course",
    "Previous qualification (grade)",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Tuition fees up to date",
    "Scholarship holder",
    "Age at enrollment",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

CLASS_NAMES = ["Dropout", "Enrolled", "Graduate"]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Outcome Predictor</h1>
    <p>Machine-learning powered classification of student academic outcomes</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-card"><div class="metric-title">Model</div><div class="metric-value">XGBoost</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-title">Selected Features</div><div class="metric-value">22</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-title">Development Accuracy</div><div class="metric-value">77.15%</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-title">Validation Accuracy</div><div class="metric-value">74.04%</div></div>', unsafe_allow_html=True)

if model is None:
    st.error("Model file not found. Ensure model/student_performance_xgboost.json exists.")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## About")
    st.write(
        "This application uses the exported XGBoost model from the Student Performance Prediction project."
    )
    st.markdown("### Possible outcomes")
    st.write("**Dropout**\n\n**Enrolled**\n\n**Graduate**")
    st.markdown("### Important")
    st.caption(
        "The model uses first- and second-semester academic variables. "
        "It should not be presented as an enrollment-time early-warning model."
    )

st.markdown(
    '<div class="section-header"><h3>Student Information</h3></div>',
    unsafe_allow_html=True
)
st.caption("Use the same numeric coding scheme as the original UCI dataset.")

with st.form("prediction_form"):

    st.markdown('<div class="section-header"><h3>Admission & Background</h3></div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    with a1:
        application_mode = st.number_input("Application mode", min_value=1, max_value=57, value=17, step=1)
        course = st.number_input("Course", min_value=1, max_value=9999, value=171, step=1)
        prev_grade = st.number_input("Previous qualification grade", min_value=0.0, max_value=200.0, value=130.0, step=0.1)
    with a2:
        mother_qual = st.number_input("Mother's qualification", min_value=0, max_value=100, value=19, step=1)
        father_qual = st.number_input("Father's qualification", min_value=0, max_value=100, value=22, step=1)
        mother_occ = st.number_input("Mother's occupation", min_value=0, max_value=200, value=10, step=1)
    with a3:
        father_occ = st.number_input("Father's occupation", min_value=0, max_value=200, value=11, step=1)
        admission_grade = st.number_input("Admission grade", min_value=0.0, max_value=200.0, value=125.0, step=0.1)
        age = st.number_input("Age at enrollment", min_value=15, max_value=80, value=20, step=1)

    st.markdown('<div class="section-header"><h3>Financial & Support</h3></div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        tuition = st.selectbox(
            "Tuition fees up to date",
            [1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
    with b2:
        scholarship = st.selectbox(
            "Scholarship holder",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
    with b3:
        st.info("Binary fields use the original 0/1 dataset encoding.")

    st.markdown('<div class="section-header"><h3>First Semester Performance</h3></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s1_enrolled = st.number_input("Enrolled", min_value=0, max_value=50, value=6, step=1, key="s1e")
    with c2:
        s1_eval = st.number_input("Evaluations", min_value=0, max_value=60, value=8, step=1, key="s1v")
    with c3:
        s1_approved = st.number_input("Approved", min_value=0, max_value=50, value=5, step=1, key="s1a")
    with c4:
        s1_grade = st.number_input("Grade", min_value=0.0, max_value=20.0, value=11.0, step=0.1, key="s1g")

    st.markdown('<div class="section-header"><h3>Second Semester Performance</h3></div>', unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        s2_enrolled = st.number_input("Enrolled", min_value=0, max_value=50, value=6, step=1, key="s2e")
    with d2:
        s2_eval = st.number_input("Evaluations", min_value=0, max_value=60, value=8, step=1, key="s2v")
    with d3:
        s2_approved = st.number_input("Approved", min_value=0, max_value=50, value=5, step=1, key="s2a")
    with d4:
        s2_grade = st.number_input("Grade", min_value=0.0, max_value=20.0, value=11.0, step=0.1, key="s2g")

    st.markdown('<div class="section-header"><h3>Macroeconomic Context</h3></div>', unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        unemployment = st.number_input("Unemployment rate (%)", value=11.5, step=0.1, format="%.2f")
    with e2:
        inflation = st.number_input("Inflation rate (%)", value=1.2, step=0.1, format="%.2f")
    with e3:
        gdp = st.number_input("GDP", value=1.5, step=0.1, format="%.2f")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "🔍  Predict Student Outcome",
        type="primary",
        use_container_width=True
    )

# -----------------------------
# Prediction
# -----------------------------
if submitted:
    row = pd.DataFrame([[
        application_mode, course, prev_grade, mother_qual, father_qual,
        mother_occ, father_occ, admission_grade, tuition, scholarship, age,
        s1_enrolled, s1_eval, s1_approved, s1_grade,
        s2_enrolled, s2_eval, s2_approved, s2_grade,
        unemployment, inflation, gdp
    ]], columns=FEATURES)

    pred = int(model.predict(row)[0])
    probs = model.predict_proba(row)[0]
    label = CLASS_NAMES[pred]
    confidence = float(np.max(probs)) * 100

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    left, right = st.columns([1, 1.35])

    with left:
        st.markdown('<div class="result-label">Predicted academic outcome</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{label}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-note">Highest predicted probability: {confidence:.1f}%</div>',
            unsafe_allow_html=True
        )

        if label == "Dropout":
            st.warning("Model prediction: Dropout. Consider additional academic or student-support review.")
        elif label == "Enrolled":
            st.info("Model prediction: Enrolled.")
        else:
            st.success("Model prediction: Graduate.")

    with right:
        prob_df = pd.DataFrame({
            "Outcome": CLASS_NAMES,
            "Probability": probs * 100
        })

        fig = px.bar(
            prob_df,
            x="Probability",
            y="Outcome",
            orientation="h",
            text=prob_df["Probability"].map(lambda x: f"{x:.1f}%"),
        )

        fig.update_traces(
            marker_color="#2563eb",
            textposition="outside"
        )

        fig.update_layout(
            height=260,
            margin=dict(l=10, r=35, t=10, b=10),
            xaxis_title="Probability (%)",
            yaxis_title="",
            xaxis=dict(range=[0, 105]),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#172033"),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "These are model predictions, not definitive student-status decisions. "
        "The underlying project achieved 74.04% accuracy on the final validation subset."
    )

st.divider()

st.markdown(
    '<div style="text-align:center;color:#64748b;font-size:.82rem;">'
    'Student Outcome Predictor • XGBoost • Academic Project'
    '</div>',
    unsafe_allow_html=True
)
