
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
# Theme / styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: #f7f9fc;
    }
    .hero {
        padding: 2.0rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #102a43 0%, #1f4e79 55%, #2f80ed 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 12px 30px rgba(16,42,67,.16);
    }
    .hero h1 { margin: 0; font-size: 2.35rem; }
    .hero p { margin: .45rem 0 0; color: #e7f0fa; font-size: 1.03rem; }
    .metric-card {
        background: white;
        padding: 1.15rem 1.25rem;
        border-radius: 16px;
        border: 1px solid #e5eaf0;
        box-shadow: 0 5px 18px rgba(15,23,42,.05);
        height: 100%;
    }
    .metric-title { color: #64748b; font-size: .82rem; margin-bottom: .25rem; }
    .metric-value { color: #102a43; font-size: 1.45rem; font-weight: 750; }
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 1.6rem;
        border: 1px solid #e5eaf0;
        box-shadow: 0 8px 25px rgba(15,23,42,.07);
        margin-top: 1rem;
    }
    .result-label { color: #64748b; font-size: .9rem; }
    .result-value { font-size: 2.15rem; font-weight: 800; margin-top: .15rem; }
    .small-note { color: #64748b; font-size: .82rem; }
    .section-title { color: #102a43; margin-top: .4rem; }
    div[data-testid="stForm"] {
        background: white;
        border: 1px solid #e5eaf0;
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
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

# Top metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-title">Model</div><div class="metric-value">XGBoost</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><div class="metric-title">Input Features</div><div class="metric-value">22</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><div class="metric-title">Development Accuracy</div><div class="metric-value">77.15%</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-title">Validation Accuracy</div><div class="metric-value">74.04%</div></div>', unsafe_allow_html=True)

if model is None:
    st.error(
        "Model file not found. Put `student_performance_xgboost.json` inside the `model/` folder "
        "before deploying this app."
    )
    st.info(
        "The model is the exact artifact exported by your Colab workflow. Do not substitute a different "
        "model unless you retrain and validate it."
    )

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("### About the model")
    st.write(
        "The submitted project uses Random Forest feature selection followed by an XGBoost classifier. "
        "The final model uses 22 selected predictors."
    )
    st.markdown("### Classes")
    st.write("• Dropout\n\n• Enrolled\n\n• Graduate")
    st.markdown("### Model note")
    st.caption(
        "The underlying project uses first- and second-semester academic variables. "
        "This should not be presented as an enrollment-time early-warning model."
    )

st.markdown('<h2 class="section-title">Student Information</h2>', unsafe_allow_html=True)
st.caption("Enter the values using the same coding scheme used by the original UCI dataset.")

with st.form("prediction_form"):
    st.markdown("#### Admission & Background")
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

    st.markdown("#### Financial & Support")
    b1, b2, b3 = st.columns(3)
    with b1:
        tuition = st.selectbox("Tuition fees up to date", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    with b2:
        scholarship = st.selectbox("Scholarship holder", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    with b3:
        st.write("")
        st.caption("Binary values are encoded as 0/1 to match the source dataset.")

    st.markdown("#### First Semester")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        s1_enrolled = st.number_input("1st sem — enrolled", min_value=0, max_value=50, value=6, step=1)
    with c2:
        s1_eval = st.number_input("1st sem — evaluations", min_value=0, max_value=60, value=8, step=1)
    with c3:
        s1_approved = st.number_input("1st sem — approved", min_value=0, max_value=50, value=5, step=1)
    with c4:
        s1_grade = st.number_input("1st sem — grade", min_value=0.0, max_value=20.0, value=11.0, step=0.1)

    st.markdown("#### Second Semester")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        s2_enrolled = st.number_input("2nd sem — enrolled", min_value=0, max_value=50, value=6, step=1)
    with d2:
        s2_eval = st.number_input("2nd sem — evaluations", min_value=0, max_value=60, value=8, step=1)
    with d3:
        s2_approved = st.number_input("2nd sem — approved", min_value=0, max_value=50, value=5, step=1)
    with d4:
        s2_grade = st.number_input("2nd sem — grade", min_value=0.0, max_value=20.0, value=11.0, step=0.1)

    st.markdown("#### Macroeconomic Context")
    e1, e2, e3 = st.columns(3)
    with e1:
        unemployment = st.number_input("Unemployment rate (%)", value=11.5, step=0.1, format="%.2f")
    with e2:
        inflation = st.number_input("Inflation rate (%)", value=1.2, step=0.1, format="%.2f")
    with e3:
        gdp = st.number_input("GDP", value=1.5, step=0.1, format="%.2f")

    submitted = st.form_submit_button("Predict Student Outcome", type="primary", use_container_width=True)

if submitted:
    if model is None:
        st.stop()

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
    r1, r2 = st.columns([1.2, 1])
    with r1:
        st.markdown('<div class="result-label">Predicted academic outcome</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-value">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="small-note">Highest predicted probability: {confidence:.1f}%</div>', unsafe_allow_html=True)
    with r2:
        if label == "Dropout":
            st.warning("The model predicts Dropout. Consider academic, financial, or counseling support.")
        elif label == "Enrolled":
            st.info("The model predicts Enrolled. The student is not classified as a graduate or dropout.")
        else:
            st.success("The model predicts Graduate.")
    st.markdown("</div>", unsafe_allow_html=True)

    prob_df = pd.DataFrame({"Outcome": CLASS_NAMES, "Probability": probs * 100})
    fig = px.bar(
        prob_df,
        x="Probability",
        y="Outcome",
        orientation="h",
        text=prob_df["Probability"].map(lambda x: f"{x:.1f}%"),
        title="Model prediction probabilities",
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Probability (%)",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Important: these probabilities are model outputs, not medically, academically, or administratively "
        "certified risk scores. The project achieved 74.04% accuracy on its final validation subset, and "
        "Enrolled recall was only 36%."
    )

st.divider()
st.caption(
    "Student Outcome Predictor • Built from the submitted machine-learning project • "
    "Use predictions as decision support, not as the sole basis for student decisions."
)
