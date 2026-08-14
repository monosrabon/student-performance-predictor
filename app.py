
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from xgboost import XGBClassifier

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Outcome Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Light-blue / black professional theme
# ---------------------------------------------------------
st.markdown("""
<style>
/* Main page */
.stApp {
    background: #eaf4ff;
    color: #000000 !important;
}

/* Force normal text to black */
p, label, span, small, div {
    color: #000000;
}

/* Headings */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #000000 !important;
}

/* Hero */
.hero {
    background: #b9dcff;
    border: 1px solid #8fc5f5;
    border-radius: 22px;
    padding: 28px 32px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(0, 70, 140, 0.10);
}

.hero h1 {
    color: #000000 !important;
    font-size: 2.3rem;
    font-weight: 800;
    margin: 0;
}

.hero p {
    color: #000000 !important;
    margin: 7px 0 0;
    font-size: 1rem;
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #b7d7f4;
    border-radius: 16px;
    padding: 16px 18px;
    min-height: 92px;
    box-shadow: 0 5px 18px rgba(0, 70, 140, 0.07);
}

.metric-title {
    color: #000000 !important;
    font-size: 0.82rem;
    font-weight: 600;
}

.metric-value {
    color: #000000 !important;
    font-size: 1.4rem;
    font-weight: 800;
    margin-top: 4px;
}

/* Section headers */
.section-header {
    background: #cfe7ff;
    border: 1px solid #9ecdf5;
    border-left: 6px solid #3182ce;
    border-radius: 12px;
    padding: 12px 16px;
    margin: 18px 0 10px;
}

.section-header h3 {
    color: #000000 !important;
    margin: 0;
    font-size: 1.08rem;
}

/* Form */
div[data-testid="stForm"] {
    background: #ffffff;
    border: 1px solid #b7d7f4;
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 8px 25px rgba(0, 70, 140, 0.07);
}

/* Input labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #000000 !important;
    font-weight: 650 !important;
}

/* Number input */
[data-testid="stNumberInput"] input {
    color: #000000 !important;
    background: #ffffff !important;
    border: 1px solid #7fb4df !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

/* Number input buttons */
[data-testid="stNumberInput"] button {
    color: #000000 !important;
    background: #dceeff !important;
    border-color: #7fb4df !important;
}

/* Select box closed state */
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #7fb4df !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] [role="combobox"],
[data-baseweb="select"] [role="combobox"] * {
    color: #000000 !important;
    background: #ffffff !important;
}

/* Select dropdown popup */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
ul[role="listbox"],
ul[role="listbox"] * {
    color: #000000 !important;
    background: #ffffff !important;
}

[role="listbox"] {
    border: 1px solid #9ecdf5 !important;
    border-radius: 10px !important;
    box-shadow: 0 12px 28px rgba(0, 70, 140, 0.18) !important;
}

[role="option"] {
    color: #000000 !important;
    background: #ffffff !important;
}

[role="option"]:hover,
[role="option"][aria-selected="true"] {
    color: #000000 !important;
    background: #dceeff !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p {
    color: #333333 !important;
}

/* Info boxes */
[data-testid="stAlert"] {
    background: #dceeff !important;
    border: 1px solid #9ecdf5 !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div {
    color: #000000 !important;
}

/* Main prediction button */
.stFormSubmitButton > button {
    width: 100%;
    background: #3182ce !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    box-shadow: 0 6px 16px rgba(49, 130, 206, 0.25);
}

.stFormSubmitButton > button:hover {
    background: #256bb0 !important;
}

/* Result */
.result-card {
    background: #ffffff;
    border: 1px solid #9ecdf5;
    border-radius: 20px;
    padding: 22px;
    margin-top: 20px;
    box-shadow: 0 9px 26px rgba(0, 70, 140, 0.10);
}

.result-label {
    color: #333333 !important;
    font-size: 0.9rem;
    font-weight: 650;
}

.result-value {
    color: #000000 !important;
    font-size: 2.2rem;
    font-weight: 850;
    margin-top: 4px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #dceeff !important;
    border-right: 1px solid #9ecdf5;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] span {
    color: #000000 !important;
}

/* Divider */
hr {
    border-color: #9ecdf5 !important;
}

/* Mobile */
@media (max-width: 800px) {
    .hero h1 {
        font-size: 1.8rem;
    }

    .hero {
        padding: 22px;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
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

if model is None:
    st.error(
        "Model file not found. Please make sure "
        "`model/student_performance_xgboost.json` exists."
    )
    st.stop()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎓 Student Outcome Predictor</h1>
    <p>Machine Learning Based Student Academic Outcome Prediction</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Dashboard metrics
# ---------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Machine Learning Model</div>
        <div class="metric-value">XGBoost</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Selected Features</div>
        <div class="metric-value">22</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Development Accuracy</div>
        <div class="metric-value">77.15%</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Validation Accuracy</div>
        <div class="metric-value">74.04%</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## About the System")
    st.write(
        "This application predicts a student's academic outcome "
        "using the XGBoost model developed in the project."
    )

    st.markdown("### Prediction Classes")
    st.write("**Dropout**")
    st.write("**Enrolled**")
    st.write("**Graduate**")

    st.markdown("### Important Note")
    st.caption(
        "The model uses first- and second-semester academic variables. "
        "Therefore, it should not be described as a purely enrollment-time "
        "early-warning system."
    )

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
st.markdown(
    '<div class="section-header"><h3>Student Information</h3></div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter values using the same coding scheme as the original dataset."
)

with st.form("prediction_form"):

    # Admission
    st.markdown(
        '<div class="section-header"><h3>Admission & Background</h3></div>',
        unsafe_allow_html=True
    )

    a1, a2, a3 = st.columns(3)

    with a1:
        application_mode = st.number_input(
            "Application mode",
            min_value=1,
            max_value=57,
            value=17,
            step=1
        )

        course = st.number_input(
            "Course",
            min_value=1,
            max_value=9999,
            value=171,
            step=1
        )

        prev_grade = st.number_input(
            "Previous qualification grade",
            min_value=0.0,
            max_value=200.0,
            value=130.0,
            step=0.1
        )

    with a2:
        mother_qual = st.number_input(
            "Mother's qualification",
            min_value=0,
            max_value=100,
            value=19,
            step=1
        )

        father_qual = st.number_input(
            "Father's qualification",
            min_value=0,
            max_value=100,
            value=22,
            step=1
        )

        mother_occ = st.number_input(
            "Mother's occupation",
            min_value=0,
            max_value=200,
            value=10,
            step=1
        )

    with a3:
        father_occ = st.number_input(
            "Father's occupation",
            min_value=0,
            max_value=200,
            value=11,
            step=1
        )

        admission_grade = st.number_input(
            "Admission grade",
            min_value=0.0,
            max_value=200.0,
            value=125.0,
            step=0.1
        )

        age = st.number_input(
            "Age at enrollment",
            min_value=15,
            max_value=80,
            value=20,
            step=1
        )

    # Financial
    st.markdown(
        '<div class="section-header"><h3>Financial & Support</h3></div>',
        unsafe_allow_html=True
    )

    b1, b2 = st.columns(2)

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

    # First semester
    st.markdown(
        '<div class="section-header"><h3>First Semester Performance</h3></div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        s1_enrolled = st.number_input(
            "Enrolled",
            min_value=0,
            max_value=50,
            value=6,
            step=1,
            key="s1_enrolled"
        )

    with c2:
        s1_eval = st.number_input(
            "Evaluations",
            min_value=0,
            max_value=60,
            value=8,
            step=1,
            key="s1_eval"
        )

    with c3:
        s1_approved = st.number_input(
            "Approved",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            key="s1_approved"
        )

    with c4:
        s1_grade = st.number_input(
            "Grade",
            min_value=0.0,
            max_value=20.0,
            value=11.0,
            step=0.1,
            key="s1_grade"
        )

    # Second semester
    st.markdown(
        '<div class="section-header"><h3>Second Semester Performance</h3></div>',
        unsafe_allow_html=True
    )

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        s2_enrolled = st.number_input(
            "Enrolled",
            min_value=0,
            max_value=50,
            value=6,
            step=1,
            key="s2_enrolled"
        )

    with d2:
        s2_eval = st.number_input(
            "Evaluations",
            min_value=0,
            max_value=60,
            value=8,
            step=1,
            key="s2_eval"
        )

    with d3:
        s2_approved = st.number_input(
            "Approved",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            key="s2_approved"
        )

    with d4:
        s2_grade = st.number_input(
            "Grade",
            min_value=0.0,
            max_value=20.0,
            value=11.0,
            step=0.1,
            key="s2_grade"
        )

    # Macroeconomic
    st.markdown(
        '<div class="section-header"><h3>Macroeconomic Context</h3></div>',
        unsafe_allow_html=True
    )

    e1, e2, e3 = st.columns(3)

    with e1:
        unemployment = st.number_input(
            "Unemployment rate (%)",
            value=11.5,
            step=0.1,
            format="%.2f"
        )

    with e2:
        inflation = st.number_input(
            "Inflation rate (%)",
            value=1.2,
            step=0.1,
            format="%.2f"
        )

    with e3:
        gdp = st.number_input(
            "GDP",
            value=1.5,
            step=0.1,
            format="%.2f"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    submitted = st.form_submit_button(
        "🔍  PREDICT STUDENT OUTCOME",
        type="primary",
        use_container_width=True
    )

# ---------------------------------------------------------
# Prediction result
# ---------------------------------------------------------
if submitted:

    input_data = pd.DataFrame([[
        application_mode,
        course,
        prev_grade,
        mother_qual,
        father_qual,
        mother_occ,
        father_occ,
        admission_grade,
        tuition,
        scholarship,
        age,
        s1_enrolled,
        s1_eval,
        s1_approved,
        s1_grade,
        s2_enrolled,
        s2_eval,
        s2_approved,
        s2_grade,
        unemployment,
        inflation,
        gdp
    ]], columns=FEATURES)

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    predicted_class = CLASS_NAMES[prediction]
    confidence = float(np.max(probabilities)) * 100

    st.markdown(
        '<div class="section-header"><h3>Prediction Result</h3></div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1.5])

    with left:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted Academic Outcome</div>
                <div class="result-value">{predicted_class}</div>
                <br>
                <div class="result-label">
                    Model confidence: <strong>{confidence:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:

        probability_df = pd.DataFrame({
            "Outcome": CLASS_NAMES,
            "Probability": probabilities * 100
        })

        fig = px.bar(
            probability_df,
            x="Probability",
            y="Outcome",
            orientation="h",
            text=probability_df["Probability"].map(
                lambda x: f"{x:.1f}%"
            )
        )

        fig.update_traces(
            marker_color="#3182ce",
            textposition="outside"
        )

        fig.update_layout(
            height=300,
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis_title="Probability (%)",
            yaxis_title="",
            xaxis=dict(range=[0, 105]),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(color="#000000"),
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.info(
        "This prediction is a machine-learning estimate and should be "
        "used as decision support, not as the sole basis for decisions "
        "about a student."
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#000000;">
        <strong>Student Outcome Predictor</strong><br>
        XGBoost Machine Learning Model • Academic Project
    </div>
    """,
    unsafe_allow_html=True
)
