
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# ======================================
# 1. PAGE CONFIGURATION
# ======================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ======================================
# 2. CUSTOM DESIGN
# ======================================

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
}

[data-testid="stSidebar"] {
    background-color: #10243A;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.main-title {
    color: #10243A;
    font-size: 34px;
    font-weight: 800;
}

.subtitle {
    color: #64748B;
    font-size: 16px;
}

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
}

.metric-label {
    color: #64748B;
    font-size: 14px;
}

.metric-value {
    color: #10243A;
    font-size: 27px;
    font-weight: 800;
}

div.stButton > button {
    background-color: #0D9488;
    color: white;
    border-radius: 9px;
    height: 48px;
    border: none;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #0F766E;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ======================================
# 3. LOAD MODEL
# ======================================

@st.cache_resource
def load_model():
    return joblib.load("diabetes_model.pkl")

if not os.path.exists("diabetes_model.pkl"):
    st.error("diabetes_model.pkl is missing.")
    st.stop()

model = load_model()

# ======================================
# 4. SIDEBAR
# ======================================

with st.sidebar:
    st.title("🩺 AI HEALTH")
    st.caption("DIABETES PREDICTION SYSTEM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Home", "Prediction", "About Project"]
    )

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("Dataset: Pima Indians Diabetes")

# ======================================
# 5. HOME PAGE
# ======================================

if page == "Home":

    st.markdown(
        '<div class="main-title">Diabetes Prediction Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine Learning Powered Healthcare Analytics</div>',
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">Dataset Records</div>
        <div class="metric-value">768</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">Input Features</div>
        <div class="metric-value">8</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-label">Algorithms Compared</div>
        <div class="metric-value">6</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.subheader("Welcome to AI Health")

    st.write("""
    This application uses a trained Machine Learning model
    to analyze eight health-related measurements and predict
    a diabetes classification.

    Navigate to the Prediction page to enter measurements
    and view the model output.
    """)

    st.info(
        "This application is developed for educational "
        "and demonstration purposes."
    )

# ======================================
# 6. PREDICTION PAGE
# ======================================

elif page == "Prediction":

    st.markdown(
        '<div class="main-title">Diabetes Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Enter the health measurements below</div>',
        unsafe_allow_html=True
    )

    st.write("")

    with st.form("prediction_form"):

        st.subheader("Health Measurements")

        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=0
            )

            glucose = st.number_input(
                "Glucose Level",
                min_value=1,
                max_value=300,
                value=120
            )

            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=1,
                max_value=200,
                value=70
            )

            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0,
                max_value=100,
                value=20
            )

        with col2:
            insulin = st.number_input(
                "Insulin Level",
                min_value=0,
                max_value=1000,
                value=80
            )

            bmi = st.number_input(
                "BMI",
                min_value=1.0,
                max_value=100.0,
                value=25.0
            )

            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=3.0,
                value=0.5
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=30
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    # ==================================
    # 7. DISPLAY PREDICTION
    # ==================================

    if submitted:

        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }])

        prediction = model.predict(input_data)[0]

        probabilities = model.predict_proba(input_data)[0]

        non_diabetic = float(probabilities[0] * 100)
        diabetic = float(probabilities[1] * 100)

        st.divider()
        st.subheader("Prediction Results")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-label">NON-DIABETIC SCORE</div>
            <div class="metric-value">{non_diabetic:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div class="metric-card">
            <div class="metric-label">DIABETIC SCORE</div>
            <div class="metric-value">{diabetic:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        if prediction == 1:
            st.error("Model Prediction: Diabetic Class (1)")
        else:
            st.success("Model Prediction: Non-diabetic Class (0)")

        # Probability chart
        fig = go.Figure(
            data=[go.Pie(
                labels=["Non-diabetic", "Diabetic"],
                values=[non_diabetic, diabetic],
                hole=0.65,
                marker=dict(
                    colors=["#0D9488", "#F59E0B"]
                )
            )]
        )

        fig.update_layout(
            title="Model Probability Distribution",
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Probabilities represent the model's estimated "
            "class probabilities, not calibrated medical risks."
        )

        st.warning(
            "This is an educational Machine Learning prediction, "
            "not a medical diagnosis. Consult a qualified "
            "healthcare professional for medical assessment."
        )

# ======================================
# 8. ABOUT PROJECT
# ======================================

elif page == "About Project":

    st.markdown(
        '<div class="main-title">About This Project</div>',
        unsafe_allow_html=True
    )

    st.subheader("Project Objective")

    st.write("""
    To develop a Machine Learning-based system that analyzes
    health measurements and predicts diabetes classification.
    """)

    st.subheader("Dataset Information")

    st.write("""
    Dataset: Pima Indians Diabetes Dataset

    Total Records: 768

    Input Features: 8

    Target Variable: Outcome
    """)

    st.subheader("Algorithms Used")

    algorithms = [
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "XGBoost"
    ]

    for item in algorithms:
        st.write("•", item)

    st.subheader("Technologies Used")

    st.write("""
    Python, Pandas, NumPy, Scikit-learn, XGBoost,
    Streamlit, Plotly and Joblib.
    """)

    st.info(
        "Developed as an educational Machine Learning project."
    )

# ======================================
# 9. FOOTER
# ======================================

st.divider()

st.caption(
    "AI Health | Diabetes Prediction Using Machine Learning"
)
