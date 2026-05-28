import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load Model
model = pickle.load(open('student_model.pkl', 'rb'))
encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Custom CSS
st.markdown("""
    <style>

    .main {
        background-color: #0f172a;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #cbd5e1;
        margin-bottom: 40px;
    }

    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-size: 18px;
        border-radius: 12px;
        padding: 12px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: scale(1.02);
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 30px;
    }

    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Predict academic performance using Machine Learning</div>',
    unsafe_allow_html=True
)

# Input Section
st.subheader("📋 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance Percentage", 0, 100, 75)
    study_hours = st.slider("Study Hours Per Day", 0, 12, 4)
    internal_marks = st.slider("Internal Marks", 0, 100, 70)

with col2:
    sleep_hours = st.slider("Sleep Hours", 0, 12, 6)
    assignments_completed = st.slider("Assignments Completed", 0, 10, 7)

st.write("")

# Prediction Button
if st.button("🚀 Predict Performance"):

    input_data = np.array([[
        attendance,
        study_hours,
        internal_marks,
        sleep_hours,
        assignments_completed
    ]])

    prediction = model.predict(input_data)
    result = encoder.inverse_transform(prediction)

    # LOW RISK
    if result[0] == "Low Risk":

        st.markdown(f"""
        <div class="result-box" style="
            background-color:#052e16;
            color:#4ade80;
            border:2px solid #22c55e;">
            ✅ LOW RISK
            <br><br>
            Student is likely to perform very well academically.
        </div>
        """, unsafe_allow_html=True)

        st.balloons()

    # MEDIUM RISK
    elif result[0] == "Medium Risk":

        st.markdown(f"""
        <div class="result-box" style="
            background-color:#3f2f00;
            color:#facc15;
            border:2px solid #eab308;">
            ⚠️ MEDIUM RISK
            <br><br>
            Student may require moderate improvement.
        </div>
        """, unsafe_allow_html=True)

    # HIGH RISK
    else:

        st.markdown(f"""
        <div class="result-box" style="
            background-color:#450a0a;
            color:#f87171;
            border:2px solid #ef4444;">
            ❌ HIGH RISK
            <br><br>
            Student is at high academic risk.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.write("")
st.write("---")
st.caption("Built using Python, Scikit-learn and Streamlit")