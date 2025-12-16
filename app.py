import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.express as px
import zipfile

# -------------------------------
# Load model from ZIP
# -------------------------------
MODEL_ZIP = "calories_model.zip"
MODEL_PATH = "calories_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with zipfile.ZipFile(MODEL_ZIP, "r") as zip_ref:
            zip_ref.extractall(".")
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="🔥 Calories Burn Predictor",
    page_icon="🔥",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #ffb703;
}
.stButton button {
    background-color: #ffb703;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 0.5rem 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🔥 Calories Burnt Prediction App")
st.write("### Estimate your calories burned based on workout and body stats.")

# -------------------------------
# Input Section
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["male", "female"])
    age = st.number_input("Age (years)", 10, 100, 25)
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)

with col2:
    duration = st.number_input("Workout Duration (minutes)", 1, 300, 60)
    heart_rate = st.number_input("Average Heart Rate (bpm)", 60, 200, 120)
    body_temp = st.number_input("Body Temperature (°C)", 35.0, 42.0, 37.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🚀 Predict Calories Burned"):
    input_data = pd.DataFrame({
        "Gender": [gender.lower()],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "Duration": [duration],
        "Heart_Rate": [heart_rate],
        "Body_Temp": [body_temp],
    })

    prediction = model.predict(input_data)[0]

    st.success(f"🔥 **Estimated Calories Burned:** {prediction:.2f} kcal")

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📊 Comparison with Common Activities")

    sample_data = pd.DataFrame({
        "Activity": [
            "Walking (30 min)",
            "Jogging (30 min)",
            "Cycling (30 min)",
            "Your Workout"
        ],
        "Calories Burned": [120, 240, 180, prediction]
    })

    fig = px.bar(
        sample_data,
        x="Activity",
        y="Calories Burned",
        text="Calories Burned",
        title="Calories Burned Comparison Chart"
    )

    fig.update_traces(texttemplate='%{text:.0f} kcal', textposition='outside')
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("💻 Built with Streamlit | 🤖 Model: Random Forest Regressor")

