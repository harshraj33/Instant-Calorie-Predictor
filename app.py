
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.express as px

# -------------------------------
# Load the model
# -------------------------------
MODEL_PATH = "calories_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please train and save it first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(page_title="🔥 Calories Burn Predictor", page_icon="🔥", layout="wide")

# Custom CSS for a clean, modern look
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

st.title("🔥 Calories Burn Prediction App")
st.write("### Estimate your calories burned based on workout and body stats.")

# -------------------------------
# Input Section (2-column layout)
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["male", "female"])
    age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

with col2:
    duration = st.number_input("Workout Duration (minutes)", min_value=1, max_value=300, value=60)
    heart_rate = st.number_input("Average Heart Rate (bpm)", min_value=60, max_value=200, value=120)
    body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

# -------------------------------
# Prediction Section
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

    # Sample comparison data
    sample_data = pd.DataFrame({
        "Activity": ["Walking (30 min)", "Jogging (30 min)", "Cycling (30 min)", "Your Workout"],
        "Calories Burned": [120, 240, 180, prediction]
    })

    fig = px.bar(
        sample_data,
        x="Activity",
        y="Calories Burned",
        color="Activity",
        color_discrete_sequence=px.colors.sequential.Viridis,
        text="Calories Burned",
        title="Calories Burned Comparison Chart"
    )

    fig.update_traces(texttemplate='%{text:.0f} kcal', textposition='outside')
    fig.update_layout(showlegend=False, yaxis_title="Calories Burned (kcal)")

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("💻 Built with Streamlit | 🤖 Model: Random Forest Regressor")
