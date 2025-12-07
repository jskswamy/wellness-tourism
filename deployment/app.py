"""
Wellness Tourism Prediction - Streamlit App

Interactive web application for predicting customer purchases of Wellness Tourism Package.
Loads model directly from Hugging Face Model Hub using hf_hub_download().
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import plotly.graph_objects as go

# =============================================================================
# Configuration
# =============================================================================

HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
MODEL_REPO_ID = f"{HF_USERNAME}/wellness-tourism-model"
MODEL_FILENAME = "wellness_tourism_model.joblib"

st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Compact CSS
st.markdown("""
<style>
    .block-container { padding-top: 2.5rem; padding-bottom: 0; max-width: 100%; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
    .stSelectbox, .stNumberInput, .stRadio { margin-bottom: 0; }
    div[data-testid="stNumberInput"] > div { margin-bottom: 0; }
    .main-header { font-size: 1.8rem; font-weight: bold; color: #2E86AB; text-align: center; margin-bottom: 0.5rem; }
    .prediction-box { padding: 1rem; border-radius: 8px; text-align: center; margin: 0.5rem 0; }
    .purchase { background-color: #d4edda; border: 2px solid #28a745; }
    .no-purchase { background-color: #f8d7da; border: 2px solid #dc3545; }
    .section-header { font-size: 1rem; font-weight: 600; margin: 0.5rem 0; color: #333; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Download and load model from Hugging Face Model Hub."""
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        return joblib.load(model_path), None
    except Exception as e:
        return None, str(e)


def prepare_input(data: dict) -> pd.DataFrame:
    """Prepare input data with feature engineering."""
    df = pd.DataFrame([data])

    def create_age_group(age):
        if age < 25: return "Young"
        elif age < 35: return "Young Adult"
        elif age < 45: return "Middle Age"
        elif age < 55: return "Senior"
        else: return "Elder"

    def create_income_group(income):
        if income < 15000: return "Low"
        elif income < 25000: return "Medium"
        elif income < 35000: return "High"
        else: return "Very High"

    df["AgeGroup"] = df["Age"].apply(create_age_group)
    df["IncomeGroup"] = df["MonthlyIncome"].apply(create_income_group)
    df["TotalTravelers"] = df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]
    return df


def main():
    st.markdown('<div class="main-header">🏝️ Wellness Tourism Package Predictor</div>', unsafe_allow_html=True)

    model, error = load_model()
    if error:
        st.error(f"Failed to load model: {error}")
        return

    # Sidebar - About
    with st.sidebar:
        st.markdown(f"""
        ### 🏝️ About

        **Business Context:**
        Predict which customers are likely to purchase the Wellness Tourism Package.

        **Key Predictive Factors:**
        - 💼 Occupation & Income
        - 🎯 Pitch satisfaction
        - 🛫 Travel history
        - 👥 Family size

        **Model:** Ensemble classifier trained on 4,888 records

        [View Model Card](https://huggingface.co/{MODEL_REPO_ID})
        """)

    # 3 columns layout (no nested columns to avoid Streamlit error)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p class="section-header">📋 Demographics</p>', unsafe_allow_html=True)
        age = st.number_input("Age", 18, 80, 35)
        monthly_income = st.number_input("Income ($)", 5000, 100000, 25000, step=5000)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        passport = st.radio("Passport", ["Yes", "No"], horizontal=True)
        passport = 1 if passport == "Yes" else 0
        own_car = st.radio("Own Car", ["Yes", "No"], horizontal=True)
        own_car = 1 if own_car == "Yes" else 0

    with col2:
        st.markdown('<p class="section-header">💼 Trip & Sales</p>', unsafe_allow_html=True)
        type_of_contact = st.selectbox("Contact Type", ["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox("Product", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        num_persons = st.number_input("Persons Visiting", 1, 10, 3)
        num_children = st.number_input("Children (<5 yrs)", 0, 5, 0)
        num_trips = st.number_input("Trips/Year", 0, 20, 3)
        preferred_star = st.selectbox("Hotel Rating", [3.0, 3.5, 4.0, 4.5, 5.0], index=2)
        duration_of_pitch = st.number_input("Pitch Duration (min)", 1, 60, 15)
        num_followups = st.number_input("Follow-ups", 0, 10, 4)
        pitch_satisfaction = st.select_slider("Satisfaction", options=[1, 2, 3, 4, 5], value=4)

    # Prepare input and predict (auto on any change)
    input_data = {
        "Age": float(age),
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": float(duration_of_pitch),
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": num_persons,
        "NumberOfFollowups": float(num_followups),
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": float(preferred_star),
        "MaritalStatus": marital_status,
        "NumberOfTrips": float(num_trips),
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": float(num_children),
        "Designation": designation,
        "MonthlyIncome": float(monthly_income)
    }

    df = prepare_input(input_data)

    with col3:
        st.markdown('<p class="section-header">🎯 Prediction</p>', unsafe_allow_html=True)
        try:
            prediction = model.predict(df)[0]
            probability = model.predict_proba(df)[0]
            gauge_value = probability[1] - probability[0]

            # Prediction result box
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box purchase">
                    <h2>✅ Will Purchase</h2>
                    <h3>{probability[1]*100:.1f}% confidence</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box no-purchase">
                    <h2>❌ Won't Purchase</h2>
                    <h3>{probability[0]*100:.1f}% confidence</h3>
                </div>
                """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge",
                value=gauge_value,
                gauge={
                    "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "darkgray",
                             "ticktext": ["No", "", "", "", "Yes"],
                             "tickvals": [-1, -0.5, 0, 0.5, 1]},
                    "bar": {"color": "darkblue", "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [-1, -0.5], "color": "#ff4444"},
                        {"range": [-0.5, 0], "color": "#ffaaaa"},
                        {"range": [0, 0.5], "color": "#aaffaa"},
                        {"range": [0.5, 1], "color": "#44ff44"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": gauge_value}
                }
            ))
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")


if __name__ == "__main__":
    main()
