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

# =============================================================================
# Configuration
# =============================================================================

# Get HF username from environment or use default
HF_USERNAME = os.environ.get("HF_USERNAME", "your-username")
MODEL_REPO_ID = f"{HF_USERNAME}/wellness-tourism-model"
MODEL_FILENAME = "wellness_tourism_model.joblib"

# Page configuration
st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .purchase {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .no-purchase {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Model Loading
# =============================================================================

@st.cache_resource
def load_model():
    """Download and load model from Hugging Face Model Hub."""
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME
        )
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


# =============================================================================
# Feature Engineering
# =============================================================================

def prepare_input(data: dict) -> pd.DataFrame:
    """Prepare input data with feature engineering."""
    df = pd.DataFrame([data])

    # Create age groups
    def create_age_group(age):
        if age < 25:
            return "Young"
        elif age < 35:
            return "Young Adult"
        elif age < 45:
            return "Middle Age"
        elif age < 55:
            return "Senior"
        else:
            return "Elder"

    df["AgeGroup"] = df["Age"].apply(create_age_group)

    # Create income groups
    def create_income_group(income):
        if income < 15000:
            return "Low"
        elif income < 25000:
            return "Medium"
        elif income < 35000:
            return "High"
        else:
            return "Very High"

    df["IncomeGroup"] = df["MonthlyIncome"].apply(create_income_group)

    # Total travelers
    df["TotalTravelers"] = df["NumberOfPersonVisiting"] + df["NumberOfChildrenVisiting"]

    return df


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Header
    st.markdown('<div class="main-header">🏝️ Wellness Tourism Package Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict customer likelihood to purchase the Wellness Tourism Package</div>', unsafe_allow_html=True)

    # Load model
    model, error = load_model()

    if error:
        st.error(f"Failed to load model: {error}")
        st.info(f"Expected model at: {MODEL_REPO_ID}/{MODEL_FILENAME}")
        return

    st.success("Model loaded successfully from Hugging Face!")

    # Sidebar
    with st.sidebar:
        st.title("About")
        st.markdown(f"""
        ### 🏝️ Wellness Tourism Predictor

        **Business Context:**
        "Visit with Us" travel company wants to identify customers
        likely to purchase their new Wellness Tourism Package.

        **How it works:**
        Enter customer details and the model predicts purchase likelihood
        based on patterns learned from historical data.

        ---

        **Key Predictive Factors:**
        - 💼 Occupation & Income level
        - 🎯 Pitch satisfaction score
        - 🛫 Travel history & Passport status
        - 👥 Family size (travelers + children)

        ---

        **Model Details:**
        - 🤖 Algorithm: Ensemble classifier
        - 📊 Dataset: 4,888 customer records
        - 🎯 Optimized for: F1 Score
        - 🔗 [View Model Card](https://huggingface.co/{MODEL_REPO_ID})

        ---

        *Built with Streamlit + scikit-learn*
        """)

    # Create form with two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Customer Demographics")

        age = st.slider("Age", min_value=18, max_value=80, value=35, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income ($)", min_value=5000, max_value=100000, value=25000, step=1000)
        city_tier = st.selectbox("City Tier", [1, 2, 3], help="1 = Metropolitan, 2 = Urban, 3 = Semi-Urban")
        passport = st.radio("Has Passport?", ["Yes", "No"], horizontal=True)
        passport = 1 if passport == "Yes" else 0
        own_car = st.radio("Owns Car?", ["Yes", "No"], horizontal=True)
        own_car = 1 if own_car == "Yes" else 0

    with col2:
        st.subheader("💼 Trip & Interaction Details")

        type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        num_persons = st.slider("Number of Persons Visiting", min_value=1, max_value=10, value=3)
        num_children = st.slider("Number of Children (below 5)", min_value=0, max_value=5, value=0)
        num_trips = st.slider("Average Annual Trips", min_value=0, max_value=20, value=3)
        preferred_star = st.slider("Preferred Hotel Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.5)

        st.subheader("📞 Sales Interaction")
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        duration_of_pitch = st.slider("Duration of Pitch (minutes)", min_value=1, max_value=60, value=15)
        num_followups = st.slider("Number of Follow-ups", min_value=0, max_value=10, value=4)
        pitch_satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=4)

    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Purchase Likelihood", type="primary", use_container_width=True):
        # Prepare input data
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

        # Prepare features
        df = prepare_input(input_data)

        # Make prediction
        with st.spinner("Making prediction..."):
            try:
                prediction = model.predict(df)[0]
                probability = model.predict_proba(df)[0]

                # Display result
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="prediction-box purchase">
                            <h1>✅ Will Purchase</h1>
                            <h2>Confidence: {probability[1]*100:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("**High Purchase Intent Detected!** Prioritize this customer for follow-up.")
                    else:
                        st.markdown(f"""
                        <div class="prediction-box no-purchase">
                            <h1>❌ Will Not Purchase</h1>
                            <h2>Confidence: {probability[0]*100:.1f}%</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        st.warning("**Low Purchase Intent.** Consider adjusting the product pitch.")

                # Show gauge chart for prediction confidence
                st.subheader("📊 Purchase Likelihood Gauge")

                # Calculate gauge value: -1 (No Purchase) to +1 (Purchase)
                # Map probability to gauge: if Purchase prob > 0.5, positive; else negative
                gauge_value = probability[1] - probability[0]  # Range: -1 to +1

                import plotly.graph_objects as go

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=gauge_value,
                    number={"suffix": "", "font": {"size": 40}},
                    gauge={
                        "axis": {"range": [-1, 1], "tickwidth": 1, "tickcolor": "darkgray",
                                 "ticktext": ["No Purchase", "", "Neutral", "", "Purchase"],
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
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": gauge_value
                        }
                    },
                    title={"text": f"<b>{'Purchase' if gauge_value > 0 else 'No Purchase'}</b><br>"
                                   f"<span style='font-size:0.8em'>Confidence: {max(probability)*100:.1f}%</span>",
                           "font": {"size": 16}}
                ))

                fig.update_layout(
                    height=300,
                    margin=dict(l=30, r=30, t=80, b=30),
                    font={"family": "Arial"}
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")


if __name__ == "__main__":
    main()
