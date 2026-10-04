import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Zenith: Student Stress Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure the model is trained.")
    st.stop()

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: None;
        padding: 10px 24px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)


# --- SIDEBAR INPUTS ---
st.sidebar.title("🎓 Zenith: Student Input")
st.sidebar.markdown("Enter student lifestyle metrics below to generate a stress prediction and AI explanation.")

study_hours = st.sidebar.slider("Study Hours (Daily)", 0.0, 12.0, 5.0, 0.5)
hobbies_hours = st.sidebar.slider("Hobbies Hours (Daily)", 0.0, 10.0, 2.0, 0.5)
sleep_hours = st.sidebar.slider("Sleep Hours (Daily)", 0.0, 12.0, 7.0, 0.5)
social_hours = st.sidebar.slider("Social Interaction (Daily)", 0.0, 10.0, 2.0, 0.5)
physical_hours = st.sidebar.slider("Physical Activity (Daily)", 0.0, 10.0, 1.0, 0.5)
cgpa = st.sidebar.slider("Current CGPA (0.0 - 4.0)", 0.0, 4.0, 3.0, 0.1)

# Create DataFrame
input_df = pd.DataFrame([[
    study_hours, hobbies_hours, sleep_hours, social_hours, physical_hours, cgpa
]], columns=[
    "Study_Hours", "Hobbies_Hours", "Sleep_Hours", 
    "Social_Interaction_Hours", "Physical_Activity_Hours", "CGPA"
])

# --- MAIN DASHBOARD ---
st.title("Zenith: Student Well-being & Stress Analytics")
st.markdown("Welcome to Zenith. This dashboard utilizes Machine Learning to predict student stress levels and provides an **Explainable AI (SHAP)** breakdown of the contributing factors.")

st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Metrics")
    st.markdown(f"**📖 Study:** {study_hours} hrs")
    st.markdown(f"**🎨 Hobbies:** {hobbies_hours} hrs")
    st.markdown(f"**🛌 Sleep:** {sleep_hours} hrs")
    st.markdown(f"**🗣️ Social:** {social_hours} hrs")
    st.markdown(f"**🏃 Physical:** {physical_hours} hrs")
    st.markdown(f"**🎓 CGPA:** {cgpa}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Generate Prediction & Analysis")

with col2:
    if predict_btn:
        with st.spinner("Analyzing data and generating SHAP explanations..."):
            # Prediction
            pred_num = model.predict(input_df)[0]
            
            # Mapping target back ('High': 0, 'Moderate': 1, 'Low': 2)
            mapping = {0: ("HIGH STRESS", "#FF4B4B"), 
                       1: ("MODERATE STRESS", "#FFA500"), 
                       2: ("LOW STRESS", "#4CAF50")}
            
            pred_label, color = mapping[pred_num]
            
            # Display Result
            st.markdown(f"""
                <div style="background-color: {color}; padding: 20px; border-radius: 10px; color: white; text-align: center;">
                    <h2 style="color: white; margin: 0;">Predicted Status: {pred_label}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Explainable AI (SHAP) Breakdown")
            st.markdown(f"This chart explains why the model predicted **{pred_label}**. It shows how much each feature contributed to this specific prediction outcome.")
            
            # SHAP Explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # For RandomForest multiclass, shap_values format depends on the shap version.
            # We want to explain the class that was predicted.
            if isinstance(shap_values, list):
                class_shap_values = shap_values[pred_num][0]
            else:
                # Modern SHAP (0.42+) returns a 3D numpy array: (n_samples, n_features, n_classes)
                class_shap_values = shap_values[0, :, pred_num]
            
            # Plotting a clean horizontal bar chart for feature importance for this single prediction
            fig, ax = plt.subplots(figsize=(8, 4))
            features = input_df.columns
            y_pos = np.arange(len(features))
            
            # Colors: red for positive push towards the class, blue for negative push
            colors = ['#FF4B4B' if val > 0 else '#1E90FF' for val in class_shap_values]
            
            ax.barh(y_pos, class_shap_values, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel(f'SHAP Value (Impact on {pred_label})')
            ax.set_title('Feature Contributions')
            
            # Add value labels
            for i, v in enumerate(class_shap_values):
                ax.text(v, i, f" {v:+.2f}", va='center', color='black', fontweight='bold')
                
            st.pyplot(fig)
            
    else:
        st.info("👈 Adjust the student metrics in the sidebar and click 'Generate Prediction & Analysis' to view the AI breakdown.")
