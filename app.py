import gradio as gr
import pandas as pd
import joblib



# Load the trained model
model = joblib.load("model.pkl")

# Define prediction function
def predict_stress(Study_Hours, Hobbies_Hours, Sleep_Hours, Social_Interaction_Hours, Physical_Activity_Hours, CGPA):
    data = pd.DataFrame([[Study_Hours, Hobbies_Hours, Sleep_Hours, Social_Interaction_Hours, Physical_Activity_Hours, CGPA]],
                        columns=["Study_Hours", "Hobbies_Hours", "Sleep_Hours", "Social_Interaction_Hours", "Physical_Activity_Hours", "CGPA"])
    result = model.predict(data)
    if(result[0]==0):return f"Predicted Stress Level: High"
    elif( result[0]==1):return f"Predicted Stress Level: Moderate"
    else:return f"Predicted Stress Level:Low "


# Create Gradio interface
interface = gr.Interface(
    fn=predict_stress,
    inputs=[
        gr.Number(label="Study Hours per day"),
        gr.Number(label="Hobbies Hours per day"),
        gr.Number(label="Sleep Hours per day"),
        gr.Number(label="Social Interaction Hours per day"),
        gr.Number(label="Physical Activity Hours per day"),
        gr.Number(label="CGPA"),
         ],
            title="🎓 Student Stress Level Predictor",
        description=" Enter daily activity hours and CGPA to predict stress level",
    outputs="text",
    
)

# Launch app
interface.launch()
