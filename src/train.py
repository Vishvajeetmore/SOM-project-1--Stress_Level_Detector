import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def train_model():
    print("Loading data...")
    df = pd.read_csv("student_lifestyle_dataset2.csv")
    
    # Feature columns
    features = [
        "Study_Hours", "Hobbies_Hours", "Sleep_Hours", 
        "Social_Interaction_Hours", "Physical_Activity_Hours", "CGPA"
    ]
    
    X = df[features]
    y = df["Stress_Level"]
    
    # Map target variables explicitly to ensure consistency
    # 0: High, 1: Moderate, 2: Low
    target_mapping = {'High': 0, 'Moderate': 1, 'Low': 2}
    y = y.map(target_mapping)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    # max_depth=5 to prevent the overfitting mentioned in the original README
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=['High', 'Moderate', 'Low']))
    
    print("Saving model to model.pkl...")
    joblib.dump(model, "model.pkl")
    print("Training complete!")

if __name__ == "__main__":
    train_model()
