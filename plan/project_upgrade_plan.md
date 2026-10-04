# 🚀 Stress Level Predictor — Full Project Upgrade Plan

## 📊 Dataset Analysis (Done)

| Property | Value |
|---|---|
| Rows | 2,000 |
| Features | Study_Hours, Hobbies_Hours, Sleep_Hours, Social_Interaction_Hours, Physical_Activity_Hours, CGPA |
| Target | Stress_Level (High: 1029, Moderate: 674, Low: 297) |
| Null Values | None |
| Class Imbalance | Yes — High (51%) >> Low (15%) |

> [!WARNING]
> **The dataset is synthetic/too clean.** Every tree-based model (Random Forest, XGBoost) gets 100% accuracy even with 5-fold cross-validation. This will look suspicious to any evaluator. You MUST either replace it or add noise/real-world features.

---

## 🤖 Model Comparison Results (Actual Numbers)

| Model | Train Acc | Test Acc | CV (5-fold) | Verdict |
|---|---|---|---|---|
| Logistic Regression | 82.56% | **84.50%** | 82.40% | ✅ Best honest model |
| Random Forest (no limit) | 100% | 100% | 100% | ❌ Overfitting / Synthetic data |
| Random Forest (max_depth=5) | 100% | 100% | 100% | ❌ Still suspicious |
| XGBoost | 100% | 100% | 100% | ❌ Overfitting / Synthetic data |
| KNN | 95.37% | 91.75% | 90.85% | ⚠️ Decent but no explainability |
| SVM | 93.31% | **92.75%** | 92.30% | ✅ Good generalizer |

> [!IMPORTANT]
> **Best model to use: SVM (RBF kernel)** — 92.75% test accuracy, generalizes well, no overfitting signs. Use this for your final model. Logistic Regression is your honest baseline. Show this full table in your README to prove you tested and compared properly.

---

## 🏗️ Recommended Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Frontend** | React.js + Vite | Fast, modern, component-based |
| **Styling** | Tailwind CSS | Rapid premium UI |
| **Backend API** | FastAPI (Python) | Best for ML APIs, async, auto docs |
| **Database** | Supabase (PostgreSQL) | Free tier, real-time, easy to set up |
| **ML Model** | SVM + SHAP | Best real accuracy + explainability |
| **Chatbot** | Rule-based + OpenAI API | Contextual stress tips |
| **Deployment** | Vercel (Frontend) + Hugging Face Spaces (Backend) | Free, fast |

---

## 🔧 What To Build — Feature List

### Core Features
- [ ] React dashboard with animated stress meter (gauge chart)
- [ ] Sliders for all 6 input features
- [ ] FastAPI backend that loads SVM model and returns prediction + SHAP values
- [ ] SHAP explanation bar chart rendered in React (recharts library)
- [ ] Supabase database to **save each prediction** with timestamp

### Strong Extras (impress evaluators)
- [ ] **Model Comparison Page** — show the table above visually with charts
- [ ] **EDA Page** — show dataset distributions, correlation heatmap, class balance pie chart
- [ ] **Stress Tips Chatbot** — rule-based chatbot that gives advice based on predicted stress level (e.g. if High Stress → "Try reducing study hours to under 6hrs, improve sleep to 8hrs")
- [ ] **History Dashboard** — show past predictions stored in Supabase (line chart over time)
- [ ] **README with full analysis** — model comparison table, why SVM was chosen, dataset observations, limitations

---

## 📁 Recommended Folder Structure

```
stress-predictor/
├── frontend/                  # React + Vite app
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx       # Main prediction page
│   │   │   ├── Analysis.jsx   # EDA + model comparison
│   │   │   ├── History.jsx    # Past predictions from Supabase
│   │   │   └── Chatbot.jsx    # Stress tips chatbot
│   │   ├── components/
│   │   │   ├── StressMeter.jsx
│   │   │   ├── ShapChart.jsx
│   │   │   └── Navbar.jsx
│   │   └── App.jsx
│   └── package.json
│
├── backend/                   # FastAPI Python app
│   ├── main.py                # API routes
│   ├── model/
│   │   ├── train.py           # Training script (SVM)
│   │   └── model.pkl          # Saved SVM model
│   ├── routers/
│   │   ├── predict.py         # POST /predict
│   │   └── history.py        # GET /history
│   └── requirements.txt
│
├── notebooks/
│   └── EDA_and_Model_Comparison.ipynb
│
├── README.md                  # Detailed analysis
└── docker-compose.yml
```

---

## 📝 README Structure (To Impress Evaluators)

```
1. Project Overview
2. Dataset Analysis
   - Shape, features, class imbalance
   - EDA insights (correlation, distributions)
3. Model Comparison Table (all 6 models, with results)
4. Why SVM Was Chosen (explanation)
5. Why We Rejected Random Forest / XGBoost (overfitting/synthetic data evidence)
6. SHAP Explainability — what it is and why we added it
7. Tech Stack and Architecture Diagram
8. How to Run Locally
9. API Documentation
10. Future Improvements
```

---

## 🤖 Chatbot Design (Simple but Effective)

No need for expensive AI — a **rule-based chatbot** is fine and actually more reliable:

```python
if stress == "High":
  tips = [
    "Reduce study hours below 6/day",
    "Aim for 8 hours of sleep",
    "Add 30 mins of physical activity",
    "Increase social interaction"
  ]
elif stress == "Moderate":
  tips = ["Maintain current balance", "Add 1 hobby hour/day"]
else:
  tips = ["Great balance! Keep it up."]
```

You can wrap this in a React chat UI that feels like a real chatbot.

---

## 🎯 Instructions for Your Developer (OpenCode)

Tell them exactly this:

> Build a full-stack stress level prediction web app with:
> - **Frontend:** React + Vite + Tailwind CSS. Pages: Home (prediction), Analysis (EDA charts), History (past predictions), Chatbot (stress tips)
> - **Backend:** FastAPI in Python. Routes: POST /predict (takes 6 features, returns stress label + SHAP values), GET /history (returns past predictions from DB)
> - **Database:** Supabase (PostgreSQL). Table: `predictions` with columns: id, study_hours, hobbies_hours, sleep_hours, social_hours, physical_hours, cgpa, stress_level, timestamp
> - **ML Model:** SVM with RBF kernel, trained on student_lifestyle_dataset2.csv (6 features). Save with joblib. Use SHAP KernelExplainer for explanations.
> - **Chatbot:** Rule-based, shows tips in a chat UI based on predicted stress level
> - **Design:** Dark mode, glassmorphism cards, smooth animations, stress meter gauge on result

