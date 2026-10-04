# 🧠 Zenith — Student Stress Level Predictor

> An **explainable, full‑stack ML web application** that predicts a student's stress level (High / Moderate / Low) from six lifestyle features using a **Support Vector Machine (RBF kernel)** — Test Accuracy: **94.75 %** | 5-Fold CV: **95.45 %**

---

## 📌 Table of Contents
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Dataset & EDA](#-dataset--eda)
- [ML Pipeline — All Steps](#-ml-pipeline--all-steps)
- [Model Comparison & Why SVM](#-model-comparison--why-svm)
- [Explainability (SHAP)](#-explainability-shap)
- [How to Run Locally](#-how-to-run-locally)
- [API Docs](#-api-docs)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)

---

## 🌐 Demo

> **Repo:** [https://github.com/Vishvajeetmore/SOM-project-1--Stress_Level_Detector](https://github.com/Vishvajeetmore/SOM-project-1--Stress_Level_Detector)

**Features:**
- 🎚️ Predict stress level via interactive sliders (6 features)
- 📊 Real-time SHAP explanation — see *why* the model predicted what it did
- 📜 History page — every prediction stored in a database with timestamp
- 🤖 Rule-based stress advisor chatbot giving personalised tips
- 📈 ML Pipeline page — visual walkthrough of every ML step

---

## 🏗️ Architecture

```
React (Vite)  ──[/api/predict POST]──►  FastAPI Backend
                                              │
                                    sklearn Pipeline
                                    (StandardScaler + SVM)
                                              │
                                         model.pkl
                                              │
                                       Supabase DB
                                    (predictions table)
```

| Layer | Technology |
|---|---|
| Frontend | React + Vite, Recharts, Lucide-React, CSS Glassmorphism |
| Backend | FastAPI (Python), Uvicorn |
| ML Model | scikit-learn Pipeline: StandardScaler + SVC (RBF) |
| Explainability | SHAP KernelExplainer |
| Database | Supabase (PostgreSQL) |
| Container | Docker (python:3.11-slim) |

---

## 📂 Dataset & EDA

**Source:** Kaggle — *Student Lifestyle Dataset*  
**Shape:** 2,000 rows × 8 columns (synthetic but realistic distribution)

### Features Used

| Feature | Description | Range |
|---|---|---|
| `Study_Hours` | Hours studying per day | 0–24 |
| `Hobbies_Hours` | Time on hobbies/extracurriculars | 0–24 |
| `Sleep_Hours` | Sleep duration per night | 0–24 |
| `Social_Interaction_Hours` | Time with friends/family | 0–24 |
| `Physical_Activity_Hours` | Exercise time per day | 0–24 |
| `CGPA` | Cumulative GPA | 0–4.0 |

### Class Distribution (Imbalanced!)

| Stress Level | Count | % |
|---|---|---|
| High | 1,029 | 51.5 % |
| Moderate | 674 | 33.7 % |
| Low | 297 | 14.8 % |

> ⚠️ **Imbalance handled** using `class_weight='balanced'` in SVM — otherwise the model ignores the "Low" minority class.

### Key EDA Insights
- **Sleep hours** are strongly **negatively** correlated with stress (r ≈ −0.71)
- **Study hours** are strongly **positively** correlated with stress (r ≈ +0.79)
- **Physical activity** reduces stress — High-stress students average only 0.8 hrs/day vs 2.1 hrs for Low-stress
- Zero null values — dataset is clean but synthetic (explains 100 % accuracy in tree models)

---

## 🔬 ML Pipeline — All Steps

Every step below is also **visualised interactively** in the app's "ML Pipeline" page.

| Step | Action | Tool Used |
|---|---|---|
| 1️⃣ Data Collection | Downloaded from Kaggle, loaded with Pandas | `pandas.read_csv` |
| 2️⃣ EDA | Distribution plots, correlation heatmap, box plots | `matplotlib`, `seaborn` |
| 3️⃣ Data Cleaning | Checked nulls (none found), removed `Student_ID` column | `df.isnull().sum()` |
| 4️⃣ Class Imbalance | Identified imbalance, applied `class_weight='balanced'` | SVC parameter |
| 5️⃣ Feature Engineering | Selected 6 meaningful features, dropped ID | Manual selection |
| 6️⃣ Preprocessing | `StandardScaler` — normalise to mean=0, std=1 | `sklearn.preprocessing` |
| 7️⃣ Train/Test Split | 80 % train / 20 % test, stratified by class | `train_test_split` |
| 8️⃣ Model Training | SVM with RBF kernel, C=1.0, probability=True | `sklearn.svm.SVC` |
| 9️⃣ Model Comparison | Benchmarked 6 models, recorded accuracy + CV | See table below |
| 🔟 Evaluation | Classification report, confusion matrix | `sklearn.metrics` |
| 1️⃣1️⃣ Explainability | SHAP values per prediction | `shap.KernelExplainer` |
| 1️⃣2️⃣ Serialisation | Saved `Pipeline(scaler+svm)` as `model.pkl` | `joblib.dump` |
| 1️⃣3️⃣ Deployment | FastAPI serves predictions via REST API | `uvicorn`, `FastAPI` |

> **Why wrap scaler + model in a Pipeline?**  
> `sklearn.Pipeline` guarantees the same scaler transform is applied at inference time automatically — no risk of forgetting to scale input data.

---

## 📊 Model Comparison & Why SVM

I benchmarked **6 models** on the same 80/20 train-test split with 5-fold cross-validation:

| # | Model | Test Acc | 5-Fold CV | Verdict |
|---|---|---|---|---|
| 1 | Logistic Regression | 84.50 % | 82.40 % | ✅ Honest baseline, underfits |
| 2 | Random Forest (no depth limit) | **100 %** | **100 %** | ❌ Overfitting — memorised synthetic data |
| 3 | Random Forest (max_depth=5) | **100 %** | **100 %** | ❌ Still suspicious |
| 4 | XGBoost | **100 %** | **100 %** | ❌ Overfitting |
| 5 | KNN (k=5) | 91.75 % | 90.85 % | ⚠️ Good but no explainability |
| 6 | **SVM (RBF kernel)** ← **Chosen** | **94.75 %** | **95.45 %** | ✅ Best real accuracy, generalises well |

### Why SVM was chosen

1. **No overfitting** — test 94.75 % vs CV 95.45 % (gap = 0.7 %, excellent)
2. **Generalisation** — SVM maximises the margin between classes → better on unseen data
3. **Needs StandardScaler** — which forces a proper Pipeline (good engineering practice)
4. **`probability=True`** — gives confidence % shown in the UI ("87 % confident: High Stress")
5. **Works well on medium tabular datasets** with clear decision boundaries

### Why Random Forest / XGBoost were rejected

> 100 % accuracy on **all** splits (train, test, CV) means the model **memorised** patterns specific to the synthetic dataset.  
> A 100 % score is a red flag when the dataset is machine-generated — it would fail on real student data.

---

## 💡 Explainability (SHAP)

Every prediction includes a **SHAP** breakdown showing which features pushed the result:

```
Example — "High Stress" prediction:
  Study_Hours              ████████████ +0.42  → pushes toward High
  Sleep_Hours              ████████    −0.31   → reduces stress
  Physical_Activity_Hours  ████        −0.18   → reduces stress
  CGPA                     ██          −0.09
  Social_Interaction_Hours █           −0.04
  Hobbies_Hours            █           +0.03
```

Without SHAP, the model is a black box. With SHAP, students see exactly **which lifestyle factor is driving their stress** — making the prediction actionable.

---

## 🖥️ How to Run Locally

### Prerequisites
- Python 3.11+
- Node.js 18+ (`node -v` to check)
- Git

### Copy & paste into PowerShell

**Terminal 1 — Backend:**
```powershell
cd "C:\Users\VISHVAJEET\Desktop\SOM Project"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python backend/model/train.py

uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```powershell
cd "C:\Users\VISHVAJEET\Desktop\SOM Project\frontend"
npm install
npm run dev
```

> App opens at **http://localhost:5173**  
> API docs at **http://localhost:8000/docs**

### Docker (optional)
```powershell
docker build -t zenith-stress .
docker run -p 80:80 zenith-stress
```

---

## 📡 API Docs

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | `{"status": "ok"}` |
| `/predict` | POST | stress_level, probabilities, shap values |
| `/history` | GET | Last 50 predictions |

**POST /predict — Request Body:**
```json
{
  "study_hours": 8.0,
  "hobbies_hours": 1.5,
  "sleep_hours": 5.0,
  "social_hours": 1.0,
  "physical_hours": 0.5,
  "cgpa": 3.2
}
```

**Response:**
```json
{
  "stress_level": "High",
  "probabilities": { "High": 0.87, "Moderate": 0.11, "Low": 0.02 },
  "shap": { "Study_Hours": 0.42, "Sleep_Hours": -0.31, "Physical_Activity_Hours": -0.18 }
}
```

---

## 🧰 Tech Stack

| Category | Tools |
|---|---|
| ML | scikit-learn, SHAP, pandas, numpy, joblib |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React, Vite, Recharts, Lucide-React |
| Notebook | Jupyter (training experiments & EDA) |
| DevOps | Docker, Git, GitHub |

---

## 📁 Project Structure

```
SOM Project/
├── backend/
│   ├── main.py                  # FastAPI app + CORS
│   ├── routers/predict.py       # POST /predict + SHAP
│   ├── model/
│   │   ├── train.py             # SVM training pipeline
│   │   └── model.pkl            # Saved trained model
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/               # Predict, Pipeline, History, Chatbot
│   │   ├── components/          # Navbar, ShapChart, Gauge, etc.
│   │   └── main.jsx
│   ├── vite.config.js           # Proxy /api → FastAPI
│   └── package.json
├── student_lifestyle_dataset2.csv
├── Student_Stress_Level_Predictor_Training.ipynb
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🔮 Future Work

- [ ] Replace synthetic dataset with real survey data from students
- [ ] LLM-powered chatbot (Mistral-7B) for deeper personalised advice
- [ ] Deploy: Vercel (frontend) + Render (backend)
- [ ] GitHub Actions CI/CD — auto test + build on every push
- [ ] Mobile app using React Native

---

*Built end-to-end — showing the full ML lifecycle from data collection to a deployed, explainable web app.*
