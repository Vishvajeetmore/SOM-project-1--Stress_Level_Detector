---
title: Stress Level Predictor
emoji: 🏃
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.38.0
app_file: app.py
pinned: false
short_description: This determines stress level in students
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

 # Stress-Level Predictor

I created a model to Predict Stress-Level using different types of regression models:

Random Forest
XGBoost
Logistic Regression
Among these, the logistic Regression model performed the best, with an accuracy of 85.667% and R2 score of       
0.4585498907600657 and root mean squared error(rmse)value = 0.6658328118479393.
XGBOOST and Random Forest were having accuracy 100% which is not good as it was overfitting the model.
## Data Source
I gathered the  data from kaggle and online sources.

## Features
-Study Hours Per Day
-Extracurricular Hours Per Day
-Sleep Hours Per Day
-Social Hours Per Day
-Physical Activity Hours Per Day
-CGPA(0-4)
-Stress Level


