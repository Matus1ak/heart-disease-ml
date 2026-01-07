# Heart Disease Classification (ML)

A simple machine learning project that predicts the presence of heart disease using basic clinical features and **Logistic Regression**.

## Dataset
- Source: Kaggle — *Heart Failure Prediction*
- Target: `HeartDisease` (0 = no disease, 1 = disease)

## What’s done
- Basic exploratory data analysis (EDA)
- One-hot encoding of categorical variables
- Feature scaling with `StandardScaler`
- Logistic Regression as a baseline model
- Model evaluation using confusion matrix and classification report
- Coefficient analysis for feature impact

## How to run
```bash
pip install -r requirements.txt
python heart_classification.py
```
