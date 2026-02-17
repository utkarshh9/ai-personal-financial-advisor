# MoneyMeow - AI Personal Financial Advisor
🔗 **Live Application:** https://moneymeow.streamlit.app

### Smart Budgeting, Expense Analysis & Investment Suggestions using Machine Learning

---

## Project Overview

The AI Personal Financial Advisor is an end-to-end machine learning based financial analytics dashboard that helps users analyze spending patterns, forecast future expenses, evaluate financial health, and receive intelligent budgeting and investment suggestions.

The system integrates multiple ML techniques including clustering, time series forecasting, and NLP-based expense categorization within an interactive Streamlit dashboard designed with a fintech-style UI.

This project demonstrates a production-grade ML pipeline with model evaluation, model persistence, and real-time inference through a user-friendly dashboard.

---

## How to Run the Project (Reproducibility Guide)

### Step 1: Clone Repository

```bash
git clone https://github.com/utkarshh9/ai-personal-financial-advisor.git
cd ai-personal-financial-advisor
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser automatically.

---

## Objectives

* Analyze personal financial transaction data
* Categorize expenses using NLP techniques
* Identify spending patterns using clustering
* Forecast future expenses using time series modeling
* Calculate financial health score
* Provide intelligent financial and investment suggestions
* Deploy an interactive AI dashboard for real-time analysis

---

## Key Features

* Interactive FinTech Dashboard (Streamlit)
* Dataset Upload for Real-Time Analysis
* Machine Learning-Based Expense Categorization (NLP)
* Time Series Expense Forecasting
* Spending Pattern Clustering (K-Means)
* Financial Health Scoring System
* Personalized Budget & Investment Suggestions
* Saved Models for Production-Level Inference

---

## Tech Stack

| Category             | Technologies Used  |
| -------------------- | ------------------ |
| Programming Language | Python             |
| Machine Learning     | Scikit-learn       |
| Data Processing      | Pandas, NumPy      |
| Visualization        | Plotly, Matplotlib |
| Dashboard            | Streamlit          |
| Model Persistence    | Joblib             |
| Version Control      | Git & GitHub       |

---

## System Architecture

Data Pipeline:

```
Dataset → Preprocessing → Feature Engineering → ML Models →
Model Evaluation → Model Saving (.pkl) → Streamlit Dashboard (Inference)
```

Modules:

* Data Preprocessing Module
* Expense Categorization (NLP)
* Spending Clustering (Unsupervised ML)
* Budget Forecasting (Time Series Regression)
* Financial Health Scoring Engine
* Investment Recommendation System
* Interactive Dashboard (Deployment Layer)

---

## Dataset

This project utilizes two datasets to ensure both realistic financial analysis and correct NLP validation within the AI Personal Financial Advisor system.

1. Primary Dataset (Main Financial Analysis)
- Dataset Name: Personal Finance Data
- Source: Kaggle
- Link: https://www.kaggle.com/datasets/ramyapintchy/personal-finance-data

The dataset contains transaction-level financial data including date, transaction description, category, amount, and transaction type (income/expense), which is used for budget prediction, financial health scoring, and investment recommendation

2. Secondary Dataset (NLP Validation Dataset)
- File: data/transactions_custom.csv
- Type: Custom realistic financial transaction dataset
- Purpose: Used exclusively to validate the NLP-based expense categorization module.

Using this semantically meaningful dataset, the TF-IDF + Logistic Regression model achieved ~94% accuracy, confirming that the NLP expense categorization module functions correctly when trained on relevant textual data.

Key Columns:

* Date
* Category
* Amount
* Type (Income/Expense)
* Description (for NLP validation)

---

## Machine Learning Modules

### 1. Expense Categorization (NLP)

* Technique: TF-IDF Vectorization + Classification
* Purpose: Automatically categorize transaction descriptions
* Demonstrates Natural Language Processing integration

### 2. Spending Clustering (Unsupervised Learning)

* Algorithm: K-Means Clustering
* Goal: Identify Low, Medium, and High Spending Categories
* Output: Clustered financial behavior patterns

### 3. Budget Forecasting (Time Series)

* Model: Linear Regression (Time Index Based)
* Input: Monthly aggregated expense data
* Output: Future expense predictions for upcoming months

---

## Model Evaluation

The forecasting model was evaluated using:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

This ensures quantitative validation of model performance instead of relying only on visual graphs.

---

## Model Saving & Production Inference

* Trained models are serialized using Joblib (`.pkl` files)
* Stored inside the `/models` directory
* Dashboard loads pre-trained models for faster inference
* Fallback mechanism ensures system works even if models are missing

This follows industry-standard ML deployment practices.

---

## Interactive Dashboard Features

Built using Streamlit with Plotly interactive charts:

* KPI Cards (Income, Expense, Savings)
* Monthly Expense Forecast (Interactive Line Chart)
* Spending Distribution (Bar & Pie Charts)
* Financial Health Score Visualization
* AI-Based Financial Advice Engine
* Upload Custom CSV Dataset for Real-Time Analysis

---

## Academic Concepts Covered

* Supervised Learning
* Unsupervised Learning
* Time Series Forecasting
* Natural Language Processing (NLP)
* Data Preprocessing & Feature Engineering
* Model Evaluation & Metrics
* Model Persistence (Joblib)
* ML Deployment via Dashboard

---

## Conclusion

This project successfully implements a production-style AI financial analytics system integrating machine learning, data visualization, and interactive dashboard deployment. It demonstrates end-to-end ML lifecycle including preprocessing, model training, evaluation, model persistence, and real-time inference through a scalable and reproducible architecture suitable for fintech applications.
