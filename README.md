# AI Personal Financial Advisor
An AI-powered financial advisor that analyzes user transactions, predicts budget trends, calculates financial health score, and provides smart investment suggestions using Machine Learning.

## Features (Planned)
- Expense Categorization (NLP)
- Budget Prediction (Time Series)
- Spending Pattern Clustering
- Financial Health Score
- Investment Recommendation Engine
- Streamlit Dashboard

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