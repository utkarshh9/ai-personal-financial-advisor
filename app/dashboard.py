import streamlit as st
import plotly.express as px
import pandas as pd
import joblib
import os
import sys

# --- MODEL LOADER ---
@st.cache_resource
def load_model(model_path):
    """
    Load saved ML model if available.
    Falls back gracefully if model file is missing.
    """
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import preprocess_pipeline
from financial_health import calculate_financial_metrics, compute_health_score, interpret_score
from investment_advisor import recommend_investment

# Page Configuration
st.set_page_config(
    page_title="MoneyMeow",
    page_icon="💰",
    layout="wide"
)

st.markdown(
    """
    <style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }

    /* KPI Card Styling */
    .metric-card {
        background-color: #0E1117;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #262730;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        text-align: center;
    }

    .metric-title {
        font-size: 16px;
        color: #9aa0a6;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #00C896;
    }

    /* Section Header Styling */
    .section-header {
        font-size: 28px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    /* Health Badge */
    .badge-good {
        background-color: #00C896;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    .badge-average {
        background-color: #FFA500;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    .badge-poor {
        background-color: #FF4B4B;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='margin-bottom:0px;'>MoneyMeow</br>AI Personal Financial Advisor</h1>
    <h2 style='margin-top:0px; font-size:18px; color:gray;'>
    Smart Budgeting, Forecasting & Investment Intelligence Dashboard
    </h2>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ["Overview", "Spending Analysis", "Budget Forecast", "Financial Health", "Investment Advisor"]
)

# --- DATA SOURCE SELECTION (UPLOAD OR DEFAULT) ---
st.sidebar.subheader("📁 Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload your financial transactions CSV",
    type=["csv"]
)

@st.cache_data
def load_default_data():
    return preprocess_pipeline("data/transactions_kaggle.csv")

@st.cache_data
def load_uploaded_data(file):
    raw_df = pd.read_csv(file)
    df = preprocess_pipeline(file)
    return df

# Choose dataset dynamically
if uploaded_file is not None:
    st.sidebar.success("Custom dataset uploaded successfully!")
    df = preprocess_pipeline(uploaded_file)
else:
    st.sidebar.info("Using default Kaggle dataset")
    df = load_default_data()

# Basic schema validation
required_columns = ["Date", "Category", "Amount", "Type"]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Uploaded dataset is missing required columns: {missing_cols}")
    st.stop()

# OVERVIEW
if section == "Overview":
    st.header("Financial Overview")

    total_income = df['Income'].sum()
    total_expense = df['Expense'].sum()
    total_savings = total_income - total_expense

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Income</div>
                <div class="metric-value">₹ {total_income:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Expense</div>
                <div class="metric-value">₹ {total_expense:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Savings</div>
                <div class="metric-value">₹ {total_savings:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("## Expense Distribution by Category")

    expense_df = df[df['Type'].str.lower() == 'expense']
    category_expense = expense_df.groupby("Category")["Amount"].sum().reset_index()

    pie_fig = px.pie(
        category_expense,
        names="Category",
        values="Amount",
        title="Expense Category Breakdown",
        hole=0.4
    )

    pie_fig.update_layout(template="plotly_dark")

    st.plotly_chart(pie_fig, width='stretch')



    st.subheader("Dataset Preview (First 5 Rows)")
    st.dataframe(df.head(), width='stretch')

    st.caption(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")


# SPENDING ANALYSIS (Clustering Output)
elif section == "Spending Analysis":
    st.header("Spending Clustering Analysis")

    clustering_model = load_model("models/kmeans_spending_model.pkl")

    # Aggregate spending by category
    category_spending = df.groupby("Category")["Amount"].sum().reset_index()

    fig = px.bar(
        category_spending,
        x="Category",
        y="Amount",
        color="Amount",
        title="Category-wise Spending Distribution",
        text_auto=True
    )

    fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Total Spending",
        template="plotly_dark"
    )

    st.plotly_chart(fig, width='stretch')

    if clustering_model:
        st.success("Loaded pre-trained clustering model")
    else:
        st.info("Clustering running without saved model (fallback mode)")


# BUDGET FORECAST
elif section == "Budget Forecast":
    st.header("Monthly Expense Forecast")

    # Load saved forecasting model (if available)
    forecast_model = load_model("models/expense_forecasting_model.pkl")

    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    expense_df = df[df['Type'].str.lower() == 'expense']

    monthly_expense = (
        expense_df
        .set_index('Date')
        .resample('ME')['Amount']
        .sum()
        .reset_index()
    )

    # Create time index (same as training)
    monthly_expense['TimeIndex'] = range(len(monthly_expense))

    fig = px.line(
        monthly_expense,
        x="Date",
        y="Amount",
        markers=True,
        title="Monthly Expense Trend (Time Series Analysis)"
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="#FAFAFA"),
        xaxis_title="Month",
        yaxis_title="Total Expense"
    )

    st.plotly_chart(fig, width='stretch')

    # Show model status
    if forecast_model:
        st.success("Loaded pre-trained forecasting model (production mode)")
    else:
        st.info("Using dynamic computation (model file not found)")



# FINANCIAL HEALTH
elif section == "Financial Health":
    st.header("Financial Health Analysis")

    metrics = calculate_financial_metrics(df)
    score = compute_health_score(metrics)
    interpretation = interpret_score(score)

    col1, col2 = st.columns(2)
    # Determine badge class
    if score >= 80:
        badge_class = "badge-good"
    elif score >= 60:
        badge_class = "badge-average"
    else:
        badge_class = "badge-poor"
    
    st.markdown(f"<h4>Financial Health Score: {score}/100</h4>", unsafe_allow_html=True)
    st.markdown(
        f"<span class='{badge_class}'>{interpretation}</span>",
        unsafe_allow_html=True
    )

    # Extract metrics
    income = metrics["total_income"]
    expense = metrics["total_expense"]
    savings = metrics["total_savings"]

    health_df = pd.DataFrame({
        "Metric": ["Income", "Expense", "Savings"],
        "Amount": [income, expense, savings]
    })

    health_fig = px.bar(
        health_df,
        x="Metric",
        y="Amount",
        color="Metric",
        text_auto=True,
        title="Income vs Expense vs Savings Analysis"
    )

    health_fig.update_layout(
        template="plotly_dark",
        xaxis_title="Financial Metric",
        yaxis_title="Amount (₹)"
    )

    st.plotly_chart(health_fig, width='stretch')


# INVESTMENT ADVISOR
elif section == "Investment Advisor":
    st.header("AI Investment Recommendations")

    metrics = calculate_financial_metrics(df)
    score = compute_health_score(metrics)

    savings = metrics["total_savings"]
    savings_ratio = metrics["savings_ratio"]

    risk_profile, advice = recommend_investment(score, savings, savings_ratio)

    st.subheader("Recommended Investor Profile")
    st.success(risk_profile)

    st.subheader("Personalized Investment Suggestions")
    for i, rec in enumerate(advice, 1):
        st.write(f"{i}. {rec}")
