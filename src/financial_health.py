import pandas as pd
import matplotlib.pyplot as plt
import os


def load_and_preprocess_data(file_path):
    """
    Load dataset using existing preprocessing pipeline
    """
    from preprocessing import preprocess_pipeline
    df = preprocess_pipeline(file_path)
    return df


def calculate_financial_metrics(df):
    """
    Calculate key financial metrics: income, expenses, savings
    """
    total_income = df['Income'].sum()
    total_expense = df['Expense'].sum()
    total_savings = total_income - total_expense

    savings_ratio = (total_savings / total_income) if total_income > 0 else 0
    expense_ratio = (total_expense / total_income) if total_income > 0 else 0

    return {
        "total_income": total_income,
        "total_expense": total_expense,
        "total_savings": total_savings,
        "savings_ratio": savings_ratio,
        "expense_ratio": expense_ratio
    }


def compute_health_score(metrics):
    """
    Compute Financial Health Score (0–100)
    Based on savings ratio and expense management
    """
    savings_ratio = metrics["savings_ratio"]
    expense_ratio = metrics["expense_ratio"]

    score = 0

    # Savings contribution (50 points)
    if savings_ratio >= 0.30:
        score += 50
    elif savings_ratio >= 0.20:
        score += 40
    elif savings_ratio >= 0.10:
        score += 30
    elif savings_ratio > 0:
        score += 20
    else:
        score += 5

    # Expense control contribution (50 points)
    if expense_ratio <= 0.50:
        score += 50
    elif expense_ratio <= 0.70:
        score += 40
    elif expense_ratio <= 0.85:
        score += 25
    else:
        score += 10

    return min(score, 100)


def interpret_score(score):
    """
    Provide financial health interpretation
    """
    if score >= 80:
        return "Excellent Financial Health"
    elif score >= 60:
        return "Good Financial Health"
    elif score >= 40:
        return "Average Financial Health"
    else:
        return "Poor Financial Health"


def generate_advice(metrics, score):
    """
    Generate personalized financial advice
    """
    advice = []

    if metrics["savings_ratio"] < 0.20:
        advice.append("Increase your monthly savings to improve financial stability.")

    if metrics["expense_ratio"] > 0.70:
        advice.append("Your expenses are high relative to income. Consider budget optimization.")

    if metrics["total_savings"] > 0:
        advice.append("You are maintaining positive savings. Good financial discipline.")
    else:
        advice.append("Your expenses exceed income. Immediate expense control is recommended.")

    if score >= 80:
        advice.append("You are financially healthy. Consider increasing investments.")

    return advice

def visualize_financial_health(metrics, score):
    """
    Create and save financial health visualization
    """
    # Create reports folder if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    labels = ['Total Income', 'Total Expense', 'Total Savings']
    values = [
        metrics['total_income'],
        metrics['total_expense'],
        metrics['total_savings']
    ]

    plt.figure(figsize=(10, 6), num="Financial Health Analysis - AI Advisor")

    # Bar chart for financial metrics
    plt.bar(labels, values)
    plt.title(f"Financial Health Analysis (Score: {score}/100)")
    plt.ylabel("Amount")
    plt.xlabel("Financial Metrics")

    # Add value labels on top of bars
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:,.0f}", ha='center', va='bottom')

    plt.tight_layout()

    # Save image for report & UI
    plt.savefig("reports/financial_health_score.png", dpi=300)

    plt.show()


def main():
    file_path = "data/transactions_kaggle.csv"

    print("Loading and preprocessing financial data...")
    df = load_and_preprocess_data(file_path)

    print("\nCalculating financial metrics...")
    metrics = calculate_financial_metrics(df)

    print("\nFinancial Metrics:")
    print(f"Total Income: {metrics['total_income']:.2f}")
    print(f"Total Expense: {metrics['total_expense']:.2f}")
    print(f"Total Savings: {metrics['total_savings']:.2f}")
    print(f"Savings Ratio: {metrics['savings_ratio']:.2f}")
    print(f"Expense Ratio: {metrics['expense_ratio']:.2f}")

    print("\nComputing Financial Health Score...")
    score = compute_health_score(metrics)
    interpretation = interpret_score(score)

    print(f"\nFinancial Health Score: {score}/100")
    print(f"Health Status: {interpretation}")

    print("\nPersonalized Financial Advice:")
    advice_list = generate_advice(metrics, score)
    for i, advice in enumerate(advice_list, 1):
        print(f"{i}. {advice}")

    # Generate financial health visualization
    visualize_financial_health(metrics, score)


if __name__ == "__main__":
    main()
