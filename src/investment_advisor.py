import pandas as pd


def load_and_preprocess_data(file_path):
    """
    Load dataset using existing preprocessing pipeline
    """
    from preprocessing import preprocess_pipeline
    df = preprocess_pipeline(file_path)
    return df


def calculate_financial_metrics(df):
    """
    Reuse core financial metrics for investment decision
    """
    total_income = df['Income'].sum()
    total_expense = df['Expense'].sum()
    total_savings = total_income - total_expense

    savings_ratio = (total_savings / total_income) if total_income > 0 else 0
    expense_ratio = (total_expense / total_income) if total_income > 0 else 0

    return total_income, total_expense, total_savings, savings_ratio, expense_ratio


def get_financial_health_score(savings_ratio, expense_ratio):
    """
    Simplified health score logic (same philosophy as Step 9)
    """
    score = 0

    # Savings contribution
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

    # Expense control contribution
    if expense_ratio <= 0.50:
        score += 50
    elif expense_ratio <= 0.70:
        score += 40
    elif expense_ratio <= 0.85:
        score += 25
    else:
        score += 10

    return min(score, 100)


def recommend_investment(score, savings, savings_ratio):
    """
    Generate personalized investment recommendations
    """
    recommendations = []

    # Case 1: Poor financial health
    if score < 40:
        recommendations.append("Focus on emergency fund before investments.")
        recommendations.append("Avoid high-risk investments like stocks.")
        recommendations.append("Consider safe options: Fixed Deposits, Recurring Deposits.")
        risk_profile = "Low Risk Investor"

    # Case 2: Moderate financial health
    elif 40 <= score < 70:
        recommendations.append("You can start balanced investments.")
        recommendations.append("Consider Mutual Funds and SIPs for steady growth.")
        recommendations.append("Maintain a balance between savings and investments.")
        risk_profile = "Moderate Risk Investor"

    # Case 3: Strong financial health
    else:
        recommendations.append("You are financially stable for long-term investments.")
        recommendations.append("Consider Stocks, Index Funds, and Diversified Portfolios.")
        recommendations.append("Increase investment allocation for wealth growth.")
        risk_profile = "High Growth Investor"

    # Additional logic based on savings
    if savings <= 0:
        recommendations.append("Currently negative savings detected. Prioritize expense control.")

    elif savings > 0 and savings_ratio > 0.20:
        recommendations.append("Good savings rate detected. You can allocate surplus to investments.")

    return risk_profile, recommendations


def main():
    file_path = "data/transactions_kaggle.csv"

    print("Loading financial data for investment analysis...")
    df = load_and_preprocess_data(file_path)

    print("\nCalculating financial metrics...")
    income, expense, savings, savings_ratio, expense_ratio = calculate_financial_metrics(df)

    print(f"\nTotal Income: {income:.2f}")
    print(f"Total Expense: {expense:.2f}")
    print(f"Total Savings: {savings:.2f}")
    print(f"Savings Ratio: {savings_ratio:.2f}")
    print(f"Expense Ratio: {expense_ratio:.2f}")

    print("\nEvaluating financial health for investment suitability...")
    score = get_financial_health_score(savings_ratio, expense_ratio)
    print(f"Financial Health Score: {score}/100")

    print("\nGenerating AI Investment Recommendations...")
    risk_profile, advice = recommend_investment(score, savings, savings_ratio)

    print(f"\nRecommended Investor Profile: {risk_profile}")
    print("\nPersonalized Investment Suggestions:")
    for i, rec in enumerate(advice, 1):
        print(f"{i}. {rec}")


if __name__ == "__main__":
    main()
