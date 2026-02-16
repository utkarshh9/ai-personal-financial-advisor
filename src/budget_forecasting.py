import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def load_and_preprocess_data(file_path):
    """
    Load dataset using existing preprocessing pipeline
    """
    from preprocessing import preprocess_pipeline
    df = preprocess_pipeline(file_path)
    return df


def prepare_monthly_expense(df):
    """
    Aggregate monthly expenses for time series forecasting
    """
    # Filter only expenses (important for budget prediction)
    expense_df = df[df['Type'] == 'expense']

    # Ensure Date column is datetime
    expense_df['Date'] = pd.to_datetime(expense_df['Date'])

    # Extract Year-Month for aggregation
    expense_df['YearMonth'] = expense_df['Date'].dt.to_period('M')

    # Monthly total expense
    monthly_expense = expense_df.groupby('YearMonth')['Amount'].sum().reset_index()

    # Convert Period to timestamp for plotting
    monthly_expense['YearMonth'] = monthly_expense['YearMonth'].astype(str)

    return monthly_expense


def train_forecasting_model(monthly_data):
    """
    Train Linear Regression model for time series forecasting
    """
    # Create time index as numerical feature
    monthly_data['TimeIndex'] = np.arange(len(monthly_data))

    X = monthly_data[['TimeIndex']]
    y = monthly_data['Amount']

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    return model, X, y, monthly_data


def forecast_future_expense(model, monthly_data, steps=3):
    """
    Forecast future expenses for next n months
    """
    last_index = monthly_data['TimeIndex'].iloc[-1]

    future_indices = np.array(range(last_index + 1, last_index + 1 + steps)).reshape(-1, 1)
    future_predictions = model.predict(future_indices)

    # Create future months labels
    last_month = pd.Period(monthly_data['YearMonth'].iloc[-1], freq='M')
    future_months = [(last_month + i).strftime('%Y-%m') for i in range(1, steps + 1)]

    forecast_df = pd.DataFrame({
        'YearMonth': future_months,
        'Predicted_Expense': future_predictions
    })

    return forecast_df


def visualize_forecast(monthly_data, forecast_df):
    """
    Visualize historical and forecasted expenses
    """
    plt.figure(figsize=(12, 6), num="Budget Forecast - AI Financial Advisor")

    # Plot historical data
    plt.plot(monthly_data['YearMonth'], monthly_data['Amount'], marker='o', label='Actual Expenses')

    # Plot forecasted data
    future_x = list(range(len(monthly_data), len(monthly_data) + len(forecast_df)))
    plt.plot(
        list(monthly_data['YearMonth']) + list(forecast_df['YearMonth']),
        list(monthly_data['Amount']) + list(forecast_df['Predicted_Expense']),
        linestyle='--',
        marker='o',
        label='Forecasted Expenses'
    )

    plt.title("Monthly Expense Forecast (AI Personal Financial Advisor)")
    plt.xlabel("Month")
    plt.ylabel("Total Expense Amount")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save figure for report
    plt.savefig("reports/budget_forecast.png", dpi=300)

    plt.show()


def main():
    file_path = "data/transactions_kaggle.csv"

    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)

    print("\nPreparing monthly expense data...")
    monthly_data = prepare_monthly_expense(df)
    print("\nMonthly Expense Data:\n")
    print(monthly_data)

    print("\nTraining forecasting model...")
    model, X, y, monthly_data = train_forecasting_model(monthly_data)

    print("\nForecasting next 3 months expenses...")
    forecast_df = forecast_future_expense(model, monthly_data, steps=3)
    print("\nForecasted Future Expenses:\n")
    print(forecast_df)

    visualize_forecast(monthly_data, forecast_df)


if __name__ == "__main__":
    main()
