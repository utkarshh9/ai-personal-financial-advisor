import pandas as pd

def load_data(file_path):
    """
    Load the dataset from CSV file
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df):
    """
    Perform data cleaning and preprocessing
    """
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Remove rows with invalid dates (if any)
    df = df.dropna(subset=['Date'])

    # Standardize text columns (lowercase)
    df['Transaction Description'] = df['Transaction Description'].str.lower()
    df['Category'] = df['Category'].str.lower()
    df['Type'] = df['Type'].str.lower()

    # Remove duplicate transactions
    df = df.drop_duplicates()

    return df


def feature_engineering(df):
    """
    Create new features for ML models
    """
    # Extract time-based features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.day_name()

    # Create Income and Expense columns
    df['Income'] = df.apply(lambda x: x['Amount'] if x['Type'] == 'income' else 0, axis=1)
    df['Expense'] = df.apply(lambda x: x['Amount'] if x['Type'] == 'expense' else 0, axis=1)

    # Savings calculation per transaction (basic logic)
    df['Savings'] = df['Income'] - df['Expense']

    return df


def preprocess_pipeline(file_path):
    """
    Complete preprocessing pipeline
    """
    df = load_data(file_path)
    df = clean_data(df)
    df = feature_engineering(df)
    return df


# For testing the module independently
if __name__ == "__main__":
    file_path = "data/transactions_kaggle.csv"
    processed_df = preprocess_pipeline(file_path)
    print(processed_df.head())
    print("\nProcessed Data Shape:", processed_df.shape)
