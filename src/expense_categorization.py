import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(file_path):
    """
    Load the dataset
    """
    df = pd.read_csv(file_path)
    return df


def preprocess_text(df):
    """
    Prepare text data for NLP model
    """
    # Convert text to lowercase
    df['Transaction Description'] = df['Transaction Description'].str.lower()

    # Define features and target
    X = df['Transaction Description']
    y = df['Category']

    return X, y


def train_model(X, y):
    """
    Train NLP classification model using TF-IDF + Logistic Regression
    """
    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)

    # Split dataset into training and testing sets (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y

    )

    # Initialize model
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    joblib.dump(model, "models/expense_classifier.pkl")
    joblib.dump(vectorizer, "models/text_vectorizer.pkl")

    return model, vectorizer, X_test, y_test, y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance
    """
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def main():
    # Correct dataset path (from project root)
    file_path = "data/transactions_custom.csv"

    # Load dataset
    df = load_data(file_path)

    # Preprocess text data
    X, y = preprocess_text(df)

    # Train model
    model, vectorizer, X_test, y_test, y_pred = train_model(X, y)

    # Evaluate model
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main()
