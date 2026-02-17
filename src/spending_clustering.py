import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path):
    """
    Load dataset and use existing preprocessing pipeline
    """
    from preprocessing import preprocess_pipeline
    df = preprocess_pipeline(file_path)
    return df


def prepare_spending_features(df):
    """
    Create category-wise spending features for clustering
    """
    # Filter only expenses (important for financial behavior analysis)
    expense_df = df[df['Type'] == 'expense']

    # Group by category and calculate total spending
    category_spending = expense_df.groupby('Category').agg({
        'Amount': 'sum'
    }).reset_index()

    return category_spending


def apply_kmeans_clustering(data, n_clusters=3):
    """
    Apply K-Means clustering on spending data
    """
    # Select numerical feature
    X = data[['Amount']]

    # Scale features (VERY IMPORTANT for K-Means)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(X_scaled)

    # Save trained KMeans model
    joblib.dump(kmeans, "models/kmeans_spending_model.pkl")
    print("KMeans model saved to models/kmeans_spending_model.pkl")

    return data, kmeans, scaler

def interpret_clusters(data):
    """
    Add interpretation labels for clusters
    """
    # Sort by spending amount
    data = data.sort_values(by='Amount')

    # Map clusters to labels based on spending level
    cluster_means = data.groupby('Cluster')['Amount'].mean().sort_values()

    cluster_labels = {}
    labels = ['Low Spending', 'Medium Spending', 'High Spending']

    for i, cluster in enumerate(cluster_means.index):
        cluster_labels[cluster] = labels[i]

    data['Spending_Level'] = data['Cluster'].map(cluster_labels)

    return data


def visualize_clusters(data):
    """
    Visualize clustering results
    """
    plt.figure(figsize=(10, 6), num="Spending Clustering - AI Financial Advisor")
    plt.scatter(data['Category'], data['Amount'], c=data['Cluster'])
    plt.title("Spending Pattern Clustering by Category")
    plt.xlabel("Category")
    plt.ylabel("Total Spending Amount")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save figure for report
    plt.savefig("reports/spending_clustering.png", dpi=300)
    plt.show()


def main():
    # IMPORTANT: Use Kaggle dataset (main pipeline)
    file_path = "data/transactions_kaggle.csv"

    print("Loading and preprocessing data...")
    df = load_and_preprocess_data(file_path)

    print("\nPreparing spending features...")
    spending_data = prepare_spending_features(df)

    print("\nApplying K-Means Clustering...")
    clustered_data, model, scaler = apply_kmeans_clustering(spending_data)

    print("\nInterpreting clusters...")
    final_data = interpret_clusters(clustered_data)

    print("\nFinal Clustered Spending Analysis:\n")
    print(final_data)

    print("\nCluster Summary (Average Spending per Cluster):\n")
    print(final_data.groupby('Spending_Level')['Amount'].mean())

    # Visualize results
    visualize_clusters(final_data)

if __name__ == "__main__":
    main()
