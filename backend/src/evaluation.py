import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Evaluating model stability by testing various noise levels and plotting the recurrence of towns within its original cluster 
# PCA used to visualize the town clusters

def cluster_stability_existing_model(df, norm_cols, kmeans_model, n_runs=100, noise_scale=0.2):
    n_towns = len(df)
    cluster_assignments = np.zeros((n_towns, n_runs), dtype=int)

    # Loop through and add noise to data and feed through existing model
    for i in range(n_runs):
        noisy = df[norm_cols].copy()
        noise = np.random.normal(loc=0.0, scale=noise_scale, size=noisy.shape)
        noisy = np.clip(noisy + noise, 0, 1)

        cluster_assignments[:, i] = kmeans_model.predict(noisy)

    # Count how many times the town stayed in the same cluster after adding noise, and divide by the total 
    stability = []
    for row in cluster_assignments:
        most_common = np.bincount(row).max()
        stability.append(most_common / n_runs)

    df = df.copy()
    df['ClusterStability'] = stability
    return df.sort_values(by="ClusterStability", ascending=True)

def sweeping_stability(df, norm_cols, kmeans_model, n_runs=100, max_scale=0.5):
    # The range of the noise to test
    scales = np.linspace(0.01, max_scale, 10)
    stability_scores = []

    for scale in scales:
        print(f"Testing noise scale: {scale:.2f}")

        stability_df = cluster_stability_existing_model(df, norm_cols, kmeans_model, n_runs=n_runs, noise_scale=scale)
        
        # Calculate percentage of towns that remained the same
        score = stability_df['ClusterStability'].mean() * 100
        stability_scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(scales, stability_scores, marker='o', linestyle='-', color='royalblue')
    plt.title("Cluster Stability vs. Gaussian Noise Scale")
    plt.xlabel("Noise Scale (std dev of added noise)")
    plt.ylabel("Stability (% same cluster)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graph_pca(df, kmeans_model, norm_cols):
    X = df[norm_cols]

    # Predict the cluster labels
    df['Cluster'] = kmeans_model.predict(X)

    # Calculate PCA components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))
    unique_clusters = sorted(df['Cluster'].unique())

    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = df['Cluster'] == cluster_id
        plt.scatter(
            X_pca[cluster_mask, 0],
            X_pca[cluster_mask, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.7,
            edgecolor='black',
            s=60
        )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Town Clustering (PCA projection)")
    plt.legend(title="Cluster", loc="best", frameon=True)
    plt.grid(True, linestyle='-', alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Run functions when file is executed directly. Plots shown by default, but not saved
    data_path = "../data/processed/master_town_data.parquet"
    model_path = "../data/processed/master_town_data.kmeans.pkl"

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    print(f"Loading model from {model_path}...")
    kmeans_model = joblib.load(model_path)

    norm_cols = [c for c in df.columns if c.endswith('_norm')]

    print("Running cluster stability evaluation on existing model...")
    df_with_stability = cluster_stability_existing_model(df, norm_cols, kmeans_model, n_runs=20, noise_scale=0.02)
    df_with_stability[['TOWN_ID', 'ClusterStability']].to_csv("data/processed/cluster_stability_scores.csv", index=False)

    sweeping_stability(df_with_stability, norm_cols, kmeans_model, n_runs=20, max_scale=0.5)

    print("Graphing with PCA Analysis on existing model...")
    graph_pca(df, kmeans_model, norm_cols)


