import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


DEFAULT_NUM_CLUSTERS = 3
RANDOM_STATE = 42


def _select_feature_columns(df: pd.DataFrame, include: List[str] = None) -> List[str]:
    """Return a list of numeric columns that should be used for clustering.

    If *include* is provided, only those columns (and present in *df*) are used.
    Otherwise all numeric columns are selected and the obvious identifier cols are skipped.
    """
    if include is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {
            "TOWN_ID",
            "GEOID",
            "MatchScore",
        }
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        numeric_cols = [c for c in numeric_cols if df[c].notna().any()]
        return numeric_cols
    else:
        return [c for c in include if c in df.columns]


def add_normalised_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Add *_norm columns. Fills NaN with column median (or 0) before scaling."""
    for col in feature_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median).fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols])
    for idx, col in enumerate(feature_cols):
        df[f"{col}_norm"] = scaled[:, idx]
    return df


def fit_kmeans(df: pd.DataFrame, feature_cols: List[str], k: int = DEFAULT_NUM_CLUSTERS) -> KMeans:
    """Fit a KMeans model and attach *Cluster* integer labels to *df*."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    df["Cluster"] = kmeans.fit_predict(df[feature_cols])
    return kmeans

def _map_cluster_to_label(df: pd.DataFrame) -> pd.Series:
    """Map numeric clusters to lifestyle labels.
    Uses PopulationDensity heuristic:
    - Highest mean density -> "City"
    - Lowest -> "Rural"
    - Middle -> "Suburb"
    Uses generic names if the heuristic fails.
    """
    density_col = "PopulationDensity"

    if density_col not in df.columns:
        print(f"[debug] WARN: {density_col} not found. Using generic labels.")
        unique_clusters = sorted(df["Cluster"].unique())
        mapping = {cluster_id: f"Cluster{cluster_id}" for cluster_id in unique_clusters}
        return df["Cluster"].map(mapping)

    mean_density = (
        df.groupby("Cluster")[density_col].mean().sort_values(ascending=False)
    )
    
    ordered_clusters = mean_density.index.tolist()
    human_labels = ["City", "Suburb", "Rural", "Cluster4", "Cluster5"]
    
    labels_to_use = human_labels[:len(ordered_clusters)]
    
    cluster_to_label = {c: labels_to_use[i] for i, c in enumerate(ordered_clusters)}
    
    all_clusters = df['Cluster'].unique()
    for c in all_clusters:
        if c not in cluster_to_label:
            print(f"[debug] WARN: Cluster {c} had no valid density data; labeling as 'Rural'.")
            cluster_to_label[c] = "Rural"
            
    return df["Cluster"].map(cluster_to_label)


def find_optimal_k(df: pd.DataFrame, feature_cols: List[str], max_k: int = 10, plot: bool = True) -> int:
    """Calculate and plot WCSS and Silhouette Scores to find optimal k.

    Args:
        df: DataFrame with data.
        feature_cols: List of column names to use for clustering.
        max_k: Maximum number of clusters to test.
        plot: If True, generate and save plots to `data/processed/`.

    Returns:
        The optimal number of clusters based on silhouette score.
    """
    wcss = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    print(f"Finding optimal k by testing k from 2 to {max_k}…")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        kmeans.fit(df[feature_cols])
        wcss.append(kmeans.inertia_)
        score = silhouette_score(df[feature_cols], kmeans.labels_)
        silhouette_scores.append(score)
        print(f"  k={k}, WCSS={wcss[-1]:.2f}, Silhouette Score={score:.4f}")

    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal k based on highest silhouette score: {optimal_k}")

    if plot:
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.plot(k_range, wcss, "bo-")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
        plt.title("Elbow Method for Optimal k")
        elbow_path = output_dir / "kmeans_elbow_plot.png"
        plt.savefig(elbow_path)
        print(f"Saved Elbow Method plot to {elbow_path}")
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(k_range, silhouette_scores, "go-")
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Average Silhouette Score")
        plt.title("Silhouette Scores for Number of Clusters")
        silhouette_path = output_dir / "kmeans_silhouette_plot.png"
        plt.savefig(silhouette_path)
        print(f"Saved Silhouette Score plot to {silhouette_path}")
        plt.close()

    return optimal_k


def run_clustering(
    input_path: Path, output_path: Path, k: Optional[int] = None, validate_k: bool = False
):
    """Entry point util to load data, create normalized features, cluster, and save."""
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(input_path)
    print(f"Loaded dataset with {len(df)} rows from '{input_path}'.")

    feature_cols = _select_feature_columns(df)

    df = add_normalised_features(df, feature_cols)

    norm_cols = [f"{c}_norm" for c in feature_cols]

    if validate_k or k is None:
        final_k = find_optimal_k(df, norm_cols)
    else:
        final_k = k

    kmeans = fit_kmeans(df, norm_cols, final_k)
    print(f"KMeans trained with k={final_k}.")

    df["ClusterLabel"] = _map_cluster_to_label(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved clustered dataset to '{output_path}'.")

    model_path = output_path.with_suffix(".kmeans.pkl")
    try:
        import joblib
        joblib.dump(kmeans, model_path)
        print(f"Saved trained model to '{model_path}'.")
    except ImportError:
        print("joblib not installed, skipping model serialization.")

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-means clustering on master dataset.")
    parser.add_argument("--input", default="data/processed/master_town_data.parquet", help="Input Parquet file path.")
    parser.add_argument("--output", default="data/processed/master_town_data.parquet", help="Path to write updated Parquet file.")
    parser.add_argument("--k", type=int, default=None, help="Number of clusters. If not provided, optimal k will be determined.")
    parser.add_argument("--validate", action="store_true", help="Force validation of k and generate plots.")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_clustering(Path(args.input), Path(args.output), k=args.k, validate_k=args.validate)


if __name__ == "__main__":
    main()
