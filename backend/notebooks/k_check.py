from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from backend.src.data_pipeline import build_master_dataframe
from backend.src.clustering import run_clustering

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "processed" / "master_town_data.parquet"

FEATURE_COLS = ["EducationScore", "SafetyScore", "WalkScore"]


def ensure_dataset() -> None:
    """Build the master dataset (and clusters) if it does not yet exist."""
    if DATA_PATH.exists():
        return

    print("[evaluation] Dataset not found – building master dataframe …")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_master = build_master_dataframe()
    df_master.to_parquet(DATA_PATH, index=False)
    print(f"[evaluation] Saved fresh dataset to {DATA_PATH} (rows={len(df_master)})")

    run_clustering(DATA_PATH, DATA_PATH, k=3, validate_k=False)


def evaluate_clustering(df: pd.DataFrame, *, k_min: int = 2, k_max: int = 9) -> None:
    """Compute and plot Elbow & Silhouette metrics for K-Means clustering."""
    print("[evaluation] Starting clustering evaluation …")

    features = df[FEATURE_COLS].dropna()
    if features.empty:
        raise ValueError(
            "No rows available after dropping NA for columns " f"{FEATURE_COLS}"
        )

    inertia_vals = []
    silhouette_vals = []
    k_values = range(k_min, k_max + 1)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        inertia_vals.append(kmeans.inertia_)
        score = silhouette_score(features, labels)
        silhouette_vals.append(score)
        print(
            f"[evaluation] k={k} | Inertia={kmeans.inertia_:,.0f} | "
            f"Silhouette Score={score:.4f}"
        )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertia_vals, "bo-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_vals, "ro-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ensure_dataset()
    master_df = pd.read_parquet(DATA_PATH)
    evaluate_clustering(master_df) 