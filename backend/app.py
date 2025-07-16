import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from src.scoring import calculate_recommendations
from pathlib import Path

from src.data_pipeline import build_master_dataframe
from src.clustering import run_clustering

DATA_PATH = Path('data/processed/master_town_data.parquet')

app = Flask(__name__)
CORS(app)

if not DATA_PATH.exists():
    print("[startup] master_town_data.parquet not found. Building dataset …")
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        df_master = build_master_dataframe()
        df_master.to_parquet(DATA_PATH, index=False)
        print(f"[startup] Saved fresh dataset to {DATA_PATH} (rows={len(df_master)})")

        run_clustering(DATA_PATH, DATA_PATH, k=3, validate_k=False)
    except Exception as exc:
        print(f"[startup] ERROR during automatic data build: {exc}")
        print("The API will start with empty dataset. Run 'make data cluster' later.")

try:
    master_town_data = pd.read_parquet(DATA_PATH)
    if 'ClusterLabel' not in master_town_data.columns:
        print("[startup] ClusterLabel missing; running clustering step …")
        try:
            run_clustering(DATA_PATH, DATA_PATH, k=3, validate_k=False)
            master_town_data = pd.read_parquet(DATA_PATH)
        except Exception as exc:
            print(f"[startup] ERROR: clustering failed ({exc}). Exiting application.")
            raise
    print("[startup] Cluster label distribution:")
    print(master_town_data['ClusterLabel'].value_counts(dropna=False))
except FileNotFoundError:
    print("ERROR: The master data file 'master_town_data.parquet' was not found even after attempted build.")
    raise

@app.route('/api/towns', methods=['GET'])
def get_towns():
    """
    Endpoint to get a list of all towns or filter by cluster.
    Query Parameters:
        cluster (str, optional): Filter towns by cluster label ('City', 'Suburb', 'Rural').
    """
    if master_town_data.empty:
        return jsonify({"error": "Data not available. Please check server logs."}), 500

    cluster_filter = request.args.get('cluster')

    if cluster_filter:
        filtered_data = master_town_data[master_town_data['ClusterLabel'] == cluster_filter]
    else:
        filtered_data = master_town_data

    towns = filtered_data.drop(columns=['geometry'], errors='ignore').to_dict(orient='records')
    return jsonify(towns)

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """
    Endpoint to get personalized town recommendations based on user preferences.
    Expects a JSON payload with 'cluster' and 'weights'.
    """
    if master_town_data.empty:
        return jsonify({"error": "Data not available. Please check server logs."}), 500

    data = request.get_json()
    if not data or 'cluster' not in data or 'weights' not in data:
        return jsonify({"error": "Missing 'cluster' or 'weights' in request body"}), 400

    cluster = data['cluster']
    user_weights = data['weights']

    candidate_towns = master_town_data[master_town_data['ClusterLabel'] == cluster]

    if candidate_towns.empty:
        return jsonify({"error": f"No towns found for cluster: {cluster}"}), 404

    recommendations = calculate_recommendations(candidate_towns, user_weights)

    print(f"[debug] Received user weights: {user_weights}")
    print(f"[debug] Top recommendation before sending: {recommendations[0] if recommendations else 'None'}")    

    return jsonify(recommendations)

@app.route('/')
def index():
    """
    A simple health-check endpoint to confirm the API is running.
    """
    return "SmartTownMatch Backend API is running!"

if __name__ == '__main__':
    app.run(debug=True)