import pandas as pd
import numpy as np

# NOTE TO SELF: Monte-Carlo ranking uncertainty
# If *_ImputedStd columns exist for SafetyScore / EducationScore the function
# draws `n_draws` samples from N(mean, std) (clipped to [0,1]) for those towns
# and recomputes match scores. We return the mean as MatchScore plus the 5th
# and 95th percentile as MatchScoreLow / MatchScoreHigh.


def calculate_recommendations(candidate_towns_df, user_weights, top_n=10, n_draws: int = 20):
    """
    Calculates and ranks town recommendations based on user preference weights.

    Args:
        candidate_towns_df (pd.DataFrame): DataFrame of towns filtered by cluster.
        user_weights (dict): A dictionary of user-defined weights for each feature.
        top_n (int): The number of top recommendations to return.

    Returns:
        list: A sorted list of dictionaries, each representing a recommended town.
    """
    base_feature_directions = {
        'MedianHomePrice_norm': -1,
        'SafetyScore_norm': 1,
        'EducationScore_norm': 1,
        'WalkScore_norm': 1,
        'AmenitiesScore_norm': 1,
        'TransitScore_norm': 1,
        'BikeScore_norm': 1,
        'PopulationDensity_norm': 0,
    }

    feature_directions = {f: d for f, d in base_feature_directions.items() if f in candidate_towns_df.columns}

    if not feature_directions:
        raise ValueError("No usable feature columns found in dataset for scoring.")

    features_to_use = list(feature_directions.keys())

    df_features = candidate_towns_df[features_to_use].copy()
    df_features = df_features.apply(lambda col: col.fillna(col.median()).fillna(0))

    std_map = {}
    if "SafetyScore_norm" in features_to_use and "SafetyScoreImputedStd" in candidate_towns_df.columns:
        rng = candidate_towns_df["SafetyScore"].max() - candidate_towns_df["SafetyScore"].min()
        if rng == 0:
            rng = 1
        std_map["SafetyScore_norm"] = candidate_towns_df["SafetyScoreImputedStd"] / rng
    if "EducationScore_norm" in features_to_use and "EducationScoreImputedStd" in candidate_towns_df.columns:
        rng = candidate_towns_df["EducationScore"].max() - candidate_towns_df["EducationScore"].min()
        if rng == 0:
            rng = 1
        std_map["EducationScore_norm"] = candidate_towns_df["EducationScoreImputedStd"] / rng

    preference_vector = []
    for feature in features_to_use:
        weight_key = feature.replace('_norm', '')
        
        preference_vector.append(user_weights.get(weight_key, 0))

    n_towns = len(candidate_towns_df)
    all_scores = np.zeros((n_towns, n_draws))

    rng_global = np.random.default_rng(42)

    for draw_idx in range(n_draws):
        sim_feats = df_features.copy()

        for feat, std_series in std_map.items():
            noise = rng_global.normal(loc=0.0, scale=1.0, size=n_towns)
            sim_feats[feat] = np.clip(sim_feats[feat] + noise * std_series, 0, 1)

        sim_matrix = sim_feats.values.copy()

        for col_idx, feature in enumerate(features_to_use):
            if feature_directions[feature] == -1:
                sim_matrix[:, col_idx] = 1 - sim_matrix[:, col_idx]

        all_scores[:, draw_idx] = np.dot(sim_matrix, preference_vector)

    mean_scores = all_scores.mean(axis=1)
    low_scores = np.percentile(all_scores, 5, axis=1)
    high_scores = np.percentile(all_scores, 95, axis=1)

    recommendations_df = candidate_towns_df.copy()
    recommendations_df['MatchScore'] = mean_scores
    recommendations_df['MatchScoreLow'] = low_scores
    recommendations_df['MatchScoreHigh'] = high_scores

    recommendations_df = recommendations_df.sort_values(by='MatchScore', ascending=False)
    top_recommendations = recommendations_df.head(top_n)

    if 'geometry' in top_recommendations.columns:
        top_recommendations = top_recommendations.drop(columns=['geometry'])

    pd.set_option('future.no_silent_downcasting', True)
    return top_recommendations.fillna(0).to_dict(orient='records')