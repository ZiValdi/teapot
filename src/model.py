"""Simple KNN yield predictor based on process parameters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def predict_yield(
    df: pd.DataFrame,
    temperature: float,
    heating_rate: float,
    n2_flow: float,
    particle_size: float,
    k: int = 5,
) -> float:
    """
    Predict bio-liquid yield using k-nearest neighbours.

    Parameters
    ----------
    df : pd.DataFrame
        Clean dataset with columns: temperature_c, heating_rate_c_min,
        n2_flow_ml_min, particle_size_um, bio_liquid_yield_pct.
    temperature, heating_rate, n2_flow, particle_size : float
        Query parameters.
    k : int, optional
        Number of neighbours to average.

    Returns
    -------
    float
        Predicted yield in percent.
    """
    if df.empty:
        return np.nan

    features = ["temperature_c", "heating_rate_c_min", "n2_flow_ml_min", "particle_size_um"]
    X = df[features].values
    y = df["bio_liquid_yield_pct"].values

    # Standardize features
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # avoid division by zero
    X_scaled = (X - mean) / std

    query = np.array([temperature, heating_rate, n2_flow, particle_size])
    query_scaled = (query - mean) / std

    distances = cdist([query_scaled], X_scaled, metric="euclidean")[0]
    idx = np.argsort(distances)[:k]
    return float(np.mean(y[idx]))