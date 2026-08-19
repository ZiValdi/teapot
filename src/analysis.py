from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_kpis(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {
            "average_yield": "N/A",
            "maximum_yield": "N/A",
            "experiment_count": "0",
            "temperature_range": "N/A",
        }

    return {
        "average_yield": f"{df['bio_liquid_yield_pct'].mean():.1f}%",
        "maximum_yield": f"{df['bio_liquid_yield_pct'].max():.1f}%",
        "experiment_count": f"{len(df):,}",
        "temperature_range": (
            f"{df['temperature_c'].min():.0f}-{df['temperature_c'].max():.0f} °C"
        ),
    }


def create_temperature_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group experiments by temperature into at most 3 groups.
    Handles datasets with very few unique temperatures gracefully.
    """
    if df.empty:
        return pd.DataFrame(
            columns=["Process-data Group", "Temperature Window", "Average Yield", "Maximum Yield", "Experiments"]
        )

    unique_temps = df["temperature_c"].unique()
    n_unique = len(unique_temps)

    # Single temperature – one group
    if n_unique == 1:
        temp_val = unique_temps[0]
        group_name = f"Single Temperature ({temp_val:.0f} °C)"
        row = {
            "Process-data Group": group_name,
            "Temperature Window": f"{temp_val:.0f} °C",
            "Average Yield": f"{df['bio_liquid_yield_pct'].mean():.1f}%",
            "Maximum Yield": f"{df['bio_liquid_yield_pct'].max():.1f}%",
            "Experiments": len(df),
        }
        return pd.DataFrame([row])

    # Determine number of groups (max 3, but at most n_unique)
    n_groups = min(3, n_unique)

    # Compute quantile-based bin edges
    quantiles = np.linspace(0, 1, n_groups + 1)
    bin_edges = df["temperature_c"].quantile(quantiles).values

    # Remove duplicate edges to avoid empty bins
    bin_edges = np.unique(bin_edges)

    # If after dedup we have only one edge, fallback to a single group
    if len(bin_edges) < 2:
        temp_min = df["temperature_c"].min()
        temp_max = df["temperature_c"].max()
        group_name = f"Single Group ({temp_min:.0f}-{temp_max:.0f} °C)"
        row = {
            "Process-data Group": group_name,
            "Temperature Window": f"{temp_min:.0f}-{temp_max:.0f} °C",
            "Average Yield": f"{df['bio_liquid_yield_pct'].mean():.1f}%",
            "Maximum Yield": f"{df['bio_liquid_yield_pct'].max():.1f}%",
            "Experiments": len(df),
        }
        return pd.DataFrame([row])

    # Generate labels dynamically based on the number of bins
    labels = [f"Group {i+1}" for i in range(len(bin_edges) - 1)]

    # Assign groups using pd.cut
    grouped = df.copy()
    grouped["Process-data Group"] = pd.cut(
        df["temperature_c"],
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
    )

    # Aggregation
    table = (
        grouped.groupby("Process-data Group", observed=True)
        .agg(
            **{
                "Average Yield": ("bio_liquid_yield_pct", "mean"),
                "Maximum Yield": ("bio_liquid_yield_pct", "max"),
                "Experiments": ("bio_liquid_yield_pct", "size"),
                "Temperature Min": ("temperature_c", "min"),
                "Temperature Max": ("temperature_c", "max"),
            }
        )
        .reset_index()
    )

    # Add temperature window
    table["Temperature Window"] = table.apply(
        lambda row: f"{row['Temperature Min']:.0f}-{row['Temperature Max']:.0f} °C",
        axis=1,
    )

    # Rename groups to Low/Medium/High based on temperature order
    if len(table) == 3:
        table["Process-data Group"] = ["Low Temperature", "Medium Temperature", "High Temperature"]
    elif len(table) == 2:
        table["Process-data Group"] = ["Low Temperature", "High Temperature"]
    # If len == 1, already handled above, but keep fallback
    # Drop temporary columns
    table = table.drop(columns=["Temperature Min", "Temperature Max"])

    # Format yield columns
    table["Average Yield"] = table["Average Yield"].map(lambda v: f"{v:.1f}%")
    table["Maximum Yield"] = table["Maximum Yield"].map(lambda v: f"{v:.1f}%")

    return table[["Process-data Group", "Temperature Window", "Average Yield", "Maximum Yield", "Experiments"]]


def calculate_feedstock_composition(df: pd.DataFrame) -> dict[str, float]:
    columns = {
        "Cellulose": "cellulose_pct",
        "Hemicellulose": "hemicellulose_pct",
        "Lignin": "lignin_pct",
    }
    return {
        label: float(df[column].mean())
        for label, column in columns.items()
        if column in df and df[column].notna().any()
    }