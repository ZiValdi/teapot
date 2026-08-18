from __future__ import annotations

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
    if df.empty:
        return pd.DataFrame(
            columns=["Process-data Group", "Average Yield", "Maximum Yield", "Experiments"]
        )

    grouped = df.copy()
    grouped["Process-data Group"] = pd.qcut(
        grouped["temperature_c"],
        q=min(3, grouped["temperature_c"].nunique()),
        labels=["Low Temperature", "Medium Temperature", "High Temperature"][
            : min(3, grouped["temperature_c"].nunique())
        ],
        duplicates="drop",
    )

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
    table["Temperature Window"] = table.apply(
        lambda row: f"{row['Temperature Min']:.0f}-{row['Temperature Max']:.0f} °C",
        axis=1,
    )
    table["Average Yield"] = table["Average Yield"].map(lambda value: f"{value:.1f}%")
    table["Maximum Yield"] = table["Maximum Yield"].map(lambda value: f"{value:.1f}%")
    return table[
        [
            "Process-data Group",
            "Temperature Window",
            "Average Yield",
            "Maximum Yield",
            "Experiments",
        ]
    ]


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
