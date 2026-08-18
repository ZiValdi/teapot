from __future__ import annotations

import pandas as pd


def build_recommendations(df: pd.DataFrame) -> list[dict[str, str]]:
    if df.empty:
        return [
            {
                "title": "No Data In Current Filter",
                "body": "Relax the selected ranges or upload a CSV with the required process parameters.",
                "tone": "warning",
            }
        ]

    recommendations: list[dict[str, str]] = []

    if df["temperature_c"].nunique() >= 3:
        temp_groups = pd.qcut(df["temperature_c"], q=3, duplicates="drop")
        by_temp = df.groupby(temp_groups, observed=True)["bio_liquid_yield_pct"].mean()
        if not by_temp.empty:
            best_window = by_temp.idxmax()
            recommendations.append(
                {
                    "title": "Highest Yield Region",
                    "body": (
                        f"Experiments between {best_window.left:.0f}-"
                        f"{best_window.right:.0f} °C show the highest average bio-liquid yield."
                    ),
                    "tone": "positive",
                }
            )

    if df["heating_rate_c_min"].nunique() > 1:
        yield_spread = df["bio_liquid_yield_pct"].max() - df["bio_liquid_yield_pct"].min()
        rate_min = df["heating_rate_c_min"].min()
        rate_max = df["heating_rate_c_min"].max()
        if yield_spread >= 8:
            recommendations.append(
                {
                    "title": "Investigate Heating Rate",
                    "body": (
                        f"Yield spans {yield_spread:.1f} percentage points across "
                        f"{rate_min:.0f}-{rate_max:.0f} °C/min."
                    ),
                    "tone": "warning",
                }
            )

    if df["lignin_pct"].nunique() > 1 and len(df) >= 6:
        median_lignin = df["lignin_pct"].median()
        high = df[df["lignin_pct"] >= median_lignin]["bio_liquid_yield_pct"].mean()
        low = df[df["lignin_pct"] < median_lignin]["bio_liquid_yield_pct"].mean()
        delta = high - low
        if abs(delta) >= 2:
            direction = "higher" if delta > 0 else "lower"
            recommendations.append(
                {
                    "title": "Feedstock Composition",
                    "body": (
                        f"High-lignin samples average {abs(delta):.1f} percentage points "
                        f"{direction} yield than lower-lignin samples."
                    ),
                    "tone": "neutral",
                }
            )

    missing_particle = df["particle_size_um"].isna().mean() * 100
    mismatch_particle = df["particle_size_mismatch"].mean() * 100
    if missing_particle or mismatch_particle:
        recommendations.append(
            {
                "title": "Data Quality",
                "body": (
                    f"Particle-size data is missing in {missing_particle:.0f}% and "
                    f"unit-mismatched in {mismatch_particle:.0f}% of filtered observations."
                ),
                "tone": "warning",
            }
        )
    else:
        recommendations.append(
            {
                "title": "Particle Size Coverage",
                "body": "Particle-size fields are complete and consistent after conversion to micrometers.",
                "tone": "positive",
            }
        )

    return recommendations[:4]
