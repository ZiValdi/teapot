from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.economics import EconomicAssumptions, calculate_economics


SAMPLED_PARAMETERS = {
    "Yield factor": "yield_factor",
    "Bio-liquid price": "bio_liquid_price_per_t",
    "Feedstock cost": "feedstock_cost_per_t",
    "Utilities": "utility_cost_per_t_feed",
    "Base CAPEX": "base_capex_meur",
}


def run_uncertainty_analysis(
    df: pd.DataFrame,
    assumptions: EconomicAssumptions,
    sample_count: int,
    variation_pct: float,
    seed: int = 42,
    return_samples: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if df.empty:
        empty_summary = pd.DataFrame(columns=["Indicator", "P5", "P50", "P95"])
        empty_sensitivity = pd.DataFrame(columns=["Parameter", "NPV correlation", "MSP correlation"])
        if return_samples:
            return empty_summary, empty_sensitivity, pd.DataFrame()
        return empty_summary, empty_sensitivity

    rng = np.random.default_rng(seed)
    variation = variation_pct / 100
    samples: list[dict[str, float]] = []
    outputs: list[dict[str, float]] = []

    for _ in range(sample_count):
        sampled_values = {
            key: rng.triangular(1 - variation, 1, 1 + variation)
            for key in SAMPLED_PARAMETERS.values()
        }
        sampled_assumptions = replace(
            assumptions,
            bio_liquid_price_per_t=assumptions.bio_liquid_price_per_t
            * sampled_values["bio_liquid_price_per_t"],
            feedstock_cost_per_t=assumptions.feedstock_cost_per_t
            * sampled_values["feedstock_cost_per_t"],
            utility_cost_per_t_feed=assumptions.utility_cost_per_t_feed
            * sampled_values["utility_cost_per_t_feed"],
            base_capex_meur=assumptions.base_capex_meur
            * sampled_values["base_capex_meur"],
        )
        sampled_df = df.copy()
        sampled_df["bio_liquid_yield_pct"] = (
            sampled_df["bio_liquid_yield_pct"] * sampled_values["yield_factor"]
        ).clip(0, 100)

        result = calculate_economics(sampled_df, sampled_assumptions)["raw"]
        samples.append(sampled_values)
        outputs.append(
            {
                "NPV": result["npv"],
                "MSP": result["msp"],
                "IRR": result["irr"] * 100 if result["irr"] is not None else np.nan,
                "Payback": result["payback"],
            }
        )

    output_df = pd.DataFrame(outputs)
    sample_df = pd.DataFrame(samples)
    summary = pd.DataFrame(
        [
            {
                "Indicator": "NPV",
                "P5": _format_meur(output_df["NPV"].quantile(0.05)),
                "P50": _format_meur(output_df["NPV"].quantile(0.50)),
                "P95": _format_meur(output_df["NPV"].quantile(0.95)),
            },
            {
                "Indicator": "MSP",
                "P5": _format_eur_per_t(output_df["MSP"].quantile(0.05)),
                "P50": _format_eur_per_t(output_df["MSP"].quantile(0.50)),
                "P95": _format_eur_per_t(output_df["MSP"].quantile(0.95)),
            },
            {
                "Indicator": "IRR",
                "P5": _format_pct(output_df["IRR"].quantile(0.05)),
                "P50": _format_pct(output_df["IRR"].quantile(0.50)),
                "P95": _format_pct(output_df["IRR"].quantile(0.95)),
            },
            {
                "Indicator": "Payback",
                "P5": _format_years(output_df["Payback"].quantile(0.05)),
                "P50": _format_years(output_df["Payback"].quantile(0.50)),
                "P95": _format_years(output_df["Payback"].quantile(0.95)),
            },
        ]
    )

    sensitivity_rows = []
    for label, column in SAMPLED_PARAMETERS.items():
        sensitivity_rows.append(
            {
                "Parameter": label,
                "NPV correlation": sample_df[column].corr(output_df["NPV"]),
                "MSP correlation": sample_df[column].corr(output_df["MSP"]),
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity["Abs NPV correlation"] = sensitivity["NPV correlation"].abs()
    sensitivity = sensitivity.sort_values("Abs NPV correlation", ascending=False).drop(
        columns="Abs NPV correlation"
    )
    sensitivity["NPV correlation"] = sensitivity["NPV correlation"].map(lambda value: round(value, 2))
    sensitivity["MSP correlation"] = sensitivity["MSP correlation"].map(lambda value: round(value, 2))

    if return_samples:
        return summary, sensitivity, output_df
    return summary, sensitivity, None

def _format_meur(value: float) -> str:
    return f"€{value / 1_000_000:.2f}M"


def _format_eur_per_t(value: float) -> str:
    return f"€{value:,.0f}/t"


def _format_pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.1f}%"


def _format_years(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.1f} yr"
