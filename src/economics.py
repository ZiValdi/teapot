from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EconomicAssumptions:
    capacity_tpd: float
    operating_days: int
    bio_liquid_price_per_t: float
    feedstock_cost_per_t: float
    utility_cost_per_t_feed: float
    labor_cost_per_year: float
    base_capex_meur: float
    base_capacity_tpd: float
    scaling_exponent: float
    maintenance_pct_capex: float
    discount_rate: float
    project_life_years: int


def _format_meur(value: float) -> str:
    return f"€{value / 1_000_000:.2f}M"


def _format_eur_per_t(value: float) -> str:
    return f"€{value:,.0f}/t"


def _npv(discount_rate: float, cash_flows: list[float]) -> float:
    return sum(value / ((1 + discount_rate) ** year) for year, value in enumerate(cash_flows))


def _irr(cash_flows: list[float]) -> float | None:
    low, high = -0.95, 1.5
    for _ in range(120):
        mid = (low + high) / 2
        value = _npv(mid, cash_flows)
        if abs(value) < 1:
            return mid
        if value > 0:
            low = mid
        else:
            high = mid
    result = (low + high) / 2
    return result if -0.95 < result < 1.5 else None


def _capital_recovery_factor(discount_rate: float, years: int) -> float:
    if years <= 0:
        return 0
    if discount_rate == 0:
        return 1 / years
    return discount_rate * (1 + discount_rate) ** years / ((1 + discount_rate) ** years - 1)


def calculate_economics(df: pd.DataFrame, assumptions: EconomicAssumptions) -> dict:
    if df.empty:
        return {
            "kpis": {
                "capex": "N/A",
                "annual_opex": "N/A",
                "ebitda": "N/A",
                "npv": "N/A",
                "irr": "N/A",
                "payback": "N/A",
                "msp": "N/A",
                "annual_bio_liquid": "N/A",
            },
            "raw": {},
            "capex_breakdown": {},
            "opex_breakdown": {},
            "cash_flow_table": pd.DataFrame(),
        }

    yield_fraction = max(float(df["bio_liquid_yield_pct"].mean()) / 100, 0)
    annual_feed_t = assumptions.capacity_tpd * assumptions.operating_days
    annual_bio_liquid_t = annual_feed_t * yield_fraction

    installed_capex = (
        assumptions.base_capex_meur
        * 1_000_000
        * (assumptions.capacity_tpd / assumptions.base_capacity_tpd) ** assumptions.scaling_exponent
    )
    capex_breakdown = {
        "Reactor Island": installed_capex * 0.38,
        "Feed Handling": installed_capex * 0.16,
        "Condensation": installed_capex * 0.18,
        "Utilities": installed_capex * 0.12,
        "Installation & Contingency": installed_capex * 0.16,
    }

    feedstock = annual_feed_t * assumptions.feedstock_cost_per_t
    utilities = annual_feed_t * assumptions.utility_cost_per_t_feed
    maintenance = installed_capex * assumptions.maintenance_pct_capex / 100
    labor = assumptions.labor_cost_per_year
    opex_breakdown = {
        "Feedstock": feedstock,
        "Utilities": utilities,
        "Maintenance": maintenance,
        "Labor": labor,
    }
    annual_opex = sum(opex_breakdown.values())
    annual_revenue = annual_bio_liquid_t * assumptions.bio_liquid_price_per_t
    ebitda = annual_revenue - annual_opex
    cash_flows = [-installed_capex] + [ebitda] * assumptions.project_life_years
    npv = _npv(assumptions.discount_rate / 100, cash_flows)
    irr = _irr(cash_flows)
    payback = installed_capex / ebitda if ebitda > 0 else np.nan
    annualized_capex = installed_capex * _capital_recovery_factor(
        assumptions.discount_rate / 100, assumptions.project_life_years
    )
    msp = (annual_opex + annualized_capex) / annual_bio_liquid_t if annual_bio_liquid_t else np.nan

    cash_flow_table = pd.DataFrame(
        {
            "Year": list(range(0, assumptions.project_life_years + 1)),
            "Net Cash Flow": cash_flows,
            "Discounted Cash Flow": [
                value / ((1 + assumptions.discount_rate / 100) ** year)
                for year, value in enumerate(cash_flows)
            ],
        }
    )
    cash_flow_table["Cumulative Cash Flow"] = cash_flow_table["Net Cash Flow"].cumsum()

    return {
        "kpis": {
            "capex": _format_meur(installed_capex),
            "annual_opex": _format_meur(annual_opex),
            "ebitda": _format_meur(ebitda),
            "npv": _format_meur(npv),
            "irr": f"{irr * 100:.1f}%" if irr is not None else "N/A",
            "payback": f"{payback:.1f} yr" if np.isfinite(payback) else "N/A",
            "msp": _format_eur_per_t(msp) if np.isfinite(msp) else "N/A",
            "annual_bio_liquid": f"{annual_bio_liquid_t:,.0f} t/y",
        },
        "raw": {
            "yield_pct": yield_fraction * 100,
            "annual_feed_t": annual_feed_t,
            "annual_bio_liquid_t": annual_bio_liquid_t,
            "installed_capex": installed_capex,
            "annual_revenue": annual_revenue,
            "annual_opex": annual_opex,
            "ebitda": ebitda,
            "npv": npv,
            "irr": irr,
            "payback": payback,
            "msp": msp,
        },
        "capex_breakdown": capex_breakdown,
        "opex_breakdown": opex_breakdown,
        "cash_flow_table": cash_flow_table,
    }


def create_economic_summary_table(economics: dict) -> pd.DataFrame:
    raw = economics.get("raw", {})
    if not raw:
        return pd.DataFrame(columns=["Metric", "Value"])

    rows = [
        ("Average process yield", f"{raw['yield_pct']:.1f}%"),
        ("Annual feed processed", f"{raw['annual_feed_t']:,.0f} t/y"),
        ("Annual bio-liquid production", f"{raw['annual_bio_liquid_t']:,.0f} t/y"),
        ("Annual revenue", _format_meur(raw["annual_revenue"])),
        ("Annual OPEX", _format_meur(raw["annual_opex"])),
        ("EBITDA", _format_meur(raw["ebitda"])),
        ("NPV", _format_meur(raw["npv"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])
