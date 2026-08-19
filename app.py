from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    calculate_feedstock_composition,
    calculate_kpis,
    create_temperature_groups,
)
from src.data import load_data
from src.economics import (
    EconomicAssumptions,
    calculate_economics,
    create_economic_summary_table,
)
from src.model import predict_yield
from src.recommendations import build_recommendations
from src.uncertainty import run_uncertainty_analysis


DATA_PATH = Path("data/experiments.csv")
DEFAULT_ECONOMICS = EconomicAssumptions(
    capacity_tpd=100,
    operating_days=330,
    bio_liquid_price_per_t=650,
    feedstock_cost_per_t=75,
    utility_cost_per_t_feed=45,
    labor_cost_per_year=900_000,
    base_capex_meur=18,
    base_capacity_tpd=100,
    scaling_exponent=0.65,
    maintenance_pct_capex=4,
    discount_rate=10,
    project_life_years=15,
)


st.set_page_config(
    page_title="Agile Lab TEA",
    page_icon="assets/icon.webp",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

        :root {
            --surface: #0b1326;
            --panel: #172337;
            --card: #1e293b;
            --border: #334155;
            --text: #dae2fd;
            --muted: #94a3b8;
            --cyan: #22d3ee;
            --emerald: #10b981;
            --amber: #f59e0b;
        }

        .stApp {
            background: var(--surface);
            color: var(--text);
            font-family: Inter, sans-serif;
        }

        [data-testid="stSidebar"] {
            background: #131b2e;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * { color: var(--text); }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }

        h1, h2, h3 { color: var(--text); letter-spacing: 0; }

        .status-line {
            color: var(--cyan);
            font-family: "JetBrains Mono", monospace;
            font-size: 0.78rem;
            letter-spacing: 0.04em;
            margin-top: -0.5rem;
            text-transform: uppercase;
        }

        .kpi-card, .section-panel, .rec-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            font-family: "JetBrains Mono", monospace;
        }

        .kpi-value {
            color: var(--text);
            font-family: "JetBrains Mono", monospace;
            font-size: 1.65rem;
            font-weight: 600;
            margin-top: 0.35rem;
        }

        .section-title {
            color: var(--text);
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }

        .rec-title {
            color: var(--text);
            font-weight: 700;
            font-size: 0.94rem;
            margin-bottom: 0.35rem;
        }

        .rec-body {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }

        .badge {
            display: inline-block;
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--muted);
            font-family: "JetBrains Mono", monospace;
            font-size: 0.72rem;
            margin-top: 0.75rem;
            padding: 3px 7px;
            text-transform: uppercase;
        }

        .positive { border-left: 3px solid var(--emerald); }
        .warning { border-left: 3px solid var(--amber); }
        .neutral { border-left: 3px solid var(--cyan); }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        /* Sidebar navigation – plain text buttons */
        section[data-testid="stSidebar"] .stButton {
            display: flex;
            justify-content: flex-start;
            margin: 0 !important;
            padding: 0 !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            background: transparent !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px;
            color: var(--text) !important;
            font-weight: 400;
            text-align: left;
            padding: 4px 12px !important;
            font-size: 0.95rem;
            width: 100%;
            display: flex;
            justify-content: flex-start;
            margin-bottom: -10px;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background: rgba(34, 211, 238, 0.1) !important;
            color: var(--cyan) !important;
        }
        section[data-testid="stSidebar"] .stButton button[data-baseweb="button"][data-kind="primary"] {
            color: var(--cyan) !important;
            font-weight: 600;
            border-color: var(--cyan) !important;
        }
        /* Coming soon items – dotted border */
        .coming-soon {
            color: #64748b;
            font-size: 0.95rem;
            padding: 4px 12px;
            border: 1px dashed var(--border) !important;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return load_data(uploaded_file)
    return load_data(DATA_PATH)


def sidebar_controls(df: pd.DataFrame) -> tuple[pd.DataFrame, EconomicAssumptions, bool, int, float]:
    """Shared sidebar: logo, file upload, navigation, economic assumptions."""
    st.sidebar.image("assets/icon.webp", width=80)
    st.sidebar.markdown("### Project teapot")
    st.sidebar.caption("Pyrolysis process-data MVP")

    uploaded_file = st.sidebar.file_uploader("CSV dataset", type=["csv"])
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
        except ValueError as error:
            st.sidebar.error(str(error))
            st.stop()
        except Exception as error:
            st.sidebar.error(f"Could not read uploaded CSV: {error}")
            st.stop()

    # Navigation
    st.sidebar.markdown("### Navigation")

    pages = {
        "Dashboard": "active",
        "Process Modeler": "active",
        "Feedstock": "coming",
        "Comparison": "coming",
        "Sensitivity": "coming",
        "Risk Analysis": "coming",
        "Documentation": "coming",
    }

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    for page_name, status in pages.items():
        if status == "active":
            button_type = "primary" if st.session_state.page == page_name else "secondary"
            if st.sidebar.button(
                page_name,
                key=f"nav_{page_name}",
                use_container_width=True,
                type=button_type,
            ):
                st.session_state.page = page_name
                st.rerun()
        else:
            st.sidebar.markdown(f'<div class="coming-soon">{page_name} (coming later)</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Economic assumptions (popover)
    with st.sidebar.popover("Assumptions settings", width="stretch"):
        st.markdown("#### Economic assumptions")
        st.caption("Preliminary TEA inputs for the current process-data filter.")
        capacity_tpd = st.number_input(
            "Plant capacity (t feed/day)",
            min_value=1.0,
            max_value=5000.0,
            value=float(DEFAULT_ECONOMICS.capacity_tpd),
            step=10.0,
        )
        operating_days = st.number_input(
            "Operating days/year",
            min_value=1,
            max_value=365,
            value=DEFAULT_ECONOMICS.operating_days,
            step=5,
        )
        bio_liquid_price_per_t = st.number_input(
            "Bio-liquid selling price (€/t)",
            min_value=0.0,
            max_value=5000.0,
            value=float(DEFAULT_ECONOMICS.bio_liquid_price_per_t),
            step=25.0,
        )
        feedstock_cost_per_t = st.number_input(
            "Feedstock cost (€/t)",
            min_value=-500.0,
            max_value=1000.0,
            value=float(DEFAULT_ECONOMICS.feedstock_cost_per_t),
            step=5.0,
        )
        utility_cost_per_t_feed = st.number_input(
            "Utilities (€/t feed)",
            min_value=0.0,
            max_value=1000.0,
            value=float(DEFAULT_ECONOMICS.utility_cost_per_t_feed),
            step=5.0,
        )
        labor_cost_per_year = st.number_input(
            "Labor (€/year)",
            min_value=0.0,
            max_value=20_000_000.0,
            value=float(DEFAULT_ECONOMICS.labor_cost_per_year),
            step=50_000.0,
        )

        st.markdown("#### Capital and finance")
        base_capex_meur = st.number_input(
            "Base CAPEX (€M)",
            min_value=0.1,
            max_value=1000.0,
            value=float(DEFAULT_ECONOMICS.base_capex_meur),
            step=1.0,
        )
        base_capacity_tpd = st.number_input(
            "Base capacity (t/day)",
            min_value=1.0,
            max_value=5000.0,
            value=float(DEFAULT_ECONOMICS.base_capacity_tpd),
            step=10.0,
        )
        scaling_exponent = st.slider(
            "Scaling exponent",
            min_value=0.3,
            max_value=1.0,
            value=float(DEFAULT_ECONOMICS.scaling_exponent),
            step=0.01,
        )
        maintenance_pct_capex = st.slider(
            "Maintenance (% CAPEX/year)",
            min_value=0.0,
            max_value=20.0,
            value=float(DEFAULT_ECONOMICS.maintenance_pct_capex),
            step=0.5,
        )
        discount_rate = st.slider(
            "Discount rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(DEFAULT_ECONOMICS.discount_rate),
            step=0.5,
        )
        project_life_years = st.number_input(
            "Project life (years)",
            min_value=1,
            max_value=40,
            value=DEFAULT_ECONOMICS.project_life_years,
            step=1,
        )

        st.markdown("#### Uncertainty")
        run_uncertainty = st.toggle("Enable Monte Carlo panel", value=True)
        uncertainty_samples = st.slider("Samples", min_value=50, max_value=1000, value=250, step=50)
        uncertainty_variation = st.slider(
            "Input variation (+/- %)",
            min_value=1.0,
            max_value=50.0,
            value=20.0,
            step=1.0,
        )

    assumptions = EconomicAssumptions(
        capacity_tpd=capacity_tpd,
        operating_days=int(operating_days),
        bio_liquid_price_per_t=bio_liquid_price_per_t,
        feedstock_cost_per_t=feedstock_cost_per_t,
        utility_cost_per_t_feed=utility_cost_per_t_feed,
        labor_cost_per_year=labor_cost_per_year,
        base_capex_meur=base_capex_meur,
        base_capacity_tpd=base_capacity_tpd,
        scaling_exponent=scaling_exponent,
        maintenance_pct_capex=maintenance_pct_capex,
        discount_rate=discount_rate,
        project_life_years=int(project_life_years),
    )
    return df, assumptions, run_uncertainty, uncertainty_samples, uncertainty_variation


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def breakdown_chart(title: str, values: dict[str, float]) -> go.Figure:
    total = sum(values.values())
    center = title.replace(" ", "<br>")
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(values.keys()),
                values=list(values.values()),
                hole=0.62,
                marker={
                    "colors": ["#22d3ee", "#10b981", "#f59e0b", "#347AB0", "#ff5a65"],
                    "line": {"color": "#0b1326", "width": 2},
                },
                textinfo="label+percent",
                textfont={"color": "#dae2fd", "size": 12},
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 10, "b": 8},
        height=310,
        showlegend=False,
        annotations=[
            {
                "text": f"{center}<br>€{total / 1_000_000:.1f}M",
                "x": 0.5,
                "y": 0.5,
                "font": {"color": "#dae2fd", "size": 15},
                "showarrow": False,
            }
        ],
    )
    return fig


def sensitivity_chart(sensitivity: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                y=sensitivity["Parameter"],
                x=sensitivity["NPV correlation"],
                orientation="h",
                marker_color=[
                    "#10b981" if value >= 0 else "#ff5a65"
                    for value in sensitivity["NPV correlation"]
                ],
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 10, "b": 8},
        height=290,
        xaxis={"range": [-1, 1], "gridcolor": "#334155", "zerolinecolor": "#dae2fd"},
        yaxis={"tickfont": {"color": "#dae2fd"}},
        font={"color": "#dae2fd"},
        showlegend=False,
    )
    return fig


def composition_chart(composition: dict[str, float]) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(composition.keys()),
                values=list(composition.values()),
                hole=0.62,
                marker={
                    "colors": ["#22d3ee", "#10b981", "#f59e0b"],
                    "line": {"color": "#0b1326", "width": 2},
                },
                textinfo="label+percent",
                textfont={"color": "#dae2fd", "size": 13},
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 10, "b": 8},
        height=330,
        showlegend=False,
        annotations=[
            {
                "text": "Average<br>Feedstock",
                "x": 0.5,
                "y": 0.5,
                "font": {"color": "#dae2fd", "size": 16},
                "showarrow": False,
            }
        ],
    )
    return fig


def recommendation_card(item: dict[str, str]) -> None:
    tone = item.get("tone", "neutral")
    st.markdown(
        f"""
        <div class="rec-card {tone}">
            <div class="rec-title">{item["title"]}</div>
            <div class="rec-body">{item["body"]}</div>
            <div class="badge">Dataset-derived</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_dashboard(df: pd.DataFrame, assumptions: EconomicAssumptions, run_uncertainty: bool, samples: int, variation: float) -> None:
    """Dashboard view – expects the df to be already filtered."""
    economics = calculate_economics(df, assumptions)

    st.write("")
    economic_kpis = economics["kpis"]
    kpi_columns = st.columns(4)
    with kpi_columns[0]:
        kpi_card("Estimated CAPEX", economic_kpis["capex"])
    with kpi_columns[1]:
        kpi_card("Annual OPEX", economic_kpis["annual_opex"])
    with kpi_columns[2]:
        kpi_card("NPV", economic_kpis["npv"])
    with kpi_columns[3]:
        kpi_card("Minimum Selling Price", economic_kpis["msp"])

    process_kpis = calculate_kpis(df)
    st.write("")
    process_columns = st.columns(4)
    with process_columns[0]:
        kpi_card("Average Bio-Liquid Yield", process_kpis["average_yield"])
    with process_columns[1]:
        kpi_card("Annual Bio-Liquid", economic_kpis["annual_bio_liquid"])
    with process_columns[2]:
        kpi_card("IRR", economic_kpis["irr"])
    with process_columns[3]:
        kpi_card("Payback", economic_kpis["payback"])

    st.write("")
    table_col, chart_col = st.columns([1.45, 1])

    with table_col:
        st.markdown('<div class="section-title">Economic Summary</div>', unsafe_allow_html=True)
        st.dataframe(
            create_economic_summary_table(economics),
            use_container_width=True,
            hide_index=True,
        )

    with chart_col:
        st.markdown('<div class="section-title">Annual OPEX Breakdown</div>', unsafe_allow_html=True)
        if economics["opex_breakdown"]:
            st.plotly_chart(
                breakdown_chart("Annual OPEX", economics["opex_breakdown"]),
                use_container_width=True,
            )
        else:
            st.info("Economic assumptions are unavailable in the current filter.")

    st.write("")
    lower_left, lower_right = st.columns([1.45, 1])

    with lower_left:
        st.markdown('<div class="section-title">Process-Data Group Comparison</div>', unsafe_allow_html=True)
        st.dataframe(
            create_temperature_groups(df),
            use_container_width=True,
            hide_index=True,
        )

    with lower_right:
        st.markdown('<div class="section-title">CAPEX Breakdown</div>', unsafe_allow_html=True)
        if economics["capex_breakdown"]:
            st.plotly_chart(
                breakdown_chart("CAPEX", economics["capex_breakdown"]),
                use_container_width=True,
            )
        else:
            st.info("CAPEX is unavailable in the current filter.")

    st.write("")
    cash_flow_col, composition_col = st.columns([1.45, 1])
    with cash_flow_col:
        st.markdown('<div class="section-title">Project Cash Flow</div>', unsafe_allow_html=True)
        cash_flow_table = economics["cash_flow_table"].copy()
        if not cash_flow_table.empty:
            for column in ["Net Cash Flow", "Discounted Cash Flow", "Cumulative Cash Flow"]:
                cash_flow_table[column] = cash_flow_table[column].map(lambda value: f"€{value / 1_000_000:.2f}M")
        st.dataframe(cash_flow_table, use_container_width=True, hide_index=True)

    with composition_col:
        st.markdown('<div class="section-title">Average Feedstock Composition</div>', unsafe_allow_html=True)
        composition = calculate_feedstock_composition(df)
        if composition:
            st.plotly_chart(composition_chart(composition), use_container_width=True)
        else:
            st.info("Composition fields are unavailable in the current filter.")

    if run_uncertainty:
        st.write("")
        st.markdown('<div class="section-title">Uncertainty and Sensitivity</div>', unsafe_allow_html=True)
        uncertainty_summary, sensitivity = run_uncertainty_analysis(
            df,
            assumptions,
            sample_count=samples,
            variation_pct=variation,
        )
        uncertainty_col, sensitivity_col = st.columns([1.1, 1])
        with uncertainty_col:
            st.dataframe(uncertainty_summary, use_container_width=True, hide_index=True)
        with sensitivity_col:
            if not sensitivity.empty:
                st.plotly_chart(sensitivity_chart(sensitivity), use_container_width=True)
            else:
                st.info("Sensitivity results are unavailable in the current filter.")

    st.write("")
    st.markdown('<div class="section-title">Decision Recommendations</div>', unsafe_allow_html=True)
    rec_columns = st.columns(4)
    for column, item in zip(rec_columns, build_recommendations(df)):
        with column:
            recommendation_card(item)

def create_flowsheet_figure(highlight: str = None) -> go.Figure:
    """Create a flowsheet diagram with unit blocks; highlight the selected unit."""
    fig = go.Figure()

    # Define unit positions (x, y) for a horizontal layout
    units = {
        "Biomass Feed": (0.05, 0.5),
        "Rotary Dryer": (0.20, 0.5),
        "Pyrolyzer": (0.40, 0.5),
        "Cyclone": (0.55, 0.5),
        "Condenser": (0.70, 0.5),
        "Liquid Storage": (0.88, 0.5),
    }

    # Draw connecting lines (pipes) with arrows
    main_path = ["Biomass Feed", "Rotary Dryer", "Pyrolyzer", "Cyclone", "Condenser", "Liquid Storage"]
    for i in range(len(main_path)-1):
        x0, y0 = units[main_path[i]]
        x1, y1 = units[main_path[i+1]]
        fig.add_shape(
            type="line",
            x0=x0 + 0.03, y0=y0, x1=x1 - 0.03, y1=y1,
            line=dict(color="#22d3ee", width=2),
        )
        # Arrowhead
        fig.add_annotation(
            x=(x0 + x1)/2, y=y0,
            ax=x0 + 0.05, ay=0, xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowcolor="#22d3ee",
            arrowwidth=2,
        )

    # Draw unit blocks
    for label, (x, y) in units.items():
        # Determine if this unit is the highlighted one
        border_color = "#22d3ee" if label == highlight else "#334155"
        shadow = "rgba(34,211,238,0.3)" if label == highlight else "transparent"

        fig.add_shape(
            type="rect",
            x0=x-0.06, y0=y-0.08, x1=x+0.06, y1=y+0.08,
            line=dict(color=border_color, width=2),
            fillcolor="#1e293b",
            opacity=0.9,
            layer="below",
        )
        fig.add_annotation(
            x=x, y=y,
            text=label,
            showarrow=False,
            font=dict(color="#dae2fd", size=11, family="Inter"),
            align="center",
        )

    # Add a decorative box around the figure
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        font=dict(color="#dae2fd"),
    )

    return fig

def show_process_modeler(df: pd.DataFrame, assumptions: EconomicAssumptions, run_uncertainty: bool, samples: int, variation: float) -> None:
    """Process Modeler with clickable unit blocks and property inspector."""
    st.title("Process Modeler")
    st.markdown(
        '<div class="status-line">● Design your pyrolysis process – select a unit to configure</div>',
        unsafe_allow_html=True,
    )

    # Define available unit operations
    units = {
        "Biomass Feed": {
            "icon": "🍃",
            "params": {"Flow (kg/h)": 500, "Moisture (%)": 15},
            "range": {"Flow (kg/h)": (50, 2000), "Moisture (%)": (0, 50)},
            "step": {"Flow (kg/h)": 10, "Moisture (%)": 1},
        },
        "Rotary Dryer": {
            "icon": "🌡️",
            "params": {"Temperature (°C)": 120, "Moisture Out (%)": 10},
            "range": {"Temperature (°C)": (50, 300), "Moisture Out (%)": (0, 20)},
            "step": {"Temperature (°C)": 5, "Moisture Out (%)": 0.5},
        },
        "Pyrolyzer": {
            "icon": "🔥",
            "params": {"Temperature (°C)": 550, "Pressure (atm)": 1.0, "Heating Rate (°C/min)": 50, "Residence Time (s)": 2},
            "range": {"Temperature (°C)": (300, 800), "Pressure (atm)": (0.5, 2.0), "Heating Rate (°C/min)": (1, 300), "Residence Time (s)": (0.5, 10)},
            "step": {"Temperature (°C)": 5, "Pressure (atm)": 0.1, "Heating Rate (°C/min)": 1, "Residence Time (s)": 0.5},
        },
        "Cyclone": {
            "icon": "🌀",
            "params": {"Efficiency (%)": 90, "Pressure Drop (kPa)": 1.5},
            "range": {"Efficiency (%)": (50, 99), "Pressure Drop (kPa)": (0.5, 5)},
            "step": {"Efficiency (%)": 1, "Pressure Drop (kPa)": 0.1},
        },
        "Condenser": {
            "icon": "❄️",
            "params": {"Outlet Temp (°C)": 40, "Cooling Duty (kW)": -250},
            "range": {"Outlet Temp (°C)": (20, 80), "Cooling Duty (kW)": (-500, -50)},
            "step": {"Outlet Temp (°C)": 5, "Cooling Duty (kW)": 10},
        },
        "Liquid Storage": {
            "icon": "🛢️",
            "params": {"Tank Volume (m³)": 50, "Level (%)": 80},
            "range": {"Tank Volume (m³)": (10, 200), "Level (%)": (0, 100)},
            "step": {"Tank Volume (m³)": 5, "Level (%)": 5},
        },
    }

    # Store selected unit in session state
    if "selected_unit" not in st.session_state:
        st.session_state.selected_unit = "Pyrolyzer"

    # Layout: main canvas (left) and property inspector (right)
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Process Flowsheet")
        # Display a clean flowsheet (Plotly) – we reuse the previous figure but with a note that blocks are clickable
        st.markdown("""
        <div style="border:1px solid #334155; border-radius:8px; padding:16px; background: #172337;">
            <p style="color:#94a3b8; font-size:0.9rem; text-align:center;">
                <span style="color:#22d3ee;">●</span> Click a unit in the diagram below to configure it.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # We can display a more detailed Plotly flowsheet, or we can use an HTML mockup.
        # For simplicity, I'll use a static Plotly figure with unit labels as clickable annotations.
        # Since Plotly annotations cannot trigger callbacks, we'll use a selectbox or a set of buttons above the diagram.

        # But to make it look like the mockup, we can display an SVG or HTML with clickable areas,
        # but that's complex. Instead, we'll place a row of unit buttons above the figure.
        # These buttons act as the "clickable" units.

        # Create a row of unit selection buttons
        unit_cols = st.columns(len(units))
        for idx, (unit_name, _) in enumerate(units.items()):
            with unit_cols[idx]:
                if st.button(
                    f"{units[unit_name]['icon']} {unit_name}",
                    key=f"unit_btn_{unit_name}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_unit == unit_name else "secondary",
                ):
                    st.session_state.selected_unit = unit_name
                    st.rerun()

        # Now render the flowsheet as a Plotly figure (using the improved create_flowsheet_figure)
        # We'll highlight the selected unit by changing its border color.
        fig = create_flowsheet_figure(highlight=st.session_state.selected_unit)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        st.markdown("### Property Inspector")
        # Show parameters for the selected unit
        unit = units[st.session_state.selected_unit]
        st.markdown(f"**{unit['icon']} {st.session_state.selected_unit}**")
        st.caption("Adjust parameters to see impact on yield and economics.")

        # Create a dictionary to store updated parameters
        updated_params = {}
        for param, value in unit["params"].items():
            min_val, max_val = unit["range"].get(param, (0, 100))
            step = unit["step"].get(param, 1)
            # For integer parameters, use number_input; for others, use slider
            if isinstance(value, int):
                new_val = st.number_input(
                    param,
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(value),
                    step=int(step),
                    key=f"param_{st.session_state.selected_unit}_{param}",
                )
            else:
                new_val = st.slider(
                    param,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(value),
                    step=float(step),
                    key=f"param_{st.session_state.selected_unit}_{param}",
                )
            updated_params[param] = new_val

        # Update the unit's parameters in the units dict (stored in session state for persistence)
        if "unit_params" not in st.session_state:
            st.session_state.unit_params = {}
        # We'll store all unit params in a nested dict
        if st.session_state.selected_unit not in st.session_state.unit_params:
            st.session_state.unit_params[st.session_state.selected_unit] = unit["params"].copy()
        # Update with new values
        for k, v in updated_params.items():
            st.session_state.unit_params[st.session_state.selected_unit][k] = v

        # Now use these parameters to drive the yield prediction
        # Extract key parameters that affect yield (temp, heating rate, etc.)
        pyro_params = st.session_state.unit_params.get("Pyrolyzer", {})
        temp = pyro_params.get("Temperature (°C)", 550)
        heating_rate = pyro_params.get("Heating Rate (°C/min)", 50)
        # For N2 flow and particle size, we can also include them from other units or keep as sliders.
        # For simplicity, we can still use the sliders from the original Process Modeler,
        # but now we can also read from the inspector.

        # Predict yield using the current parameters
        # We'll use the same predict_yield function, but pass the values from the inspector.
        # However, we still need N2 flow and particle size – we can have them as global sliders or derive from units.
        # Let's keep them as separate sliders outside the inspector, or include them in the "Pyrolyzer" unit.
        # I'll add them to the Pyrolyzer parameters above.

        # After updating the parameters, we can recompute yield and economics
        predicted_yield = predict_yield(
            df,
            temperature=temp,
            heating_rate=heating_rate,
            n2_flow=100,  # default or from another unit
            particle_size=500,  # default
            k=5,
        )

        # Display the predicted yield and economics as before
        st.markdown("### Predicted Bio‑Liquid Yield")
        st.metric(label="", value=f"{predicted_yield:.1f}%", delta=None)

        # Prepare single-row DataFrame for economics
        sample_row = df.iloc[[0]].copy()
        sample_row["bio_liquid_yield_pct"] = predicted_yield
        economics = calculate_economics(sample_row, assumptions)
        economic_kpis = economics["kpis"]

        st.markdown("### Economic Projection")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.metric("NPV", economic_kpis["npv"])
            st.metric("IRR", economic_kpis["irr"])
        with ecol2:
            st.metric("Payback", economic_kpis["payback"])
            st.metric("CAPEX", economic_kpis["capex"])

def main() -> None:
    inject_css()
    base_df = get_data(None)
    df, assumptions, run_uncertainty, samples, variation = sidebar_controls(base_df)

    if st.session_state.page == "Dashboard":
        # Header with gear icon popover for filters
        col1, col2 = st.columns([6, 1])
        with col1:
            st.title("Economic Decision Support")
            st.markdown(
                f'<div class="status-line">● Process dataset active · {len(df):,} rows · preliminary TEA mode</div>',
                unsafe_allow_html=True,
            )
        with col2:
            with st.popover("⚙️ Dashboard Controls", use_container_width=True):
                temp_min, temp_max = float(df["temperature_c"].min()), float(df["temperature_c"].max())
                if "dash_temp_min" not in st.session_state:
                    st.session_state.dash_temp_min = temp_min
                    st.session_state.dash_temp_max = temp_max
                    st.session_state.dash_rate_min = float(df["heating_rate_c_min"].min())
                    st.session_state.dash_rate_max = float(df["heating_rate_c_min"].max())

                selected_temp = st.slider(
                    "Temperature range (°C)",
                    min_value=temp_min,
                    max_value=temp_max,
                    value=(st.session_state.dash_temp_min, st.session_state.dash_temp_max),
                    step=5.0,
                    key="dash_temp_slider",
                )
                st.session_state.dash_temp_min, st.session_state.dash_temp_max = selected_temp

                rate_min, rate_max = float(df["heating_rate_c_min"].min()), float(df["heating_rate_c_min"].max())
                selected_rate = st.slider(
                    "Heating rate (°C/min)",
                    min_value=rate_min,
                    max_value=rate_max,
                    value=(st.session_state.dash_rate_min, st.session_state.dash_rate_max),
                    step=1.0,
                    key="dash_rate_slider",
                )
                st.session_state.dash_rate_min, st.session_state.dash_rate_max = selected_rate

        # Apply filters
        filtered_df = df[
            df["temperature_c"].between(st.session_state.dash_temp_min, st.session_state.dash_temp_max)
            & df["heating_rate_c_min"].between(st.session_state.dash_rate_min, st.session_state.dash_rate_max)
        ].copy()

        show_dashboard(filtered_df, assumptions, run_uncertainty, samples, variation)
    else:
        show_process_modeler(df, assumptions, run_uncertainty, samples, variation)


if __name__ == "__main__":
    main()