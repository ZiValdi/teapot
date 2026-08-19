from __future__ import annotations

from dataclasses import replace
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
        "Equipment Costing": "active",
        "Feedstock": "coming",
        "Comparison": "coming",
        "Sensitivity": "coming",
        "Risk Analysis": "active",
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


def create_distribution_plot(
    samples: pd.Series,
    title: str = "",
    xlabel: str = "",
    target_value: float = None,
    target_label: str = "Target",
    show_cdf: bool = True,
) -> go.Figure:
    """
    Create a distribution plot with histogram, KDE, and percentile lines.
    Optionally show CDF overlay.
    """
    from scipy.stats import gaussian_kde

    # Compute KDE
    kde = gaussian_kde(samples)
    x_range = np.linspace(samples.min(), samples.max(), 200)
    kde_values = kde(x_range)

    fig = go.Figure()

    # Histogram (PDF)
    fig.add_trace(go.Histogram(
        x=samples,
        nbinsx=min(50, len(samples)//10),
        histnorm='probability density',
        name='PDF',
        marker=dict(color='#22d3ee', opacity=0.5, line=dict(color='#22d3ee', width=0.5)),
        hovertemplate='Value: %{x:.1f}<br>Density: %{y:.3f}<extra></extra>',
    ))

    # KDE curve
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values,
        mode='lines',
        name='KDE',
        line=dict(color='#22d3ee', width=2.5),
    ))

    # Percentile lines
    percentiles = [0.05, 0.50, 0.95]
    colors = ['#ffb4ab', '#4edea3', '#f59e0b']
    labels = ['VaR (P5)', 'Median', 'P95']
    for p, color, label in zip(percentiles, colors, labels):
        val = samples.quantile(p)
        fig.add_vline(
            x=val,
            line_dash='dash',
            line_color=color,
            annotation_text=f'{label}: {val:.1f}',
            annotation_position='top',
            annotation_font=dict(color=color, size=12),
        )

    # Mean line
    mean_val = samples.mean()
    fig.add_vline(
        x=mean_val,
        line_dash='solid',
        line_color='#22d3ee',
        annotation_text=f'Mean: {mean_val:.1f}',
        annotation_position='bottom',
        annotation_font=dict(color='#22d3ee', size=12),
    )

    # Target threshold (if provided)
    if target_value is not None:
        fig.add_vline(
            x=target_value,
            line_dash='dot',
            line_color='#ffb147',
            annotation_text=f'{target_label}: {target_value:.1f}%',
            annotation_position='top',
            annotation_font=dict(color='#ffb147', size=12),
        )
        # Probability of exceeding target
        prob_exceed = (samples > target_value).mean() * 100
        fig.add_annotation(
            x=0.95, y=0.9,
            xref='paper', yref='paper',
            text=f'Probability > {target_value:.1f}%: {prob_exceed:.1f}%',
            showarrow=False,
            font=dict(color='#ffb147', size=13),
            bgcolor='rgba(11,19,38,0.8)',
        )

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(color='#dae2fd', size=18)),
        xaxis=dict(title=xlabel, title_font=dict(color='#94a3b8'), tickfont=dict(color='#94a3b8'), gridcolor='#334155'),
        yaxis=dict(title='Density', title_font=dict(color='#94a3b8'), tickfont=dict(color='#94a3b8'), gridcolor='#334155'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#dae2fd'),
        legend=dict(font=dict(color='#dae2fd'), bgcolor='rgba(0,0,0,0)'),
        hovermode='x',
        margin=dict(l=40, r=40, t=80, b=40),
        height=400,
    )

    # Optional CDF overlay
    if show_cdf:
        sorted_samples = np.sort(samples)
        cdf_y = np.arange(1, len(sorted_samples)+1) / len(sorted_samples)
        fig.add_trace(go.Scatter(
            x=sorted_samples,
            y=cdf_y,
            mode='lines',
            name='CDF',
            line=dict(color='#4edea3', dash='dot', width=2),
            yaxis='y2',
        ))
        fig.update_layout(
            yaxis2=dict(
                title='Cumulative Probability',
                title_font=dict(color='#4edea3'),
                tickfont=dict(color='#4edea3'),
                gridcolor='rgba(0,0,0,0)',
                overlaying='y',
                side='right',
                range=[0, 1],
            ),
        )

    return fig


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
        # Now run_uncertainty_analysis always returns 3 values (summary, sensitivity, samples_df)
        uncertainty_summary, sensitivity, _ = run_uncertainty_analysis(
            df,
            assumptions,
            sample_count=samples,
            variation_pct=variation,
            return_samples=False,
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
    """
    Create a process flowsheet diagram with unit blocks.
    The highlighted unit (if any) gets a cyan border.
    """
    fig = go.Figure()

    # Unit positions (x, y) – horizontal layout
    units = {
        "Biomass Feed": (0.05, 0.5),
        "Rotary Dryer": (0.20, 0.5),
        "Pyrolyzer": (0.40, 0.5),
        "Cyclone": (0.55, 0.5),
        "Condenser": (0.70, 0.5),
        "Liquid Storage": (0.88, 0.5),
    }

    # Draw connecting lines with arrows
    main_path = ["Biomass Feed", "Rotary Dryer", "Pyrolyzer", "Cyclone", "Condenser", "Liquid Storage"]
    for i in range(len(main_path)-1):
        x0, y0 = units[main_path[i]]
        x1, y1 = units[main_path[i+1]]
        # Line
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

    # Draw unit rectangles with ports (small circles)
    for label, (x, y) in units.items():
        border_color = "#22d3ee" if label == highlight else "#334155"
        # Main block
        fig.add_shape(
            type="rect",
            x0=x-0.065, y0=y-0.09, x1=x+0.065, y1=y+0.09,
            line=dict(color=border_color, width=2),
            fillcolor="#1e293b",
            opacity=0.95,
            layer="below",
        )
        # Label
        fig.add_annotation(
            x=x, y=y,
            text=label,
            showarrow=False,
            font=dict(color="#dae2fd", size=11, family="Inter"),
            align="center",
        )

        # Ports (input on left, output on right) – only for units with connections
        if label in main_path and label != "Biomass Feed":
            # Input port (left)
            fig.add_shape(
                type="circle",
                x0=x-0.065-0.01, y0=y-0.015, x1=x-0.065+0.01, y1=y+0.015,
                fillcolor="#94a3b8",
                line=dict(color="#334155", width=1),
                layer="above",
            )
        if label in main_path and label != "Liquid Storage":
            # Output port (right)
            color = "#22d3ee" if label == highlight else "#94a3b8"
            fig.add_shape(
                type="circle",
                x0=x+0.065-0.01, y0=y-0.015, x1=x+0.065+0.01, y1=y+0.015,
                fillcolor=color,
                line=dict(color="#334155", width=1),
                layer="above",
            )

    # Additional labels (N₂, gas)
    fig.add_annotation(
        x=0.40, y=0.70,
        text="N₂ Supply",
        showarrow=False,
        font=dict(color="#f59e0b", size=9),
    )
    fig.add_annotation(
        x=0.70, y=0.70,
        text="Gas to flue",
        showarrow=False,
        font=dict(color="#94a3b8", size=9),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        font=dict(color="#dae2fd"),
        hovermode=False,
    )

    return fig


def show_process_modeler(df: pd.DataFrame, assumptions: EconomicAssumptions, run_uncertainty: bool, samples: int, variation: float) -> None:
    """Process Modeler with clickable unit blocks, property inspector, and live economics."""
    st.title("Process Modeler")
    st.markdown(
        '<div class="status-line">● Design your pyrolysis process – select a unit to configure</div>',
        unsafe_allow_html=True,
    )

    # Define units and their parameters (ranges, steps, initial values)
    units = {
        "Biomass Feed": {
            "icon": "🌾",
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
            "params": {
                "Temperature (°C)": 550,
                "Pressure (atm)": 1.0,
                "Heating Rate (°C/min)": 50,
                "Residence Time (s)": 2,
                "N₂ Flow (mL/min)": 100,
                "Particle Size (μm)": 500,
            },
            "range": {
                "Temperature (°C)": (300, 800),
                "Pressure (atm)": (0.5, 2.0),
                "Heating Rate (°C/min)": (1, 300),
                "Residence Time (s)": (0.5, 10),
                "N₂ Flow (mL/min)": (0, 500),
                "Particle Size (μm)": (50, 2000),
            },
            "step": {
                "Temperature (°C)": 5,
                "Pressure (atm)": 0.1,
                "Heating Rate (°C/min)": 1,
                "Residence Time (s)": 0.5,
                "N₂ Flow (mL/min)": 10,
                "Particle Size (μm)": 50,
            },
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

    # Initialize session state for selected unit and parameters
    if "selected_unit" not in st.session_state:
        st.session_state.selected_unit = "Pyrolyzer"
    if "unit_params" not in st.session_state:
        st.session_state.unit_params = {u: units[u]["params"].copy() for u in units}
    # Ensure all parameters are present
    for u, data in units.items():
        for p, v in data["params"].items():
            if p not in st.session_state.unit_params.get(u, {}):
                st.session_state.unit_params[u][p] = v

    # Layout: main canvas (left) and property inspector (right)
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Process Flowsheet")
        # Row of unit selection buttons (clickable blocks)
        unit_cols = st.columns(len(units))
        for idx, (unit_name, data) in enumerate(units.items()):
            with unit_cols[idx]:
                icon = data.get("icon", "⚙️")
                if st.button(
                    f"{icon} {unit_name}",
                    key=f"unit_btn_{unit_name}",
                    use_container_width=True,
                    type="primary" if st.session_state.selected_unit == unit_name else "secondary",
                ):
                    st.session_state.selected_unit = unit_name
                    st.rerun()

        # Display the flowsheet with highlighted unit
        fig = create_flowsheet_figure(highlight=st.session_state.selected_unit)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Floating toolbar (mockup)
        st.markdown(
            """
            <div style="display:flex; justify-content:center; gap:8px; margin-top:8px;">
                <span style="background:#1e293b; border:1px solid #334155; color:#94a3b8; padding:6px 12px; border-radius:20px; font-size:12px;">🔍 Zoom In</span>
                <span style="background:#1e293b; border:1px solid #334155; color:#94a3b8; padding:6px 12px; border-radius:20px; font-size:12px;">🔍 Zoom Out</span>
                <span style="background:#1e293b; border:1px solid #334155; color:#94a3b8; padding:6px 12px; border-radius:20px; font-size:12px;">➕ Add Unit</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### Property Inspector")
        unit_name = st.session_state.selected_unit
        data = units[unit_name]
        st.markdown(f"**{data['icon']} {unit_name}**")
        st.caption("Adjust parameters to see impact on yield and economics.")

        # Editable parameters for the selected unit
        params = st.session_state.unit_params[unit_name]
        updated = False
        for param, value in params.items():
            min_val, max_val = data["range"].get(param, (0, 100))
            step = data["step"].get(param, 1)
            # Use number_input for numeric parameters
            new_val = st.number_input(
                param,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(value),
                step=float(step),
                key=f"param_{unit_name}_{param}",
                format="%g",
            )
            if new_val != value:
                params[param] = new_val
                updated = True

        if updated:
            # Update session state
            st.session_state.unit_params[unit_name] = params
            # Force rerun to update predictions
            st.rerun()

        # Extract key parameters for yield prediction (from Pyrolyzer)
        pyro = st.session_state.unit_params.get("Pyrolyzer", {})
        temp = pyro.get("Temperature (°C)", 550)
        heating_rate = pyro.get("Heating Rate (°C/min)", 50)
        n2_flow = pyro.get("N₂ Flow (mL/min)", 100)
        particle_size = pyro.get("Particle Size (μm)", 500)

        # Predict yield using KNN
        predicted_yield = predict_yield(
            df,
            temperature=temp,
            heating_rate=heating_rate,
            n2_flow=n2_flow,
            particle_size=particle_size,
            k=5,
        )

        # Display predicted yield and economics
        st.markdown("### Predicted Bio‑Liquid Yield")
        st.metric(label="", value=f"{predicted_yield:.1f}%", delta=None)

        # Run economics on a single row with the predicted yield
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

        # Product yield table (estimated from prediction)
        st.markdown("### Estimated Product Yields")
        # Simple model: bio-oil = predicted_yield, gas = 20% of total, char = 15% of total
        bio_oil = predicted_yield
        gas = 0.2 * predicted_yield
        char = 0.15 * predicted_yield
        total = bio_oil + gas + char
        if total > 0:
            bio_oil_pct = bio_oil / total * 100
            gas_pct = gas / total * 100
            char_pct = char / total * 100
        else:
            bio_oil_pct = gas_pct = char_pct = 0
        yield_df = pd.DataFrame({
            "Component": ["Bio‑oil", "Syngas", "Biochar"],
            "wt %": [f"{bio_oil_pct:.1f}", f"{gas_pct:.1f}", f"{char_pct:.1f}"],
        })
        st.dataframe(yield_df, use_container_width=True, hide_index=True)


def show_risk_analysis(df: pd.DataFrame, assumptions: EconomicAssumptions) -> None:
    """Risk Analysis page – Monte Carlo uncertainty and sensitivity."""
    st.title("Risk Analysis")
    st.markdown(
        '<div class="status-line">● Monte Carlo uncertainty analysis – assess economic risk</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Simulation Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        samples = st.number_input("Number of samples", min_value=50, max_value=5000, value=1000, step=50, key="risk_samples")
    with col2:
        variation = st.slider("Input variation (±%)", min_value=5, max_value=50, value=20, step=5, key="risk_variation")
    with col3:
        seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1, key="risk_seed")

    target_irr = st.number_input("Target IRR (%)", min_value=0.0, max_value=50.0, value=15.0, step=0.5, key="risk_target")

    # Generate a key based on current parameters to detect changes
    params_key = f"{samples}_{variation}_{seed}"

    # Initialize session state for results if not present
    if "risk_results" not in st.session_state:
        st.session_state.risk_results = None
    if "risk_params_key" not in st.session_state:
        st.session_state.risk_params_key = None

    # Run button
    run_button = st.button("▶ Run Monte Carlo Simulation", use_container_width=True, type="primary")

    # Determine if we need to run (button clicked or parameters changed)
    should_run = run_button or (st.session_state.risk_results is None) or (st.session_state.risk_params_key != params_key)

    if should_run:
        with st.spinner("Running Monte Carlo simulation..."):
            summary, sensitivity, samples_df = run_uncertainty_analysis(
                df,
                assumptions,
                sample_count=samples,
                variation_pct=variation,
                seed=seed,
                return_samples=True,
            )
            if summary.empty or sensitivity.empty or samples_df is None:
                st.warning("Analysis returned no results. Please adjust settings and try again.")
                return
            st.session_state.risk_results = {
                "summary": summary,
                "sensitivity": sensitivity,
                "samples_df": samples_df,
            }
            st.session_state.risk_params_key = params_key
            # Rerun to display results
            st.rerun()

    # Display results if available
    if st.session_state.risk_results is not None:
        results = st.session_state.risk_results
        summary = results["summary"]
        sensitivity = results["sensitivity"]
        samples_df = results["samples_df"]

        # Extract samples
        irr_samples = samples_df["IRR"].dropna()
        npv_samples = samples_df["NPV"].dropna()

        # --- KPI Cards ---
        st.markdown("### Key Risk Metrics")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Expected IRR", f"{irr_samples.mean():.1f}%", delta="Mean")
        with k2:
            var = irr_samples.quantile(0.05)
            st.metric("Value at Risk (VaR 5%)", f"{var:.1f}%", delta="P5", delta_color="inverse")
        with k3:
            p95 = irr_samples.quantile(0.95)
            st.metric("Upside (P95)", f"{p95:.1f}%", delta="P95")
        with k4:
            st.metric("Std Deviation", f"{irr_samples.std():.1f}%")

        st.markdown("---")

        # --- Distribution Plot ---
        st.markdown("### IRR Distribution")
        show_cdf = st.toggle("Show CDF overlay", value=True, key="risk_show_cdf")
        fig = create_distribution_plot(
            irr_samples,
            title="",
            xlabel="IRR (%)",
            target_value=target_irr,
            target_label="Target IRR",
            show_cdf=show_cdf,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

        # --- Variance Contribution Chart ---
        st.markdown("### Variance Drivers")
        # Compute variance contribution from absolute correlation (normalized)
        sensitivity["abs_contrib"] = sensitivity["NPV correlation"].abs()
        total_contrib = sensitivity["abs_contrib"].sum()
        sensitivity["contrib_pct"] = (sensitivity["abs_contrib"] / total_contrib * 100).round(1)

        # Sort descending for horizontal bar
        sensitivity_sorted = sensitivity.sort_values("contrib_pct", ascending=True)

        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Bar(
            y=sensitivity_sorted["Parameter"],
            x=sensitivity_sorted["contrib_pct"],
            orientation='h',
            marker_color='#22d3ee',
            text=sensitivity_sorted["contrib_pct"].map(lambda v: f"{v:.1f}%"),
            textposition='outside',
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
        ))
        fig_contrib.update_layout(
            title="Contribution to NPV Variance",
            xaxis_title="Contribution (%)",
            yaxis_title="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#dae2fd'),
            height=350,
            margin=dict(l=10, r=40, t=40, b=10),
            xaxis=dict(gridcolor='#334155'),
        )
        st.plotly_chart(fig_contrib, use_container_width=True, config={'displayModeBar': False})

        st.markdown("---")

        # --- Tables ---
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### Percentile Summary")
            st.dataframe(summary, use_container_width=True, hide_index=True)
        with col_right:
            st.markdown("#### Scenario Table")
            scenario_data = []
            for indicator in summary["Indicator"]:
                p5 = summary[summary["Indicator"] == indicator]["P5"].values[0]
                p50 = summary[summary["Indicator"] == indicator]["P50"].values[0]
                p95 = summary[summary["Indicator"] == indicator]["P95"].values[0]
                scenario_data.append({"Indicator": indicator, "Worst (P5)": p5, "Base (P50)": p50, "Best (P95)": p95})
            scenario_df = pd.DataFrame(scenario_data)
            st.dataframe(scenario_df, use_container_width=True, hide_index=True)

        # Probability of success
        prob_success = (irr_samples > target_irr).mean() * 100
        st.success(f"✅ Probability of IRR exceeding {target_irr:.1f}%: **{prob_success:.1f}%**")
    else:
        # Show info message if no results yet
        st.info("Click the 'Run Monte Carlo Simulation' button to start the analysis.")

def show_equipment_costing(df: pd.DataFrame, assumptions: EconomicAssumptions) -> None:
    """Equipment Costing page – adjust CAPEX category multipliers and see impact on economics."""
    st.title("Equipment Costing")
    st.markdown(
        '<div class="status-line">● Adjust cost multipliers for each equipment category</div>',
        unsafe_allow_html=True,
    )

    # Get base economics with current assumptions
    base_econ = calculate_economics(df, assumptions)
    base_capex_breakdown = base_econ.get("capex_breakdown", {})
    if not base_capex_breakdown:
        st.warning("No CAPEX breakdown available. Please check your data and assumptions.")
        return

    # Compute default total CAPEX
    default_total = sum(base_capex_breakdown.values())
    categories = list(base_capex_breakdown.keys())
    default_costs = list(base_capex_breakdown.values())

    st.markdown("### Adjust Cost Multipliers")
    st.caption("Each multiplier scales the default cost for that equipment category. Changes update economics in real time.")

    # Use session state to store multipliers
    if "cost_multipliers" not in st.session_state:
        st.session_state.cost_multipliers = {cat: 1.0 for cat in categories}

    # Display sliders for each category
    col1, col2 = st.columns([1, 1])
    multipliers = {}
    with col1:
        for cat in categories[:len(categories)//2]:
            multipliers[cat] = st.slider(
                f"{cat}",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.cost_multipliers.get(cat, 1.0),
                step=0.05,
                key=f"cost_{cat}",
            )
    with col2:
        for cat in categories[len(categories)//2:]:
            multipliers[cat] = st.slider(
                f"{cat}",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.cost_multipliers.get(cat, 1.0),
                step=0.05,
                key=f"cost_{cat}",
            )

    # Update session state
    for cat, val in multipliers.items():
        st.session_state.cost_multipliers[cat] = val

    # Compute new costs
    new_costs = {cat: default_costs[i] * multipliers[cat] for i, cat in enumerate(categories)}
    new_total = sum(new_costs.values())

    # Update assumptions with new CAPEX (override base_capex_meur)
    # We keep capacity constant, so base_capex_meur = new_total / 1e6
    new_assumptions = replace(
        assumptions,
        base_capex_meur=new_total / 1_000_000,
    )

    # Recalculate economics with new assumptions
    new_econ = calculate_economics(df, new_assumptions)
    new_kpis = new_econ["kpis"]

    # Display results
    st.markdown("### Results")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total CAPEX", new_kpis["capex"], delta=f"{(new_total/default_total - 1)*100:.1f}%")
    with k2:
        st.metric("NPV", new_kpis["npv"])
    with k3:
        st.metric("IRR", new_kpis["irr"])
    with k4:
        st.metric("Payback", new_kpis["payback"])

    # Two columns: table and chart
    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.markdown("#### Cost Breakdown")
        # Create a comparison table
        comparison_data = []
        for i, cat in enumerate(categories):
            comparison_data.append({
                "Category": cat,
                "Default (€M)": f"{default_costs[i]/1_000_000:.2f}",
                "Multiplier": f"{multipliers[cat]:.2f}x",
                "Adjusted (€M)": f"{new_costs[cat]/1_000_000:.2f}",
            })
        comp_df = pd.DataFrame(comparison_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with col_chart:
        st.markdown("#### CAPEX Breakdown")
        # Plot adjusted breakdown
        fig = breakdown_chart("CAPEX", new_costs)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Optional: Show cash flow impact
    with st.expander("Updated Cash Flow"):
        cf = new_econ["cash_flow_table"].copy()
        for col in ["Net Cash Flow", "Discounted Cash Flow", "Cumulative Cash Flow"]:
            cf[col] = cf[col].map(lambda v: f"€{v/1_000_000:.2f}M")
        st.dataframe(cf, use_container_width=True, hide_index=True)

    # Reset button
    if st.button("Reset to Defaults", use_container_width=True):
        for cat in categories:
            st.session_state.cost_multipliers[cat] = 1.0
        st.rerun()

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
    elif st.session_state.page == "Process Modeler":
        show_process_modeler(df, assumptions, run_uncertainty, samples, variation)
    elif st.session_state.page == "Risk Analysis":
        show_risk_analysis(df, assumptions)
    elif st.session_state.page == "Equipment Costing":
        show_equipment_costing(df, assumptions)
    else:
        st.title(st.session_state.page)
        st.info("This page is under development. Please check back later.")


if __name__ == "__main__":
    main()