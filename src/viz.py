"""
Plotting helpers — all return Plotly figures for use in both the Streamlit
dashboard and notebooks.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent colour palette
STATUS_COLORS = {
    "Normal": "#2ecc71",
    "Underperforming": "#f39c12",
    "Fault/Stop": "#e74c3c",
    "Curtailed": "#9b59b6",
    "Low Wind": "#95a5a6",
}

COMPONENT_COLORS = px.colors.qualitative.Set2


# ---------------------------------------------------------------------------
# KPI strip (returns dict of display values — rendered as st.metric in app)
# ---------------------------------------------------------------------------

def fmt_kpi(kpis: dict) -> dict[str, str]:
    """Format fleet KPI dict into display strings."""
    avail = kpis.get("fleet_availability_pct")
    mtbf  = kpis.get("fleet_mtbf_hours")
    fails = kpis.get("total_failures", 0)
    down  = kpis.get("total_downtime_hours", 0)

    return {
        "Fleet Availability": f"{avail:.1f} %" if avail is not None else "N/A",
        "Fleet MTBF": f"{mtbf:,.0f} h" if mtbf is not None else "N/A",
        "Failure Events": str(fails),
        "Total Downtime": f"{down:,.0f} h",
    }


# ---------------------------------------------------------------------------
# 1. Power curve scatter
# ---------------------------------------------------------------------------

def plot_power_curve(
    scada: pd.DataFrame,
    turbine_id: Optional[str] = None,
    max_points: int = 15_000,
    rated_power_kw: float = 2000.0,
) -> go.Figure:
    """Wind speed vs. active power, coloured by operational status.

    Parameters
    ----------
    scada:
        SCADA dataframe — must have wind_speed_ms, active_power_kw,
        operational_status (add via features.label_power_curve_status).
    turbine_id:
        Filter to one turbine; plots all if None.
    max_points:
        Downsample to this many points for rendering performance.
    """
    df = scada.copy()
    if turbine_id and "turbine_id" in df.columns:
        df = df[df["turbine_id"] == turbine_id]

    required = {"wind_speed_ms", "active_power_kw"}
    if not required.issubset(df.columns):
        fig = go.Figure()
        fig.add_annotation(text="Missing wind_speed_ms or active_power_kw columns",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)

    status_col = "operational_status" if "operational_status" in df.columns else None

    if status_col:
        fig = px.scatter(
            df,
            x="wind_speed_ms",
            y="active_power_kw",
            color=status_col,
            color_discrete_map=STATUS_COLORS,
            opacity=0.4,
            labels={"wind_speed_ms": "Wind Speed (m/s)", "active_power_kw": "Active Power (kW)"},
            title=f"Power Curve{' — ' + turbine_id if turbine_id else ''}",
        )
    else:
        fig = px.scatter(
            df,
            x="wind_speed_ms",
            y="active_power_kw",
            opacity=0.3,
            labels={"wind_speed_ms": "Wind Speed (m/s)", "active_power_kw": "Active Power (kW)"},
            title=f"Power Curve{' — ' + turbine_id if turbine_id else ''}",
        )

    # Theoretical power curve overlay (Betz-limited cubic)
    ws_range = np.linspace(0, 25, 200)
    p_theory = np.clip(rated_power_kw * ((ws_range / 12) ** 3), 0, rated_power_kw)
    fig.add_trace(go.Scatter(
        x=ws_range, y=p_theory,
        mode="lines",
        name="Theoretical",
        line=dict(color="black", width=1.5, dash="dash"),
    ))

    fig.update_layout(
        xaxis_title="Wind Speed (m/s)",
        yaxis_title="Active Power (kW)",
        legend_title="Status",
        height=420,
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="#f9f9f9",
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Failure Pareto chart
# ---------------------------------------------------------------------------

def plot_pareto(
    pareto_df: pd.DataFrame,
    value_label: str = "Failure Count",
    title: str = "Failure Pareto by Component",
) -> go.Figure:
    """Combined bar + cumulative % line Pareto chart.

    Parameters
    ----------
    pareto_df:
        Output of rca.failure_pareto() or rca.downtime_pareto().
    """
    if pareto_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No failure data available",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=pareto_df["component"],
            y=pareto_df["count"],
            name=value_label,
            marker_color="#e74c3c",
            text=pareto_df["count"],
            textposition="outside",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=pareto_df["component"],
            y=pareto_df["cumulative_pct"],
            name="Cumulative %",
            mode="lines+markers",
            line=dict(color="#2c3e50", width=2),
            marker=dict(size=6),
        ),
        secondary_y=True,
    )

    # 80 % line
    fig.add_hline(y=80, secondary_y=True, line_dash="dot",
                  line_color="grey", annotation_text="80%")

    fig.update_layout(
        title=title,
        height=420,
        legend=dict(orientation="h", y=1.08),
        plot_bgcolor="#f9f9f9",
        margin=dict(l=50, r=50, t=60, b=60),
    )
    fig.update_yaxes(title_text=value_label, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
    fig.update_xaxes(title_text="Component")
    return fig


# ---------------------------------------------------------------------------
# 3. Temperature trend with anomaly windows
# ---------------------------------------------------------------------------

def plot_temp_trend(
    scada: pd.DataFrame,
    turbine_id: Optional[str] = None,
    temp_col: str = "temp_delta",
    zscore_col: str = "temp_zscore",
    logbook: Optional[pd.DataFrame] = None,
    title: str = "Gearbox Temperature Trend & Anomaly Windows",
) -> go.Figure:
    """Temperature over time with pre-failure anomaly windows shaded.

    Expects scada to already have pre_failure_window, temp_delta,
    temp_zscore columns (output of features.compute_anomaly_windows).
    """
    df = scada.copy()
    if turbine_id and "turbine_id" in df.columns:
        df = df[df["turbine_id"] == turbine_id]
        log = logbook[logbook["turbine_id"] == turbine_id] if logbook is not None and "turbine_id" in logbook.columns else logbook
    else:
        log = logbook

    if "timestamp" not in df.columns or temp_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Missing timestamp or {temp_col} column",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df = df.sort_values("timestamp")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=["Temperature (ambient-corrected, °C)", "Z-score (30-day rolling)"],
    )

    # Shade anomaly windows
    if "pre_failure_window" in df.columns:
        in_window = df["pre_failure_window"]
        starts = df["timestamp"][in_window & ~in_window.shift(1, fill_value=False)]
        ends   = df["timestamp"][in_window & ~in_window.shift(-1, fill_value=False)]
        for s, e in zip(starts, ends):
            for row in (1, 2):
                fig.add_vrect(
                    x0=s, x1=e,
                    fillcolor="rgba(231,76,60,0.12)",
                    layer="below", line_width=0,
                    row=row, col=1,
                )

    # Temperature line
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"], y=df[temp_col],
            mode="lines", name="Temp (°C)",
            line=dict(color="#3498db", width=1),
        ),
        row=1, col=1,
    )

    # Z-score line
    if zscore_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"], y=df[zscore_col],
                mode="lines", name="Z-score",
                line=dict(color="#e67e22", width=1),
            ),
            row=2, col=1,
        )
        fig.add_hline(y=2, row=2, col=1, line_dash="dot", line_color="red",
                      annotation_text="2σ")
        fig.add_hline(y=-2, row=2, col=1, line_dash="dot", line_color="red")

    # Failure event markers — add_vline breaks on datetime subplots, use shapes
    if log is not None and not log.empty and "timestamp" in log.columns:
        for _, ev in log.iterrows():
            comp = ev.get("component", "Failure")
            t_str = str(ev["timestamp"])
            for row_ref, y_ref in [("x", "y"), ("x", "y2")]:
                fig.add_shape(
                    type="line",
                    x0=t_str, x1=t_str,
                    y0=0, y1=1,
                    yref=f"{y_ref} domain",
                    xref=row_ref,
                    line=dict(color="red", width=1.2, dash="dash"),
                )
            fig.add_annotation(
                x=t_str, y=1,
                xref="x", yref="y domain",
                text=comp, showarrow=False,
                font=dict(size=10, color="red"),
                xanchor="left", yanchor="top",
            )

    fig.update_layout(
        title=title,
        height=520,
        showlegend=True,
        plot_bgcolor="#f9f9f9",
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Corrective-actions timeline (Gantt-style)
# ---------------------------------------------------------------------------

def plot_corrective_timeline(
    logbook: pd.DataFrame,
    title: str = "Corrective Actions Timeline",
) -> go.Figure:
    """Gantt-style timeline of repairs/replacements per turbine.

    Uses duration_hours to set bar widths; falls back to point markers
    if duration is absent.
    """
    if logbook is None or logbook.empty:
        fig = go.Figure()
        fig.add_annotation(text="No logbook data available",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    df = logbook.copy()
    if "timestamp" not in df.columns:
        raise ValueError("Logbook must have a 'timestamp' column.")

    has_duration = "duration_hours" in df.columns and df["duration_hours"].notna().any()
    components = df["component"].unique() if "component" in df.columns else ["Event"]
    color_map = {c: COMPONENT_COLORS[i % len(COMPONENT_COLORS)]
                 for i, c in enumerate(sorted(components))}

    if has_duration:
        df["end_time"] = df["timestamp"] + pd.to_timedelta(df["duration_hours"].fillna(0), unit="h")

        fig = px.timeline(
            df.rename(columns={"timestamp": "Start", "end_time": "Finish",
                               "turbine_id": "Turbine", "component": "Component"}),
            x_start="Start", x_end="Finish",
            y="Turbine" if "turbine_id" in logbook.columns else None,
            color="Component",
            color_discrete_map={c: color_map[c] for c in color_map},
            hover_data=["Component", "failure_mode"] if "failure_mode" in df.columns else ["Component"],
            title=title,
        )
        fig.update_yaxes(autorange="reversed")
    else:
        fig = go.Figure()
        for comp, grp in df.groupby("component" if "component" in df.columns else []):
            fig.add_trace(go.Scatter(
                x=grp["timestamp"],
                y=grp.get("turbine_id", [""] * len(grp)),
                mode="markers",
                name=comp,
                marker=dict(size=10, symbol="triangle-down", color=color_map.get(comp, "blue")),
                text=grp.get("failure_mode", ""),
                hovertemplate="<b>%{text}</b><br>%{x}<extra>" + str(comp) + "</extra>",
            ))

    fig.update_layout(
        height=420,
        plot_bgcolor="#f9f9f9",
        margin=dict(l=50, r=20, t=60, b=40),
        legend_title="Component",
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Monthly failure rate heatmap
# ---------------------------------------------------------------------------

def plot_failure_heatmap(monthly_df: pd.DataFrame) -> go.Figure:
    """Heatmap of failure counts by month × component."""
    if monthly_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig

    pivot = monthly_df.pivot_table(
        index="component", columns="period", values="count", fill_value=0
    )
    pivot.columns = [str(c) for c in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="YlOrRd",
        colorbar_title="Count",
        hovertemplate="Period: %{x}<br>Component: %{y}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Monthly Failure Rate by Component",
        height=360,
        margin=dict(l=120, r=20, t=50, b=80),
        xaxis_tickangle=-45,
    )
    return fig
