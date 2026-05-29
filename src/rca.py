"""
Root-cause analysis helpers: Pareto analysis and failure attribution.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def failure_pareto(
    logbook: pd.DataFrame,
    group_col: str = "component",
    count_col: str | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Build Pareto table: failure count + cumulative % by component.

    Parameters
    ----------
    logbook:
        Output of load_logbook().
    group_col:
        Column to group by (default: 'component').
    count_col:
        If given, sum this column instead of counting rows.
    top_n:
        Keep only the top N components.

    Returns
    -------
    DataFrame: component, count (or sum), pct, cumulative_pct
    """
    if logbook is None or logbook.empty:
        return pd.DataFrame(columns=[group_col, "count", "pct", "cumulative_pct"])

    if group_col not in logbook.columns:
        raise ValueError(f"Column '{group_col}' not found in logbook.")

    if count_col and count_col in logbook.columns:
        summary = (
            logbook.groupby(group_col)[count_col]
            .sum()
            .reset_index()
            .rename(columns={count_col: "count"})
        )
    else:
        summary = (
            logbook.groupby(group_col)
            .size()
            .reset_index(name="count")
        )

    summary = summary.sort_values("count", ascending=False).head(top_n).reset_index(drop=True)
    total = summary["count"].sum()
    summary["pct"] = (summary["count"] / total * 100).round(1)
    summary["cumulative_pct"] = summary["pct"].cumsum().round(1)
    return summary


def downtime_pareto(logbook: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Pareto by total downtime hours per component."""
    if "duration_hours" not in logbook.columns:
        raise ValueError("Logbook missing 'duration_hours' column.")
    return failure_pareto(logbook, group_col="component", count_col="duration_hours", top_n=top_n)


def failure_timeline(logbook: pd.DataFrame) -> pd.DataFrame:
    """Prepare a tidy timeline table for the corrective-actions visual.

    Returns
    -------
    DataFrame: timestamp, turbine_id, component, failure_mode,
               work_order_type, duration_hours
    Sorted by timestamp ascending.
    """
    cols = ["timestamp", "turbine_id", "component", "failure_mode",
            "work_order_type", "duration_hours"]
    present = [c for c in cols if c in logbook.columns]
    df = logbook[present].copy()
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def monthly_failure_rate(
    logbook: pd.DataFrame,
    freq: str = "M",
) -> pd.DataFrame:
    """Failure count aggregated by month (and component).

    Returns
    -------
    DataFrame: period, component, count
    """
    if logbook is None or logbook.empty or "timestamp" not in logbook.columns:
        return pd.DataFrame()

    df = logbook.copy()
    df["period"] = df["timestamp"].dt.to_period(freq)
    grp_cols = ["period"] + (["component"] if "component" in df.columns else [])
    return df.groupby(grp_cols).size().reset_index(name="count").sort_values("period")
