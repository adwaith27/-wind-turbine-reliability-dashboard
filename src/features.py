"""
Reliability KPI computation from SCADA + logbook.

Key outputs:
  - Fleet availability (%)
  - MTBF per turbine and fleet-wide
  - Total downtime (hours)
  - Failure event count
  - Anomaly windows (temperature deviation ahead of logged failures)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def compute_availability(
    scada: pd.DataFrame,
    logbook: pd.DataFrame,
    rated_power_kw: float = 2000.0,
    min_wind_ms: float = 3.5,
) -> pd.DataFrame:
    """Compute per-turbine and fleet-wide availability.

    Availability = (scheduled production hours - downtime hours) / scheduled hours.
    Scheduled hours are estimated from timestamps where wind speed ≥ cut-in.

    Parameters
    ----------
    scada:
        Output of load_scada().
    logbook:
        Output of load_logbook().
    rated_power_kw:
        Turbine rated power; used to flag curtailed/fault periods.
    min_wind_ms:
        Cut-in wind speed threshold.

    Returns
    -------
    DataFrame: turbine_id, scheduled_hours, downtime_hours, availability_pct
    """
    required_scada = {"turbine_id", "timestamp", "active_power_kw", "wind_speed_ms"}
    required_log = {"turbine_id", "timestamp", "duration_hours"}

    has_scada = required_scada.issubset(scada.columns)
    has_log = required_log.issubset(logbook.columns) if logbook is not None else False

    results = []
    turbines = scada["turbine_id"].unique() if "turbine_id" in scada.columns else []

    for tid in turbines:
        ts = scada[scada["turbine_id"] == tid].copy()

        if not has_scada:
            results.append({"turbine_id": tid, "scheduled_hours": np.nan,
                            "downtime_hours": np.nan, "availability_pct": np.nan})
            continue

        ts = ts.sort_values("timestamp")
        # Estimate time resolution from median interval
        if len(ts) > 1:
            dt_minutes = ts["timestamp"].diff().dt.total_seconds().median() / 60
        else:
            dt_minutes = 10.0  # assume 10-min resolution

        above_cutin = ts["wind_speed_ms"] >= min_wind_ms
        scheduled_hours = above_cutin.sum() * dt_minutes / 60

        # Downtime from logbook
        downtime_hours = 0.0
        if has_log and logbook is not None and "turbine_id" in logbook.columns:
            tlog = logbook[logbook["turbine_id"] == tid]
            downtime_hours = tlog["duration_hours"].dropna().sum()

        avail = (
            (scheduled_hours - downtime_hours) / scheduled_hours * 100
            if scheduled_hours > 0 else np.nan
        )
        avail = max(0.0, min(100.0, avail))  # clamp 0–100

        results.append({
            "turbine_id": tid,
            "scheduled_hours": round(scheduled_hours, 1),
            "downtime_hours": round(downtime_hours, 1),
            "availability_pct": round(avail, 2),
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# MTBF
# ---------------------------------------------------------------------------

def compute_mtbf(logbook: pd.DataFrame) -> pd.DataFrame:
    """Mean time between failures per turbine.

    MTBF = total operating hours / number of failure events.
    Operating hours estimated from (last_event - first_event) - total_downtime.

    Returns
    -------
    DataFrame: turbine_id, failure_count, total_downtime_h, mtbf_hours
    """
    if logbook is None or logbook.empty:
        return pd.DataFrame(columns=["turbine_id", "failure_count", "total_downtime_h", "mtbf_hours"])

    required = {"turbine_id", "timestamp"}
    if not required.issubset(logbook.columns):
        raise ValueError(f"Logbook missing columns: {required - set(logbook.columns)}")

    results = []
    for tid, grp in logbook.groupby("turbine_id"):
        grp = grp.sort_values("timestamp")
        n_failures = len(grp)
        total_span_h = (
            (grp["timestamp"].max() - grp["timestamp"].min()).total_seconds() / 3600
            if n_failures > 1 else 0.0
        )
        total_downtime_h = grp["duration_hours"].dropna().sum() if "duration_hours" in grp.columns else 0.0
        operating_h = max(0.0, total_span_h - total_downtime_h)
        mtbf = operating_h / n_failures if n_failures > 0 else np.nan

        results.append({
            "turbine_id": tid,
            "failure_count": n_failures,
            "total_downtime_h": round(total_downtime_h, 1),
            "mtbf_hours": round(mtbf, 1) if not np.isnan(mtbf) else np.nan,
        })

    return pd.DataFrame(results).sort_values("turbine_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fleet-level KPI summary
# ---------------------------------------------------------------------------

def fleet_kpis(
    scada: pd.DataFrame,
    logbook: Optional[pd.DataFrame],
) -> dict:
    """Return a single-row dict of fleet-wide KPIs for the dashboard strip.

    Keys: fleet_availability_pct, fleet_mtbf_hours, total_failures,
          total_downtime_hours
    """
    avail_df = compute_availability(scada, logbook)
    mtbf_df = compute_mtbf(logbook)

    fleet_avail = avail_df["availability_pct"].mean() if not avail_df.empty else np.nan
    fleet_mtbf = mtbf_df["mtbf_hours"].mean() if not mtbf_df.empty else np.nan
    total_failures = int(mtbf_df["failure_count"].sum()) if not mtbf_df.empty else 0
    total_downtime = mtbf_df["total_downtime_h"].sum() if not mtbf_df.empty else 0.0

    return {
        "fleet_availability_pct": round(fleet_avail, 2) if not np.isnan(fleet_avail) else None,
        "fleet_mtbf_hours": round(fleet_mtbf, 1) if not np.isnan(fleet_mtbf) else None,
        "total_failures": total_failures,
        "total_downtime_hours": round(total_downtime, 1),
    }


# ---------------------------------------------------------------------------
# Anomaly windows (temperature pre-failure deviation)
# ---------------------------------------------------------------------------

def compute_anomaly_windows(
    scada: pd.DataFrame,
    logbook: pd.DataFrame,
    temp_col: str = "gearbox_oil_temp_c",
    window_days: int = 7,
    ambient_col: Optional[str] = "ambient_temp_c",
) -> pd.DataFrame:
    """Identify temperature anomaly windows in the days before each failure.

    Computes a normalised temperature metric (optionally ambient-corrected),
    then flags the *window_days* period before each logged failure event.

    Parameters
    ----------
    scada:
        SCADA dataframe with timestamp, turbine_id, temp_col.
    logbook:
        Logbook with timestamp, turbine_id.
    temp_col:
        Temperature column to analyse.
    window_days:
        Number of days before failure to mark as the anomaly window.
    ambient_col:
        If present, subtract ambient temperature to seasonally adjust.

    Returns
    -------
    scada with added columns:
        temp_delta (ambient-corrected if possible),
        temp_zscore (rolling 30-day z-score),
        pre_failure_window (bool — within window_days of a failure)
    """
    if temp_col not in scada.columns:
        raise ValueError(f"Temperature column '{temp_col}' not found in SCADA data.")

    df = scada.copy()

    # Ambient correction
    if ambient_col and ambient_col in df.columns:
        df["temp_delta"] = df[temp_col] - df[ambient_col]
    else:
        df["temp_delta"] = df[temp_col]

    # Rolling z-score (30-day window) per turbine. The caller sorts each
    # group by timestamp first, so the rolling window stays chronological.
    def _zscore(series: pd.Series) -> pd.Series:
        roll = series.rolling(window=30 * 6 * 24, min_periods=100)
        mu = roll.mean()
        sigma = roll.std().replace(0, np.nan)
        return (series - mu) / sigma

    if "turbine_id" in df.columns:
        df = df.sort_values(["turbine_id", "timestamp"])
        df["temp_zscore"] = df.groupby("turbine_id")["temp_delta"].transform(_zscore)
    else:
        df = df.sort_values("timestamp")
        df["temp_zscore"] = _zscore(df["temp_delta"])

    # Flag pre-failure windows
    df["pre_failure_window"] = False
    if logbook is not None and not logbook.empty and "timestamp" in logbook.columns:
        window = pd.Timedelta(days=window_days)
        for _, event in logbook.iterrows():
            t = event["timestamp"]
            tid = event.get("turbine_id")
            mask = df["timestamp"].between(t - window, t)
            if tid is not None and "turbine_id" in df.columns:
                mask = mask & (df["turbine_id"] == tid)
            df.loc[mask, "pre_failure_window"] = True

    return df


# ---------------------------------------------------------------------------
# Power curve status labelling
# ---------------------------------------------------------------------------

def label_power_curve_status(
    scada: pd.DataFrame,
    rated_power_kw: float = 2000.0,
    cutin_wind_ms: float = 3.5,
    cutout_wind_ms: float = 25.0,
    underperform_threshold: float = 0.7,
) -> pd.DataFrame:
    """Add an 'operational_status' column to SCADA for power-curve colouring.

    Categories: Normal, Underperforming, Fault/Stop, Curtailed, Low Wind
    """
    required = {"active_power_kw", "wind_speed_ms"}
    if not required.issubset(scada.columns):
        raise ValueError(f"Missing columns for power-curve labelling: {required - set(scada.columns)}")

    df = scada.copy()
    pwr = df["active_power_kw"]
    ws = df["wind_speed_ms"]

    conditions = [
        ws < cutin_wind_ms,
        pwr < 0,
        ws > cutout_wind_ms,
        pwr < underperform_threshold * rated_power_kw * (ws / cutout_wind_ms) ** 3,
    ]
    labels = ["Low Wind", "Fault/Stop", "Curtailed", "Underperforming"]

    df["operational_status"] = np.select(conditions, labels, default="Normal")
    return df
