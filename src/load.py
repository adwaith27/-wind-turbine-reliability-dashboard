"""
Loaders for EDP Open Data:
  - SCADA 10-min averaged signals (one CSV per turbine)
  - Failure logbook (single CSV covering all turbines)

Expected raw file layout under data/raw/:
  data/raw/
    scada/
      T01.csv  (or Turbine_01.csv, edp_wind_farm_scada_data_*.csv, etc.)
      T02.csv
      ...
    logbook/
      logbook.csv  (or failures.csv, maintenance_log.csv, etc.)

Column name normalisation handles EDP's published column naming conventions
(both the original Portuguese-style names and the English-translated variants
that appear in Kaggle mirrors).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Column-name maps  (raw name → normalised name)
# ---------------------------------------------------------------------------

# EDP SCADA: original column names vary by download mirror. This covers both
# the official EDP release and common Kaggle/OpenWindSCADA variants.
SCADA_COL_MAP: dict[str, str] = {
    # timestamp variants
    "timestamp": "timestamp",
    "date/time": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    # power
    "p_avg": "active_power_kw",
    "activepower": "active_power_kw",
    "active_power": "active_power_kw",
    "active power (kw)": "active_power_kw",
    "lv activepower (kw)": "active_power_kw",
    # wind speed
    "ws_avg": "wind_speed_ms",
    "windspeed": "wind_speed_ms",
    "wind_speed": "wind_speed_ms",
    "wind speed (m/s)": "wind_speed_ms",
    "theoreticalpower": "theoretical_power_kw",
    "theoretical power (kw)": "theoretical_power_kw",
    # rotor / generator speed
    "rs_avg": "rotor_speed_rpm",
    "rotorspeed": "rotor_speed_rpm",
    "rotor speed (rpm)": "rotor_speed_rpm",
    "wa_avg": "generator_speed_rpm",
    "generatorspeed": "generator_speed_rpm",
    "generator speed (rpm)": "generator_speed_rpm",
    # temperatures
    "ot_avg": "gearbox_oil_temp_c",
    "gearboxoiltemperature": "gearbox_oil_temp_c",
    "gearbox oil temperature (°c)": "gearbox_oil_temp_c",
    "ot_avg_2": "gearbox_bearing_temp_c",
    "gearboxbearingtemperature": "gearbox_bearing_temp_c",
    "gearbox bearing temperature (°c)": "gearbox_bearing_temp_c",
    "wt_avg": "generator_winding_temp_c",
    "generatorwindingtemperature": "generator_winding_temp_c",
    "generator winding temperature (°c)": "generator_winding_temp_c",
    "ambient_temp": "ambient_temp_c",
    "ambienttemperature": "ambient_temp_c",
    "ambient temperature (°c)": "ambient_temp_c",
    # pitch / wind direction
    "pitch_avg": "pitch_angle_deg",
    "pitchangle": "pitch_angle_deg",
    "wd_avg": "wind_direction_deg",
    "winddirection": "wind_direction_deg",
    "wind direction (°)": "wind_direction_deg",
}

# EDP Logbook column name variants
LOGBOOK_COL_MAP: dict[str, str] = {
    "timestamp": "timestamp",
    "date/time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "turbine_id": "turbine_id",
    "turbineid": "turbine_id",
    "machine": "turbine_id",
    "turbine": "turbine_id",
    "component": "component",
    "component_category": "component",
    "failure_mode": "failure_mode",
    "remarks": "failure_mode",
    "description": "failure_mode",
    "workordertype": "work_order_type",
    "work_order_type": "work_order_type",
    "cause": "cause",
    "duration_hours": "duration_hours",
    "duration (h)": "duration_hours",
    "downtime": "duration_hours",
    "downtime_h": "duration_hours",
}

# Component strings → canonical label used in Pareto / timeline
COMPONENT_ALIASES: dict[str, str] = {
    "gearbox": "Gearbox",
    "gear box": "Gearbox",
    "generator": "Generator",
    "transformer": "Transformer",
    "hydraulic": "Hydraulic Group",
    "hydraulic group": "Hydraulic Group",
    "hydraulics": "Hydraulic Group",
    "main bearing": "Main Bearing",
    "mainbearing": "Main Bearing",
    "rotor": "Rotor",
    "blade": "Rotor",
    "blades": "Rotor",
    "pitch": "Pitch System",
    "pitch system": "Pitch System",
    "converter": "Power Converter",
    "power converter": "Power Converter",
    "yaw": "Yaw System",
    "control": "Control System",
    "electrical": "Electrical",
    "brakes": "Brakes",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_cols(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Lower-strip column names, apply alias map, drop unmapped columns."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Robustly parse timestamps; tries ISO-8601 first, then mixed format."""
    try:
        return pd.to_datetime(series, format="ISO8601")
    except Exception:
        return pd.to_datetime(series, format="mixed", dayfirst=False)


def _canonicalise_component(series: pd.Series) -> pd.Series:
    """Map raw component strings to canonical labels."""
    def _map(val: str) -> str:
        if pd.isna(val):
            return "Unknown"
        lower = str(val).strip().lower()
        for alias, label in COMPONENT_ALIASES.items():
            if alias in lower:
                return label
        return str(val).strip().title()

    return series.map(_map)


def _find_files(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    files = []
    for ext in extensions:
        files.extend(sorted(directory.glob(f"*{ext}")))
    return files


# ---------------------------------------------------------------------------
# SCADA loader
# ---------------------------------------------------------------------------

def load_scada(
    scada_dir: str | Path,
    turbine_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load and concatenate EDP SCADA CSV files from *scada_dir*.

    Each file is assumed to cover one turbine. The turbine ID is inferred
    from the filename (e.g. 'T01.csv' → 'T01').

    Parameters
    ----------
    scada_dir:
        Path to directory containing per-turbine SCADA CSVs.
    turbine_ids:
        Optional list of turbine IDs to load. Loads all found files if None.

    Returns
    -------
    DataFrame with columns:
        turbine_id, timestamp, active_power_kw, wind_speed_ms,
        rotor_speed_rpm, gearbox_oil_temp_c, gearbox_bearing_temp_c,
        generator_winding_temp_c, ambient_temp_c  (+ any extras present)
    """
    scada_dir = Path(scada_dir)
    if not scada_dir.exists():
        raise FileNotFoundError(f"SCADA directory not found: {scada_dir}")

    files = _find_files(scada_dir, (".csv", ".xlsx", ".xls"))
    if not files:
        raise FileNotFoundError(f"No CSV/Excel files found in {scada_dir}")

    frames: list[pd.DataFrame] = []
    for fpath in files:
        # infer turbine ID from filename
        stem = fpath.stem
        turbine_id = re.sub(r"[^A-Za-z0-9_\-]", "", stem) or stem

        if turbine_ids and turbine_id not in turbine_ids:
            continue

        if fpath.suffix == ".csv":
            df = pd.read_csv(fpath, low_memory=False)
        else:
            df = pd.read_excel(fpath)

        df = _normalise_cols(df, SCADA_COL_MAP)

        if "timestamp" not in df.columns:
            # fallback: first column that looks like a date
            for col in df.columns:
                sample = df[col].dropna().head(5).astype(str)
                if sample.str.match(r"\d{4}[-/]\d{2}").any():
                    df = df.rename(columns={col: "timestamp"})
                    break

        if "timestamp" in df.columns:
            df["timestamp"] = _parse_timestamp(df["timestamp"])

        df.insert(0, "turbine_id", turbine_id)
        frames.append(df)

    if not frames:
        raise ValueError("No matching SCADA files loaded. Check turbine_ids filter.")

    scada = pd.concat(frames, ignore_index=True)

    if "timestamp" in scada.columns:
        scada = scada.sort_values(["turbine_id", "timestamp"]).reset_index(drop=True)

    print(
        f"[load_scada] {len(frames)} turbine file(s) loaded | "
        f"{len(scada):,} rows | "
        f"columns: {list(scada.columns)}"
    )
    return scada


# ---------------------------------------------------------------------------
# Logbook loader
# ---------------------------------------------------------------------------

def load_logbook(logbook_path: str | Path) -> pd.DataFrame:
    """Load the EDP failure / maintenance logbook.

    Parameters
    ----------
    logbook_path:
        Path to the logbook CSV or Excel file.

    Returns
    -------
    DataFrame with columns:
        timestamp, turbine_id, component, failure_mode,
        work_order_type, cause, duration_hours
    """
    logbook_path = Path(logbook_path)
    if not logbook_path.exists():
        raise FileNotFoundError(f"Logbook file not found: {logbook_path}")

    if logbook_path.suffix == ".csv":
        df = pd.read_csv(logbook_path, low_memory=False)
    else:
        df = pd.read_excel(logbook_path)

    df = _normalise_cols(df, LOGBOOK_COL_MAP)

    if "timestamp" in df.columns:
        df["timestamp"] = _parse_timestamp(df["timestamp"])

    if "component" in df.columns:
        df["component"] = _canonicalise_component(df["component"])

    # ensure duration column is numeric
    if "duration_hours" in df.columns:
        df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    print(
        f"[load_logbook] {len(df):,} log entries | "
        f"components: {df['component'].unique().tolist() if 'component' in df.columns else 'N/A'}"
    )
    return df


# ---------------------------------------------------------------------------
# CARE to Compare loader
# Real structure (semicolon-separated, Wind Farm X/datasets/*.csv)
# ---------------------------------------------------------------------------

# Wind Farm A sensor_N → our normalised column names
# Derived from feature_description.csv in the dataset
CARE_SENSOR_MAP: dict[str, str] = {
    "sensor_0_avg":  "ambient_temp_c",
    "wind_speed_3_avg": "wind_speed_ms",
    "wind_speed_3_max": "wind_speed_ms_max",
    "wind_speed_3_min": "wind_speed_ms_min",
    "wind_speed_4_avg": "wind_speed_est_ms",
    "sensor_5_avg":  "pitch_angle_deg",
    "sensor_11_avg": "gearbox_bearing_temp_c",   # gearbox bearing high-speed shaft
    "sensor_12_avg": "gearbox_oil_temp_c",        # oil in gearbox
    "sensor_13_avg": "generator_bearing_de_temp_c",
    "sensor_14_avg": "generator_bearing_nde_temp_c",
    "sensor_15_avg": "generator_winding1_temp_c",
    "sensor_16_avg": "generator_winding2_temp_c",
    "sensor_17_avg": "generator_winding3_temp_c",
    "sensor_18_avg": "generator_speed_rpm",
    "power_29_avg":  "possible_active_power_kw",
    "power_30_avg":  "active_power_kw",           # grid power — main power signal
    "sensor_38_avg": "transformer_temp_l1_c",
    "sensor_39_avg": "transformer_temp_l2_c",
    "sensor_40_avg": "transformer_temp_l3_c",
    # meta columns
    "time_stamp":       "timestamp",
    "asset_id":         "turbine_id",
    "train_test":       "split",
    "status_type_id":   "status_id",
}


def load_care_dataset(
    dataset_path: str | Path,
    rated_power_kw: float = 2000.0,
) -> pd.DataFrame:
    """Load one CARE dataset CSV (semicolon-separated) into our SCADA schema.

    Power columns in Wind Farm A are normalized 0-1 (capacity factor).
    They are scaled by rated_power_kw to produce kW values.

    Parameters
    ----------
    dataset_path:
        Path to a file under Wind Farm X/datasets/*.csv
    rated_power_kw:
        Turbine rated power for de-normalizing power signals (default 2000 kW).
    """
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path, sep=";", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop row index column
    df = df.drop(columns=[c for c in df.columns if c in {"id", "unnamed: 0"}], errors="ignore")

    df = df.rename(columns={k: v for k, v in CARE_SENSOR_MAP.items() if k in df.columns})

    if "timestamp" in df.columns:
        df["timestamp"] = _parse_timestamp(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Stringify turbine_id for consistency with EDP schema
    if "turbine_id" in df.columns:
        df["turbine_id"] = "T" + df["turbine_id"].astype(str).str.zfill(2)

    # Scale normalized power columns (0-1) → kW
    for col in ["active_power_kw", "possible_active_power_kw"]:
        if col in df.columns and df[col].abs().max() <= 1.5:
            df[col] = (df[col] * rated_power_kw).clip(lower=0).round(1)

    n_fault = df["status_id"].gt(0).sum() if "status_id" in df.columns else "N/A"
    print(f"[load_care] {dataset_path.name} | {len(df):,} rows | fault rows: {n_fault}")
    return df


def load_care_logbook(wind_farm_dir: str | Path) -> pd.DataFrame:
    """Load the event_info.csv logbook from a CARE Wind Farm directory.

    event_info.csv columns (semicolon-separated):
        asset, event_id, event_label, event_start, event_end, event_description

    Returns logbook in our standard schema:
        timestamp, turbine_id, component, failure_mode, duration_hours
    """
    wind_farm_dir = Path(wind_farm_dir)
    logbook_path = wind_farm_dir / "event_info.csv"
    if not logbook_path.exists():
        # try comma variant
        logbook_path = wind_farm_dir / "comma_event_info.csv"
    if not logbook_path.exists():
        raise FileNotFoundError(f"event_info.csv not found in {wind_farm_dir}")

    sep = ";" if "comma_" not in logbook_path.name else ","
    df = pd.read_csv(logbook_path, sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]

    # Keep only anomaly events (skip normal-behaviour datasets)
    if "event_label" in df.columns:
        df = df[df["event_label"] == "anomaly"].copy()

    df["timestamp"] = _parse_timestamp(df["event_start"])
    df["turbine_id"] = "T" + df["asset"].astype(str).str.zfill(2)

    # event_description holds the component name (e.g. "Gearbox failure")
    df["component"] = _canonicalise_component(df.get("event_description", pd.Series(["Unknown"] * len(df))))
    df["failure_mode"] = df.get("event_description", "Unknown")

    if "event_end" in df.columns:
        end = _parse_timestamp(df["event_end"])
        df["duration_hours"] = (end - df["timestamp"]).dt.total_seconds() / 3600
    else:
        df["duration_hours"] = float("nan")

    result = df[["timestamp", "turbine_id", "component", "failure_mode", "duration_hours"]].copy()
    print(f"[load_care_logbook] {len(result)} anomaly events | components: {result['component'].unique().tolist()}")
    return result.sort_values("timestamp").reset_index(drop=True)


def load_all_care(
    care_dir: str | Path,
    wind_farm: str = "Wind Farm A",
    max_datasets: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CARE to Compare data from the standard Kaggle download layout.

    Directory structure expected:
        care_dir/
          Wind Farm A/
            event_info.csv      ← logbook
            datasets/
              17.csv, 22.csv …  ← one CSV per turbine-window

    Parameters
    ----------
    care_dir:
        Root of the downloaded dataset (data/raw/care/).
    wind_farm:
        Subfolder name — "Wind Farm A" (EDP, 86 features),
        "Wind Farm B" (257 features), or "Wind Farm C" (957 features).
    max_datasets:
        Cap the number of dataset CSVs loaded (useful for quick exploration).

    Returns
    -------
    (scada_df, logbook_df) — both in our standard schema.
    """
    care_dir = Path(care_dir)
    wf_dir = care_dir / wind_farm
    datasets_dir = wf_dir / "datasets"

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    # Exclude comma_ variants (duplicate with different separator)
    files = sorted(f for f in datasets_dir.glob("*.csv") if not f.name.startswith("comma_"))
    if max_datasets:
        files = files[:max_datasets]
    if not files:
        raise FileNotFoundError(f"No dataset CSVs found in {datasets_dir}")

    frames = [load_care_dataset(f) for f in files]
    scada = pd.concat(frames, ignore_index=True)

    logbook = load_care_logbook(wf_dir)

    print(
        f"\n[load_all_care] {wind_farm} | "
        f"{len(scada):,} SCADA rows across {len(frames)} datasets | "
        f"{len(logbook)} logbook events"
    )
    return scada, logbook


# ---------------------------------------------------------------------------
# Convenience: load everything from the standard data/raw/ layout
# ---------------------------------------------------------------------------

def load_all(
    raw_dir: str | Path = "data/raw",
) -> dict[str, pd.DataFrame]:
    """Load SCADA and logbook from the standard project layout.

    Checks for data in this priority order:
      1. data/raw/scada/  + data/raw/logbook/  (EDP direct download)
      2. data/raw/care/                         (CARE to Compare Kaggle download)

    Returns
    -------
    dict with keys 'scada' and 'logbook' (values may be None if not found).
    """
    raw_dir = Path(raw_dir)
    result: dict[str, pd.DataFrame | None] = {}

    # --- Priority 1: EDP direct layout ---
    scada_dir = raw_dir / "scada"
    if scada_dir.exists() and any(scada_dir.glob("*.csv")):
        result["scada"] = load_scada(scada_dir)
        logbook_candidates = (
            list((raw_dir / "logbook").glob("*.csv")) if (raw_dir / "logbook").exists() else []
        )
        logbook_candidates += (
            list(raw_dir.glob("*logbook*")) +
            list(raw_dir.glob("*failure*")) +
            list(raw_dir.glob("*maintenance*"))
        )
        result["logbook"] = load_logbook(logbook_candidates[0]) if logbook_candidates else None
        if result["logbook"] is None:
            print("[load_all] No logbook found — KPIs will use SCADA-only estimates.")
        return result

    # --- Priority 2: CARE to Compare layout ---
    care_dir = raw_dir / "care"
    if care_dir.exists() and any((care_dir / "Wind Farm A").glob("datasets/*.csv")):
        print("[load_all] Loading CARE to Compare dataset (Wind Farm A)…")
        scada, logbook = load_all_care(care_dir, wind_farm="Wind Farm A")
        result["scada"] = scada
        result["logbook"] = logbook
        return result

    print(
        f"[load_all] No data found in {raw_dir}.\n"
        "  Run:  uv run python scripts/download_data.py\n"
        "  Then: uv run jupyter notebook notebooks/01_data_loading.ipynb"
    )
    result["scada"] = None
    result["logbook"] = None
    return result
