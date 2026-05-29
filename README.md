# Wind Turbine Reliability Dashboard

Streamlit dashboard for wind turbine gearbox and rotating equipment reliability analysis using SCADA-style field data, failure logs, root cause analysis, corrective actions, MTBF, availability, and downtime Pareto methods.

This project was built to mirror the type of analysis used by reliability, field performance, and rotating equipment teams. It connects turbine operating signals with maintenance records to identify component-level failure patterns, monitor gearbox temperature behavior, quantify downtime impact, and support corrective-action prioritization.

## Role Fit

This project demonstrates:

| Requirement | Evidence in this project |
| --- | --- |
| Gearbox, drivetrain, rotating equipment, or wind turbine field data | SCADA analysis for turbine power behavior, gearbox oil and bearing temperatures, generator winding temperature, and turbine-level operating history |
| Root cause analysis | Component-level Pareto charts for failure count and downtime, failure-mode grouping, and monthly failure-rate heatmaps |
| Corrective actions | Timeline of corrective maintenance events by turbine, component, failure mode, and downtime duration |
| Reliability engineering methods | Fleet availability, MTBF, failure event counts, downtime attribution, and pre-failure anomaly windows |
| Data and dashboarding | Python data pipeline, Streamlit app, Plotly visualizations, parquet-ready processed data structure, and synthetic demo fallback |

## What The App Shows

The dashboard contains five reliability views:

1. Fleet KPI strip
   - Availability
   - MTBF
   - Failure events
   - Total downtime

2. Power curve health
   - Wind speed vs active power
   - Operational status labeling: normal, underperforming, fault/stop, curtailed, and low wind
   - Theoretical power curve reference

3. Failure Pareto analysis
   - Failure count by component
   - Downtime by component
   - Cumulative percentage line to identify the highest-impact components

4. Gearbox temperature trend and anomaly windows
   - Ambient-corrected temperature signal
   - Rolling 30-day z-score baseline
   - Highlighted pre-failure windows before logged gearbox events

5. Corrective actions and monthly failure patterns
   - Gantt-style corrective maintenance timeline
   - Monthly component failure-rate heatmap
   - Raw filtered SCADA and logbook tables for inspection

## Data

The app is designed for EDP Open Data wind turbine SCADA and failure logbook files:

- SCADA signals: wind speed, active power, gearbox oil temperature, gearbox bearing temperature, generator winding temperature, ambient temperature, and turbine ID
- Logbook records: timestamp, turbine ID, component, failure mode, work order type, and downtime duration

The dashboard also runs without external files. If no processed parquet files are found in `data/processed/`, it falls back to synthetic demo data for four turbines so the full interface can be reviewed immediately.

## Methodology

Availability is estimated from scheduled production time, using wind-speed periods above cut-in as the operating opportunity and subtracting downtime recorded in the failure logbook.

MTBF is calculated per turbine from the elapsed time between logged failures after subtracting downtime, then averaged for the fleet-level KPI.

Temperature anomaly detection subtracts ambient temperature from the selected gearbox or generator temperature signal, then computes a rolling z-score. The app highlights the seven-day window before matching gearbox failure events.

Power curve status uses rule-based thresholds against wind speed, active power, rated power, cut-in speed, and cut-out speed. This keeps the model interpretable for reliability review instead of hiding behavior behind a black-box classifier.

Pareto analysis groups logged failures by component and supports both count-based and downtime-based views, helping identify the components that drive the largest reliability impact.

## Repository Structure

```text
wind-turbine-reliability-dashboard/
├── app/
│   └── app.py                  # Streamlit dashboard
├── notebooks/
│   ├── 01_data_loading.ipynb   # Data ingestion and processing
│   ├── 02_eda.ipynb            # Exploratory analysis
│   └── 03_rca_reliability.ipynb
├── scripts/
│   └── download_data.py
├── src/
│   ├── features.py             # KPIs, availability, MTBF, anomaly windows
│   ├── load.py                 # SCADA/logbook loading and column normalization
│   ├── rca.py                  # Pareto, timeline, and monthly failure helpers
│   └── viz.py                  # Plotly chart builders
├── .streamlit/
│   └── config.toml             # Dashboard theme
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# Install dependencies with uv
uv sync

# Run the dashboard
uv run streamlit run app/app.py
```

Or with a standard virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

## Using Real Data

Place processed parquet files here:

```text
data/processed/scada.parquet
data/processed/logbook.parquet
```

Expected minimum columns:

```text
SCADA:
turbine_id, timestamp, wind_speed_ms, active_power_kw,
gearbox_oil_temp_c, gearbox_bearing_temp_c,
generator_winding_temp_c, ambient_temp_c

Logbook:
timestamp, turbine_id, component, failure_mode,
work_order_type, duration_hours
```

If these files are missing, the app automatically uses synthetic data.

## Local Verification

Final checks run locally:

```bash
python -m compileall app src scripts
```

```python
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app/app.py", default_timeout=30)
at.run()
print(len(at.exception), len(at.error))
```

Latest verification result:

```text
exceptions 0
errors 0
plotly_charts 5
dataframes 2
```

## Tech Stack

- Python
- Pandas and NumPy
- Streamlit
- Plotly
- PyArrow/parquet-ready data flow
- Jupyter notebooks for exploration and RCA workflow

