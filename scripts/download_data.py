"""
Download wind turbine datasets via Kaggle API.

Requires ~/.kaggle/kaggle.json  (see instructions below).

Usage:
    uv run python scripts/download_data.py

What gets downloaded
--------------------
1. CARE to Compare (primary — EDP wind farm A SCADA + fault events)
   Kaggle dataset: azizkasimov/wind-turbine-scada-data-for-early-fault-detection
   → data/raw/care/

2. Simple wind turbine SCADA (single-turbine baseline, no logbook)
   Kaggle dataset: berkerisen/wind-turbine-scada-dataset
   → data/raw/scada_single/

How to get kaggle.json
----------------------
1. Go to https://www.kaggle.com/settings  (log in first)
2. Scroll to "API" section → click "Create New Token"
3. This downloads kaggle.json — move it to ~/.kaggle/kaggle.json
4. chmod 600 ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"


def check_credentials() -> bool:
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not cred.exists():
        print(
            "\n[ERROR] No Kaggle credentials found.\n\n"
            "Steps to get them:\n"
            "  1. Go to https://www.kaggle.com/settings\n"
            "  2. API section → 'Create New Token' → downloads kaggle.json\n"
            "  3. mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json\n"
            "  4. chmod 600 ~/.kaggle/kaggle.json\n"
            "  5. Re-run this script.\n"
        )
        return False
    return True


def kaggle_download(dataset: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[download] {dataset} → {dest.relative_to(ROOT)}")
    subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", dataset,
            "-p", str(dest),
            "--unzip",
        ],
        check=True,
    )
    print(f"[done] {dataset}")


def main() -> None:
    if not check_credentials():
        sys.exit(1)

    # 1. CARE to Compare — EDP SCADA + fault events (primary dataset)
    kaggle_download(
        dataset="azizkasimov/wind-turbine-scada-data-for-early-fault-detection",
        dest=DATA_RAW / "care",
    )

    # 2. Single-turbine SCADA (simple baseline, no logbook — useful for power-curve demo)
    kaggle_download(
        dataset="berkerisen/wind-turbine-scada-dataset",
        dest=DATA_RAW / "scada_single",
    )

    print(
        "\n[next] Run notebooks/01_data_loading.ipynb to inspect and process the data.\n"
        "\nNote: For the full EDP logbook (gearbox/generator/transformer repair records),\n"
        "request access at: https://edp.com/en/innovation/data\n"
        "Place the logbook CSV in data/raw/logbook/ when you receive it."
    )


if __name__ == "__main__":
    main()
