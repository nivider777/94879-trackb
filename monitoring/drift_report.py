"""monitoring/drift_report.py

Generate an Evidently report comparing an explicit reference dataset with an
explicit current dataset. CSV and Parquet inputs are supported.

The Track B B3 challenge ships controlled reference/current CSV files so the
monitoring question is not confounded by an arbitrary split inside one file.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

FEATURE_COLUMNS = [
    "midprice_returns",
    "spread_normalized",
    "order_book_imbalance",
    "trade_intensity",
]


def load_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    raise SystemExit(f"Unsupported dataset format for {path!r}; use CSV or Parquet")


def prepare(df: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{label} dataset is missing feature columns: {missing}")
    clean = df.dropna(subset=FEATURE_COLUMNS)
    if len(clean) < 20:
        raise SystemExit(f"Need at least 20 usable {label} rows, found {len(clean)}")
    return clean


def main(reference_path: str, current_path: str, out_path: str) -> None:
    reference = prepare(load_table(reference_path), "reference")
    current = prepare(load_table(current_path), "current")

    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    result = report.run(
        reference_data=reference[FEATURE_COLUMNS],
        current_data=current[FEATURE_COLUMNS],
    )

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.save_html(out_path)
    print(f"[drift_report] reference rows={len(reference)}, current rows={len(current)}")
    print(f"[drift_report] reference={reference_path}")
    print(f"[drift_report] current={current_path}")
    print(f"[drift_report] saved report to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare reference and current feature distributions")
    parser.add_argument(
        "--reference",
        default="data/challenges/b3_reference_features.csv",
        help="Reference feature dataset (.csv or .parquet)",
    )
    parser.add_argument(
        "--current",
        default="data/challenges/b3_current_features.csv",
        help="Current feature dataset (.csv or .parquet)",
    )
    parser.add_argument("--out", default="reports/drift_report.html")
    args = parser.parse_args()
    main(args.reference, args.current, args.out)
