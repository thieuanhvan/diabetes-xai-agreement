"""
Build the NHANES hypertension table used for the transfer demonstration of the
audit protocol (diagnosis versus measured-blood-pressure label).

Input: data/cdc_nhanes_diabetes_<cycle>.csv (predictors, HighBP = BPQ020, WTMEC)
and the raw oscillometric blood-pressure files (P_BPXO / BPXO_L), whose
readings are averaged per person. Measured hypertension: mean systolic >= 140
or mean diastolic >= 90 mmHg. Blood pressure is never used as a predictor.

Usage:
    python -m src.data.build_nhanes_hypertension --raw-dir <nhanes raw cdc dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"
BPXO = {"2017-2020": "2017-2020/P_BPXO.xpt", "2021-2023": "2021-2023/BPXO_L.xpt"}


def build(raw_dir: Path) -> None:
    for cycle, rel in BPXO.items():
        d = pd.read_csv(DATA / f"cdc_nhanes_diabetes_{cycle}.csv")
        b = pd.read_sas(raw_dir / rel)
        b["SBP_mean"] = b[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
        b["DBP_mean"] = b[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
        b["SEQN"] = b["SEQN"].astype(int)
        out = d.merge(b[["SEQN", "SBP_mean", "DBP_mean"]], on="SEQN", how="left")
        out.to_csv(DATA / f"cdc_nhanes_hypertension_{cycle}.csv", index=False)
        print(cycle, len(out), "rows; measured BP available:", int(out.SBP_mean.notna().sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    build(ap.parse_args().raw_dir)
