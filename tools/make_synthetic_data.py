"""Create SYNTHETIC CICIDS2017-shaped CSV files for smoke-testing the pipeline.

The data is random noise with a few class-dependent shifts. It is NOT real network traffic and results on it
mean nothing scientifically. It only reproduces the *format and quirks* of the real files (stray spaces in
column names, inf/NaN values, negative durations, duplicate rows, a duplicate column, a corrupted dash in the
web-attack labels, rare classes) so `xnids.py` can be tested without downloading ~900 MB.

Usage:  python tools/make_synthetic_data.py [output_dir]      (default: data_synthetic)
        XNIDS_DATA_DIR=data_synthetic XNIDS_FAST=1 python xnids.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COLUMNS = json.loads((HERE / "cicids_columns.json").read_text())
rng = np.random.default_rng(0)

# file name -> {label: rows}
FILES = {
    "Monday-WorkingHours.pcap_ISCX.csv": {"BENIGN": 6000},
    "Tuesday-WorkingHours.pcap_ISCX.csv": {"BENIGN": 6000, "FTP-Patator": 500, "SSH-Patator": 400},
    "Wednesday-workingHours.pcap_ISCX.csv": {"BENIGN": 6000, "DoS Hulk": 900, "DoS GoldenEye": 300, "DoS slowloris": 250, "DoS Slowhttptest": 250, "Heartbleed": 11},
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": {"BENIGN": 4000, "Web Attack \u2013 Brute Force": 300, "Web Attack \u2013 XSS": 150, "Web Attack \u2013 Sql Injection": 21},
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": {"BENIGN": 4000, "Infiltration": 36},
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": {"BENIGN": 4000, "Bot": 200},
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": {"BENIGN": 3000, "PortScan": 1500},
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": {"BENIGN": 3000, "DDoS": 1500},
}
PORTS = {"FTP-Patator": 21, "SSH-Patator": 22, "PortScan": None, "DDoS": 80, "DoS Hulk": 80, "DoS GoldenEye": 80, "DoS slowloris": 80, "DoS Slowhttptest": 80, "Bot": 8080}


def label_profile(label):
    r = np.random.default_rng(sum(ord(ch) * (i + 1) for i, ch in enumerate(label.replace("\u2013", "-"))))
    idx = r.choice(len(COLUMNS), size=6, replace=False)
    return idx, r.uniform(1.5, 4.0, size=6)


def make_block(label, n):
    data = rng.lognormal(mean=2.0, sigma=1.0, size=(n, len(COLUMNS)))
    clean = label.replace("\u2013", "-")
    if clean != "BENIGN":
        idx, shift = label_profile(clean)
        data[:, idx] *= shift
    port_col = COLUMNS.index("Destination Port")
    if clean == "BENIGN":
        data[:, port_col] = rng.choice([53, 80, 443, 443, 8080, 22, 3389, 445, 123], size=n)
    elif clean.startswith("Web Attack") or clean in PORTS and PORTS[clean]:
        data[:, port_col] = PORTS.get(clean) or 80
    else:
        data[:, port_col] = rng.integers(1, 65535, size=n)
    df = pd.DataFrame(data, columns=COLUMNS)
    df["Fwd Header Length.1"] = df["Fwd Header Length"]
    df["Label"] = label
    return df


for fname, spec in FILES.items():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data_synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([make_block(lab, n) for lab, n in spec.items()], ignore_index=True)
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    # real-data quirks
    dup = df.sample(frac=0.08, random_state=2)
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[df.sample(5, random_state=3).index, "Flow Bytes/s"] = np.inf
    df.loc[df.sample(5, random_state=4).index, "Flow Packets/s"] = np.nan
    df.loc[df.sample(3, random_state=5).index, "Flow Duration"] = -1
    df.columns = [" " + c for c in df.columns]  # real files have a leading space in every column name
    df.to_csv(out_dir / fname, index=False, encoding="cp1252")
    print(f"wrote {fname}: {len(df):,} rows")
