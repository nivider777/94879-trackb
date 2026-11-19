"""
scripts/b1_plot.py   [Track B / B1 addition]

Plots lag-over-time (from b1_lag_monitor.py) and flush-time-vs-file-size
(from featurizer --timing_log) for the before/after runs.
Needs matplotlib (pip install matplotlib) -- plotting only, not part of the pipeline.

Usage:
    python scripts/b1_plot.py --lag results/b1/lag_before.csv results/b1/lag_after.csv \
        --timing results/b1/timing_before.csv results/b1/timing_after.csv --out results/b1/b1.png
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--lag", nargs="+", required=True)
p.add_argument("--timing", nargs="*", default=[])
p.add_argument("--out", default="results/b1/b1.png")
a = p.parse_args()

fig, axes = plt.subplots(1, 2 if a.timing else 1, figsize=(12 if a.timing else 6, 4), squeeze=False)
ax = axes[0][0]
for path in a.lag:
    d = pd.read_csv(path).dropna(subset=["lag"])
    for label, g in d.groupby("label"):
        ax.plot(g["elapsed_s"], g["lag"], marker="o", ms=3, label=label)
ax.set(xlabel="seconds since monitor start", ylabel="featurizer_group LAG (messages)",
       title="Consumer lag over time (same workload & rate)")
ax.legend()

if a.timing:
    ax = axes[0][1]
    for path in a.timing:
        d = pd.read_csv(path)
        ax.plot(d["feature_rows"], d["flush_ms"], ".", ms=3, label=d["mode"].iloc[0])
    ax.set(xlabel="feature rows produced so far", ylabel="flush time (ms)",
           title="Per-flush persistence cost vs. history size")
    ax.legend()

fig.tight_layout()
fig.savefig(a.out, dpi=150)
print(f"saved {a.out}")
