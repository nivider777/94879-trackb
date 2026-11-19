"""
monitoring/b3_analysis.py   [Track B / B3 addition]

Complements monitoring/drift_report.py (Evidently) with:
  1. Per-feature drift statistics (KS, PSI, Wasserstein/std) between the
     explicit reference and current datasets.
  2. A structural check: is current a row-aligned transform of reference?
  3. Model impact: score both datasets with the deployed scoring artifact and
     compare outputs and quality against the starter kit's own label
     definition (forward 20-tick std of midprice_returns >= 90th pct).

Usage:
    python monitoring/b3_analysis.py                       # uses models/artifacts/model.pkl if present
    python monitoring/b3_analysis.py --model none          # trains a replica on reference (train_baseline logic)
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

F = ["midprice_returns", "spread_normalized", "order_book_imbalance", "trade_intensity"]


def psi(a, b, n=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, n + 1)))
    if len(q) < 3:
        return float("nan")
    q[0], q[-1] = -np.inf, np.inf
    pa = np.clip(np.histogram(a, q)[0] / len(a), 1e-4, None)
    pb = np.clip(np.histogram(b, q)[0] / len(b), 1e-4, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def label_spikes(df, horizon=20, percentile=0.90):
    fv = df["midprice_returns"].rolling(horizon).std().shift(-horizon)
    return (fv >= fv.quantile(percentile)).astype(int), fv.notna()


def main(ref_path, cur_path, model_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ref, cur = pd.read_csv(ref_path), pd.read_csv(cur_path)
    print(f"reference={ref_path} rows={len(ref)}  current={cur_path} rows={len(cur)}")

    # 1. drift statistics
    rows = []
    for f in F:
        a, b = ref[f].to_numpy(), cur[f].to_numpy()
        ks = stats.ks_2samp(a, b)
        rows.append({"feature": f, "ks_stat": ks.statistic, "ks_p": ks.pvalue, "psi": psi(a, b),
                     "wasserstein_over_ref_std": stats.wasserstein_distance(a, b) / a.std() if a.std() else np.nan,
                     "ref_median": np.median(a), "cur_median": np.median(b),
                     "ref_p95": np.quantile(a, .95), "cur_p95": np.quantile(b, .95),
                     "cur_share_above_ref_p95": float((b > np.quantile(a, .95)).mean()),
                     "cur_share_above_ref_max": float((b > a.max()).mean())})
    drift = pd.DataFrame(rows)
    drift.to_csv(os.path.join(out_dir, "b3_drift_stats.csv"), index=False)
    print("\n== drift statistics ==\n" + drift.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # 2. structure
    structure = {}
    if len(ref) == len(cur) and "timestamp" in ref and (ref["timestamp"] == cur["timestamp"]).all():
        structure["same_timestamps"] = True
        for f in F + [c for c in ("price", "midprice") if c in ref and c in cur]:
            same = bool(np.allclose(ref[f], cur[f]))
            ratio = (cur[f] / ref[f]).replace([np.inf, -np.inf], np.nan).dropna()
            structure[f] = "identical" if same else f"ratio min={ratio.min():.6g} max={ratio.max():.6g}"
    print("\n== structural check ==\n" + json.dumps(structure, indent=2))
    if "midprice" in ref:
        for name, d in (("reference", ref), ("current", cur)):
            usd = d["spread_normalized"] * d["midprice"]
            print(f"{name}: implied USD spread median={usd.median():.4f} p95={usd.quantile(.95):.4f}")

    # 3. model impact
    if model_path and model_path != "none" and os.path.exists(model_path):
        bundle = pickle.load(open(model_path, "rb"))
        model, scaler, src = bundle["model"], bundle["scaler"], model_path
    else:
        y, _ = label_spikes(ref)
        tr = ref.iloc[: int(len(ref) * 0.6)]
        scaler = StandardScaler().fit(tr[F])
        model = LogisticRegression(max_iter=500, class_weight="balanced").fit(scaler.transform(tr[F]), y.iloc[: len(tr)])
        src = "replica trained on reference (train_baseline.py logic)"
    print(f"\n== model impact ({src}) ==")
    print("standardized coefficients:", dict(zip(F, np.round(model.coef_[0], 3))))
    y_ref, valid_ref = label_spikes(ref)
    y_cur, valid_cur = label_spikes(cur)
    out = {}
    for name, d, y, v in (("reference", ref, y_ref, valid_ref), ("current", cur, y_cur, valid_cur)):
        p = model.predict_proba(scaler.transform(d[F]))[:, 1]
        out[name] = p
        print(f"{name}: mean score={p.mean():.3f}  alert rate (p>=0.5)={(p >= .5).mean():.3f}  "
              f"PR-AUC={average_precision_score(y[v], p[v]):.3f}  ROC-AUC={roc_auc_score(y[v], p[v]):.3f}  "
              f"label base rate={y[v].mean():.3f}")
    print("labels identical between datasets:", bool((y_ref == y_cur).all()),
          "(label depends only on midprice_returns)")
    if len(ref) == len(cur):
        pr, pc = out["reference"] >= .5, out["current"] >= .5
        print(f"decision flips: {(pr != pc).sum()} of {len(pr)}  (alert->no alert: {(pr & ~pc).sum()}, "
              f"no alert->alert: {(~pr & pc).sum()})")
        pd.DataFrame({"timestamp": ref.get("timestamp"), "score_ref": out["reference"], "score_cur": out["current"]}) \
            .to_csv(os.path.join(out_dir, "b3_scores.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/challenges/b3_reference_features.csv")
    ap.add_argument("--current", default="data/challenges/b3_current_features.csv")
    ap.add_argument("--model", default="models/artifacts/model.pkl")
    ap.add_argument("--out_dir", default="results/b3")
    a = ap.parse_args()
    main(a.reference, a.current, a.model, a.out_dir)
