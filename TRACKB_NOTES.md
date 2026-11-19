# Track B — team working notes, changes and runbook

Everything our team changed relative to the unmodified starter kit is listed here.
Unmodified copies of every file we touched are in `_original_starter_kit/`.
Code changes are tagged `[B1]`, `[B2]` or `[B3]` in comments.

| File | Status | Challenge | Purpose |
|---|---|---|---|
| `features/featurizer.py` | modified | B1 | `--parquet_mode {rewrite,parts,off}`, `--flush_every`, `--timing_log` (per-flush throughput + flush time) |
| `scripts/b1_lag_monitor.py` | new | B1 | Samples `featurizer_group` CURRENT-OFFSET / LOG-END-OFFSET / LAG every N s to CSV |
| `scripts/b1_plot.py` | new | B1 | Lag-over-time and flush-cost plots (needs `pip install matplotlib`) |
| `scoring/score_service.py` | modified | B2 | New `GET /ready`: 503 unless model loaded **and** a canary scores |
| `docker/compose.yaml` | modified | B2 | Scorer healthcheck now hits `/ready` instead of `/health` |
| `monitoring/b3_analysis.py` | new | B3 | KS/PSI/Wasserstein, structural check, model-impact comparison |
| `results/` | new | all | Evidence files (B2 local test, B3 outputs and figure; B1 to be filled in) |

**Not changed:** Kafka/broker config, replay, model training, Evidently report, B2 broken override.

---

## 0. Baseline (do once, then for every challenge)

```bash
cp .env.example .env
python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
docker compose -f docker/compose.yaml up -d --build
python scripts/replay.py --rate 20
python features/featurizer.py --max_messages 400 --parquet_mode rewrite   # original behaviour → data/processed/features.parquet
python models/train_baseline.py
docker compose -f docker/compose.yaml up -d --build scorer
# then the README smoke test (5 checks). Save output to results/baseline_smoke.txt
```

Note: `train_baseline.py` reads `data/processed/features.parquet` by default, so the baseline uses
`--parquet_mode rewrite`. To train from the append-only output instead:
`python models/train_baseline.py --features data/processed/features_parts`
(first check `pd.read_parquet("data/processed/features_parts").timestamp.is_monotonic_increasing`).

---

## B1 — Falling Behind

### What we found in the code (diagnosis to confirm with measurements)
Per message, the featurizer does a parse, a few arithmetic features, a deque update and a Kafka produce.
Offline that costs **~10 µs per raw message** (13,060 feature rows from 15,640 raw messages in
`b1_long_ticks.jsonl`). The suspect is `_append_parquet()`: **every 25 rows it reads the entire
`features.parquet`, concatenates, and rewrites the whole file.** That makes the cost of each flush
grow with the number of rows already on disk (O(n) per flush, O(n²) per run). The file also
**keeps growing across runs** (quickstart + smoke test + every earlier B1 run), so a second run
starts slower than the first. That is a confound for any before/after comparison.

Budget at the documented 500 msg/s: 2 ms per raw message, which is about **60 ms per 25-row flush**
(≈30 raw messages arrive per 25 feature rows). Once a flush takes longer than that, the consumer falls behind.

### Hypothesis and falsifiers
- **H1:** Lag grows because persistence cost per flush rises with history size. Throughput
  falls as the run goes on, so lag growth should *accelerate* (a convex lag curve).
- **Refuted if:** (a) `flush_ms` in the timing log stays flat as `feature_rows` grows; or (b) with
  `--parquet_mode off` lag still grows at the same rate (then the bottleneck is Kafka poll/produce
  or the per-message Python path); or (c) lag grows *linearly from t=0*, which points to a fixed per-message cost.
- **Also rule out:** the producer not actually delivering 500/s. The LOG-END-OFFSET slope in the lag
  CSV gives the real arrival rate. Report it; `time.sleep(0.002)` overshoots on macOS.

### Response (one bounded change)
`--parquet_mode parts` (now the default): each flush writes **only the new rows** to
`data/processed/features_parts/part-<run>-<seq>.parquet`. Flush cost no longer depends on history.
Same rows and same order were verified offline. Kafka, partitions, consumer count and features are unchanged.
Why not add consumers or partitions? It is a single BTC-USD key with stateful features
(`previous_midprice`, the 60 s trade-intensity window). Parallelism would need per-key partitioning
and state that moves or is shared across consumers, and it cannot speed up one key's stream.

### Experiment protocol (same workload, same rate, same starting conditions)
For each run (`before` = `rewrite`, `after` = `parts`, optional `diag` = `off`):
```bash
# clean start: broker log dir is inside the container (/tmp), so recreate it; reset persisted features
docker compose -f docker/compose.yaml down && docker compose -f docker/compose.yaml up -d
mv data/processed data/processed_old_$(date +%s) 2>/dev/null   # or delete
# T1
python scripts/b1_lag_monitor.py --label before --out results/b1/lag_before.csv --interval 5
# T2
python features/featurizer.py --parquet_mode rewrite --timing_log results/b1/timing_before.csv
# T3 (start last; keep rate FIXED across runs, record it)
python scripts/replay.py --file data/challenges/b1_long_ticks.jsonl --rate 500
```
Keep sampling until lag is back to 0 (drain time is a key metric). Repeat with `--parquet_mode parts`
and label `after`. Do ≥2 repetitions each if time allows. Then:
```bash
python scripts/b1_plot.py --lag results/b1/lag_before.csv results/b1/lag_after.csv \
  --timing results/b1/timing_before.csv results/b1/timing_after.csv --out results/b1/b1.png
```
If lag stays near 0 at 500/s on your machine, raise the rate (e.g. 1000, 2000) until `before` falls
behind, then **hold that rate fixed** for every run and write the calibration down.

**Metrics for the memo table:** actual arrival rate (LOG-END-OFFSET slope), consumer throughput
(CURRENT-OFFSET slope, early vs late), peak lag, lag trend (growing/stable/shrinking), time to drain,
flush_ms at 1k vs 10k rows.

### Tradeoffs to discuss
- Many small files (~520 per run at `flush_every=25`), which slows reads and training. Fix with a raised
  `--flush_every` or a periodic compaction job.
- Readers must read a directory. Ordering relies on file names; verify timestamps are monotonic.
- If the producer rate exceeds the new ceiling, lag still grows. This change raises capacity but sets no bound.
- A larger `flush_every` cuts per-row overhead, but more rows are lost on a crash.

---

## B2 — Green but Broken

### Root cause (from code + config, confirm with live evidence)
`compose.b2-broken.yaml` sets `MODEL_PATH=/app/models/artifacts/model-b2-missing.pkl`. The file does
not exist (the volume only holds `model.pkl`). `score_service.py` catches `FileNotFoundError` and
**starts anyway**. `/health` always returns HTTP 200 (`model_loaded: false` sits in the body), and
the Docker healthcheck only checks that `urlopen(/health)` doesn't raise. So the container is
"healthy" while every `/score` returns **503 "model not loaded"**.

### Evidence to capture (live, before repair)
```bash
docker compose -f docker/compose.yaml -f docker/compose.b2-broken.yaml up -d --build scorer
docker compose -f docker/compose.yaml ps                                   # scorer: running (healthy)
docker inspect --format '{{json .State.Health}}' scorer                    # healthcheck history
curl -s -w ' [%{http_code}]\n' localhost:8000/health                       # 200, model_loaded:false
curl -s -w ' [%{http_code}]\n' -X POST localhost:8000/score -H "Content-Type: application/json" -d '{"midprice_returns":0.001,"spread_normalized":0.0002,"order_book_imbalance":0.1,"trade_intensity":12}'
docker compose -f docker/compose.yaml logs scorer | grep score_service     # WARNING: no model found at …-missing.pkl
docker compose -f docker/compose.yaml -f docker/compose.b2-broken.yaml config | grep -A3 MODEL_PATH
docker exec scorer ls -l /app/models/artifacts/                            # only model.pkl exists
docker compose -f docker/compose.yaml restart scorer                       # restart reproduces the failure
```
**Hypothesis:** the scoring failure comes from a wrong `MODEL_PATH` and not from the model, code or network.
**Falsifier:** the file exists at that path inside the container, or `/score` still fails after only
`MODEL_PATH` is corrected.

### Response
1. **Repair:** go back to the base config: `docker compose -f docker/compose.yaml up -d --force-recreate scorer`.
2. **Harden:** `/health` = liveness (process up); new `/ready` = model loaded **and** a fixed
   canary input scores to a probability in [0,1]. The Compose healthcheck now uses `/ready`.

### Validation (verified locally with a replica model, see `results/b2/b2_local_test_output.txt`)
| MODEL_PATH | /health | /ready | /score | old healthcheck | new healthcheck |
|---|---|---|---|---|---|
| model.pkl | 200 | 200 (canary 0.41) | 200 | healthy | healthy |
| model-b2-missing.pkl | 200 | **503** "model not loaded" + file error | 503 | **healthy (false green)** | **unhealthy** |

Re-run the table live in Docker: (a) base config, (b) broken override (`ps` should now show **unhealthy**
after about 10 s + 5 retries), (c) after repair.

### Questions the brief asks (draft answers)
- *What did the original check test?* Only that the Flask process answered HTTP on `/health`.
- *Why green but unusable?* The service degrades silently, and liveness was treated as readiness.
- *What should block traffic?* Not being able to produce a valid score (no model, or canary fails).
- *Would a restart fix it?* No. The config is wrong, so a restart loads the same missing path. It
  just reproduces the failure.
- *Too strict or expensive?* The canary is one LR prediction (sub-ms), so it is cheap. Risks: in plain
  Compose "unhealthy" doesn't stop traffic or restart anything by itself (it is a signal, and it gates
  `depends_on: service_healthy`). A readiness probe that also decides restarts would crash-loop on a
  config error. Keep liveness and readiness separate. A canary can pass while real inputs are out of
  distribution (that is B3). Optional: fail fast at startup when the model is missing.

---

## B3 — Healthy but Wrong?

Reproduce: `python monitoring/drift_report.py ...` (Evidently HTML) **and**
`python monitoring/b3_analysis.py` (uses `models/artifacts/model.pkl` if present; `--model none` trains a replica).

### Findings (already computed: `results/b3/`)
- Reference and current each have 653 rows with **identical timestamps**. `midprice_returns`,
  `order_book_imbalance`, `trade_intensity`, `price`, `midprice` are **identical**.
- **Only `spread_normalized` changed: every row is exactly ×4.0.** KS = 0.956 (p≈0), PSI = 8.2, and 100% of
  current rows sit above the reference p95. Implied USD spread: median $0.01 → $0.04, p95 $0.02 → $0.08.
- The reference set is exactly what the featurizer produces from `data/sample/sample_ticks.jsonl`,
  so it is also the data the model was trained on.

### Interpretation
- **Domain:** a wider spread means thinner liquidity and higher trading costs. But a *perfectly uniform*
  ×4 on every tick, with unchanged midprice and prices, does not look like an organic market move (real
  liquidity shocks are uneven and move other features too). It looks more like a **data-source or
  pipeline change** (quote source, tick size or units). So the first response is to investigate the source.
- **What drift shows:** the inputs to the model changed. **What it doesn't show:** that predictions got
  worse. Here we can actually check, because the starter kit's label (forward std of `midprice_returns`)
  depends only on returns, which did not change:
  - PR-AUC 0.118 → 0.117, ROC-AUC 0.541 → 0.537, base rate 0.103 in both. **No measurable quality change.**
  - Outputs did shift: alert rate 63.6% → 59.9%; **24 of 653 alerts suppressed** (0 added). The model
    has a negative weight on spread, and 2% of current rows lie above anything seen in training.
  - Caveat: the model is weak to begin with (ROC-AUC ≈ 0.54, in-sample, 653 rows, alert rate 6× the
    base rate). "No degradation" really means it is no worse than an already poor model.

### Recommended operational response (no retraining)
1. **Investigate and escalate** the spread data source with the data owner before acting on the model.
2. **Monitor and alert** on `spread_normalized` drift (PSI/KS against reference) and flag requests outside
   the training range (above reference max).
3. **Collect delayed labels.** Labels are available 20 ticks later, so track live PR-AUC and alert rate.
   Retrain only if labelled current data shows real degradation *and* the shift is confirmed as real.
4. **Tradeoffs:** while the investigation runs, 24 of ~415 alerts (~6%) stay suppressed. If the shift is
   real, the model mutes alerts exactly when liquidity is thin. Drift alerts on a feature this
   heavy-tailed can cause alert fatigue. Retraining on a pipeline artefact would bake the bug into the model.

Side note: the README smoke-test payload (`spread_normalized: 0.0002`) is already ~3× above the
training maximum, which is why it returns a score close to 0.

---

## AI supervision log (pick two for the memo)
1. **B2 `/ready` design.** AI proposed a canary-scoring readiness endpoint and moving the healthcheck to it.
   Verify: run both configs and record status codes plus healthcheck exit codes (see table).
   Discuss rejecting "make the service crash on missing model" as the *only* fix (no liveness/readiness split).
2. **B1 bottleneck attribution.** AI flagged `_append_parquet` from reading the code and a ~10 µs/message
   offline profile. It is **unverified until the timing log and `--parquet_mode off` run confirm it**.
   Record whether the measurements agreed.
3. **B3 "no retraining".** AI's finding that the label is drift-invariant here, which allowed a direct
   quality comparison. Verify by re-running `b3_analysis.py` against the team's own deployed `model.pkl`.
