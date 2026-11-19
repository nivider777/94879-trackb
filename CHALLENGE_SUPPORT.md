# Track B challenge support files

These files support the student-facing Track B challenge guide. They provide the
controlled conditions for B1, B2, and B3; they do not prescribe a diagnosis or
solution.

## B1 — Falling Behind

Use the supplied extended workload instead of `--loop`. It repeats the recorded
market sequence while shifting every cycle forward in event time, so timestamps
remain strictly increasing.

Start from a clean stack if your team has accumulated earlier Kafka offsets/data.
Run the featurizer in one terminal and the workload in another:

```bash
python features/featurizer.py
python scripts/replay.py --file data/challenges/b1_long_ticks.jsonl --rate 500
```

Observe the `featurizer_group` consumer group repeatedly while the workload runs
and drains:

```bash
docker compose -f docker/compose.yaml exec broker /opt/kafka/bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 --describe --group featurizer_group
```

Record `CURRENT-OFFSET`, `LOG-END-OFFSET`, and `LAG` at several points. The
extended workload contains raw records that the featurizer intentionally filters,
so raw-message count and feature-row count are not expected to be equal.

## B2 — Green but Broken

First train the model and confirm ordinary scoring works. Then apply the supplied
broken Compose override:

```bash
docker compose -f docker/compose.yaml -f docker/compose.b2-broken.yaml up -d --build scorer
```

Use the challenge guide's evidence-first process: inspect deployment state,
health behavior, a real scoring request, logs, and effective configuration before
changing anything.

To return to the normal base configuration after the exercise:

```bash
docker compose -f docker/compose.yaml up -d --force-recreate scorer
```

## B3 — Healthy but Wrong?

The starter kit includes explicit reference and current feature datasets. Run:

```bash
python monitoring/drift_report.py --reference data/challenges/b3_reference_features.csv --current data/challenges/b3_current_features.csv
```

The report is written to `reports/drift_report.html` by default. Interpret what
changed, what that evidence establishes, and what it does *not* establish before
recommending an operational response.

### B1 offset visibility

The featurizer stores successfully handled Kafka offsets so that
`kafka-consumer-groups.sh --describe --group featurizer_group` can report
CURRENT-OFFSET and LAG during the run. If CURRENT-OFFSET shows `-`, stop and
verify that you are using this v2.1 featurizer before interpreting lag.
