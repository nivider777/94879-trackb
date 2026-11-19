"""
scripts/b1_lag_monitor.py   [Track B / B1 addition]

Samples the featurizer_group consumer group every --interval seconds using the
exact command documented in CHALLENGE_SUPPORT.md, and appends CURRENT-OFFSET,
LOG-END-OFFSET and LAG to a CSV. Run it in a third terminal before starting
the replay; stop it (Ctrl+C) once lag has returned to 0 or the run is over.

Usage:
    python scripts/b1_lag_monitor.py --out results/b1/lag_before.csv --label before
"""
import argparse
import csv
import os
import subprocess
import time

CMD = ["docker", "compose", "-f", "docker/compose.yaml", "exec", "-T", "broker",
       "/opt/kafka/bin/kafka-consumer-groups.sh", "--bootstrap-server", "localhost:9092",
       "--describe", "--group"]


def sample(group):
    out = subprocess.run(CMD + [group], capture_output=True, text=True, timeout=30).stdout
    rows, header = [], None
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "GROUP":
            header = parts
            continue
        if header and len(parts) >= 6 and parts[0] == group:
            rec = dict(zip(header, parts))
            rows.append(rec)
    return rows


def to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None  # Kafka prints "-" when no committed offset exists yet


def main(group, interval, out, label):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    new = not os.path.exists(out)
    t0 = time.time()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["label", "wall_time", "elapsed_s", "topic", "partition",
                        "current_offset", "log_end_offset", "lag"])
        print(f"[lag_monitor] sampling {group} every {interval}s -> {out} (Ctrl+C to stop)")
        try:
            while True:
                now = time.time()
                for r in sample(group):
                    cur, end, lag = (to_int(r.get("CURRENT-OFFSET")), to_int(r.get("LOG-END-OFFSET")),
                                     to_int(r.get("LAG")))
                    w.writerow([label, f"{now:.1f}", f"{now - t0:.1f}", r.get("TOPIC"), r.get("PARTITION"),
                                cur, end, lag])
                    print(f"[lag_monitor] t={now - t0:6.1f}s  current={cur}  log_end={end}  lag={lag}")
                f.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("[lag_monitor] stopped")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--group", default="featurizer_group")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--out", default="results/b1/lag.csv")
    p.add_argument("--label", default="run")
    a = p.parse_args()
    main(a.group, a.interval, a.out, a.label)
