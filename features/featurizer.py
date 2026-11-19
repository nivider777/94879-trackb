"""
features/featurizer.py

Kafka consumer/producer that turns raw Coinbase ticker messages (ticks.raw)
into a small set of predictive features (ticks.features).

Kept deliberately small: parse -> compute_features() -> produce. The pure
functions at the top have no Kafka dependency, so scripts/replay.py can
import them directly and guarantee replay == live processing.
"""
import argparse
import json
import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd

WINDOW_SIZE_SECONDS = 60
FEATURE_COLUMNS = [
    "midprice_returns",
    "spread_normalized",
    "order_book_imbalance",
    "trade_intensity",
]


def parse_ticker_message(raw_value):
    """Extract the fields we need from one Coinbase ticker JSON message."""
    try:
        data = json.loads(raw_value)
        for event in data.get("events", []):
            if event.get("type") in ("update", "snapshot") and "tickers" in event:
                t = event["tickers"][0]
                return {
                    "timestamp": data.get("timestamp"),
                    "price": float(t.get("price", 0)),
                    "best_bid": float(t.get("best_bid", 0)),
                    "best_ask": float(t.get("best_ask", 0)),
                    "best_bid_quantity": float(t.get("best_bid_quantity", 0)),
                    "best_ask_quantity": float(t.get("best_ask_quantity", 0)),
                }
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None
    return None


def compute_features(tick, previous_midprice=None):
    """Turn one parsed tick into a feature row (no windowing yet)."""
    if tick is None:
        return None
    midprice = (tick["best_bid"] + tick["best_ask"]) / 2
    if previous_midprice and previous_midprice > 0:
        midprice_returns = (midprice - previous_midprice) / previous_midprice
    else:
        midprice_returns = 0.0

    spread = tick["best_ask"] - tick["best_bid"]
    spread_normalized = spread / midprice if midprice > 0 else 0.0

    total_qty = tick["best_bid_quantity"] + tick["best_ask_quantity"]
    imbalance = (
        (tick["best_bid_quantity"] - tick["best_ask_quantity"]) / total_qty
        if total_qty > 0
        else 0.0
    )

    return {
        "timestamp": tick["timestamp"],
        "price": tick["price"],
        "midprice": midprice,
        "midprice_returns": midprice_returns,
        "spread_normalized": spread_normalized,
        "order_book_imbalance": imbalance,
    }


def parse_timestamp(ts_str):
    if ts_str.endswith("Z"):
        ts_str = ts_str[:-1] + "+00:00"
    ts = datetime.fromisoformat(ts_str)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


class TradeIntensityWindow:
    """Rolling count of ticks in the last WINDOW_SIZE_SECONDS. Shared by
    the live consumer and replay so both compute trade_intensity the same way."""

    def __init__(self, window_seconds=WINDOW_SIZE_SECONDS):
        self._timestamps = deque()
        self._delta = timedelta(seconds=window_seconds)

    def update(self, current_ts):
        window_start = current_ts - self._delta
        while self._timestamps and self._timestamps[0] < window_start:
            self._timestamps.popleft()
        self._timestamps.append(current_ts)
        return len(self._timestamps)


def run_featurizer(broker, topic_in, topic_out, max_messages=None,
                    parquet_path="data/processed/features.parquet", flush_every=25,
                    parquet_mode="parts", parts_dir="data/processed/features_parts",
                    timing_log=None):
    """Live mode: consume ticks.raw, produce ticks.features, and periodically
    persist the same rows to Parquet (used by training + drift report).

    [B1] parquet_mode:
      rewrite -> ORIGINAL starter-kit behaviour: every flush reads the whole
                 features.parquet, concatenates, and rewrites it (cost grows
                 with file size).
      parts   -> append-only: every flush writes only the new rows to a new
                 part file in parts_dir (cost independent of history).
      off     -> DIAGNOSTIC ONLY: no persistence, to isolate Kafka/compute cost.
    timing_log: optional CSV path; one row per flush with throughput and flush
                time, so the bottleneck can be localized with evidence."""
    from quixstreams import Application

    app = Application(
        broker_address=broker,
        consumer_group="featurizer_group",
        auto_offset_reset="earliest",
    )
    producer = app.get_producer()
    window = TradeIntensityWindow()
    previous_midprice = None
    processed = 0
    buffer = []
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    if parquet_mode == "parts":
        os.makedirs(parts_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    part_seq = 0
    raw_seen = 0
    t_start = time.perf_counter()
    t_last, raw_last, rows_last = t_start, 0, 0
    tlog = None
    if timing_log:
        os.makedirs(os.path.dirname(timing_log) or ".", exist_ok=True)
        tlog = open(timing_log, "w")
        tlog.write("wall_time,elapsed_s,raw_msgs,feature_rows,interval_raw_per_s,flush_ms,file_rows_after_flush,mode\n")
    print(f"[featurizer] parquet_mode={parquet_mode} flush_every={flush_every}")

    with app.get_consumer() as consumer:
        consumer.subscribe([topic_in])
        print(f"[featurizer] listening on {topic_in} -> {topic_out}")
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error() is not None:
                print(f"[featurizer] consumer error: {msg.error()}")
                continue
            raw_seen += 1

            tick = parse_ticker_message(msg.value())
            if tick is None or not tick.get("timestamp"):
                # This message was intentionally inspected and skipped. Store its
                # offset so Kafka can still report meaningful consumer-group lag.
                consumer.store_offsets(msg)
                continue

            features = compute_features(tick, previous_midprice)
            if features is None:
                continue
            features["trade_intensity"] = window.update(parse_timestamp(tick["timestamp"]))
            previous_midprice = features["midprice"]

            producer.produce(topic=topic_out, key="BTC-USD", value=json.dumps(features))
            buffer.append(features)
            processed += 1

            if len(buffer) >= flush_every:
                t0 = time.perf_counter()
                file_rows = None
                if parquet_mode == "rewrite":
                    file_rows = _append_parquet(buffer, parquet_path)
                elif parquet_mode == "parts":
                    _write_part(buffer, parts_dir, run_id, part_seq)
                    part_seq += 1
                flush_ms = (time.perf_counter() - t0) * 1000
                now = time.perf_counter()
                rate = (raw_seen - raw_last) / (now - t_last) if now > t_last else 0.0
                print(f"[featurizer] produced {processed} feature rows (flushed {len(buffer)}, "
                      f"{flush_ms:.1f} ms, {rate:.0f} raw msg/s since last flush)")
                if tlog:
                    tlog.write(f"{time.time():.3f},{now - t_start:.3f},{raw_seen},{processed},{rate:.1f},"
                               f"{flush_ms:.2f},{file_rows if file_rows is not None else ''},{parquet_mode}\n")
                    tlog.flush()
                t_last, raw_last = now, raw_seen
                buffer = []

            # get_consumer() does not automatically store consumed offsets.
            # Store only after this message has been handled successfully; the
            # consumer's normal auto-commit interval then publishes group progress.
            consumer.store_offsets(msg)

            if max_messages and processed >= max_messages:
                if buffer and parquet_mode == "rewrite":
                    _append_parquet(buffer, parquet_path)
                elif buffer and parquet_mode == "parts":
                    _write_part(buffer, parts_dir, run_id, part_seq)
                print(f"[featurizer] reached max_messages={max_messages}, exiting")
                break


def _append_parquet(rows, path):
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_parquet(path, engine="pyarrow", index=False)
    return len(new_df)


def _write_part(rows, parts_dir, run_id, seq):
    """[B1] Append-only persistence: write only the new rows. Zero-padded names
    keep lexical order == write order. Read back with pd.read_parquet(parts_dir)."""
    path = os.path.join(parts_dir, f"part-{run_id}-{seq:07d}.parquet")
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka consumer that computes features")
    parser.add_argument("--broker", default=os.environ.get("KAFKA_BROKER", "localhost:9092"))
    parser.add_argument("--topic_in", default="ticks.raw")
    parser.add_argument("--topic_out", default="ticks.features")
    parser.add_argument("--max_messages", type=int, default=None,
                         help="Optional: exit after N messages (handy for smoke tests / grading)")
    parser.add_argument("--parquet_mode", choices=["rewrite", "parts", "off"], default="parts",
                         help="[B1] rewrite = original starter-kit behaviour; parts = append-only fix; off = diagnostic")
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--parquet_path", default="data/processed/features.parquet")
    parser.add_argument("--parts_dir", default="data/processed/features_parts")
    parser.add_argument("--timing_log", default=None, help="[B1] CSV of per-flush throughput/flush time")
    args = parser.parse_args()
    run_featurizer(args.broker, args.topic_in, args.topic_out, args.max_messages,
                   parquet_path=args.parquet_path, flush_every=args.flush_every,
                   parquet_mode=args.parquet_mode, parts_dir=args.parts_dir,
                   timing_log=args.timing_log)
