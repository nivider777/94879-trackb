"""
scoring/score_service.py

A minimal HTTP scoring service. Loads the pickled model produced by
models/train_baseline.py and exposes two endpoints:

  GET  /health          -> liveness only: the process is up and answering HTTP.
                           {"status": "ok", "model_loaded": bool}  (always 200)
  GET  /ready           -> readiness: 200 only if the model is loaded AND a
                           fixed canary input scores to a finite probability;
                           otherwise 503 with the reason.  [Track B / B2 change]
  POST /score           -> {"score": float, "spike": bool}
                            body: {"midprice_returns": .., "spread_normalized": ..,
                                    "order_book_imbalance": .., "trade_intensity": ..}

Deliberately framework-light (Flask) and single-file so the whole thing
can be read in one sitting.
"""
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("MODEL_PATH", "models/artifacts/model.pkl")
FEATURE_COLUMNS = ["midprice_returns", "spread_normalized", "order_book_imbalance", "trade_intensity"]

app = Flask(__name__)
_state = {"model": None, "scaler": None, "load_error": None}

# [B2] Fixed, in-range canary input used by /ready to prove the full scoring
# path (scaler + model) works, not just that the process is alive.
CANARY = {"midprice_returns": 0.0, "spread_normalized": 1e-7,
          "order_book_imbalance": 0.0, "trade_intensity": 100}


def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    _state["model"] = bundle["model"]
    _state["scaler"] = bundle["scaler"]
    print(f"[score_service] loaded model from {MODEL_PATH}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": _state["model"] is not None})


def _predict(payload):
    x = pd.DataFrame([{c: payload[c] for c in FEATURE_COLUMNS}])
    x_scaled = _state["scaler"].transform(x)
    return float(_state["model"].predict_proba(x_scaled)[0, 1])


@app.route("/ready", methods=["GET"])
def ready():
    """[B2] Readiness = able to produce a real score. Used by the Compose healthcheck."""
    if _state["model"] is None:
        return jsonify({"ready": False, "reason": "model not loaded",
                        "model_path": MODEL_PATH, "load_error": _state["load_error"]}), 503
    try:
        prob = _predict(CANARY)
    except Exception as e:  # scaler/model mismatch, bad artifact, etc.
        return jsonify({"ready": False, "reason": f"canary scoring failed: {type(e).__name__}: {e}"}), 503
    if not (0.0 <= prob <= 1.0):
        return jsonify({"ready": False, "reason": f"canary produced invalid probability {prob}"}), 503
    return jsonify({"ready": True, "model_path": MODEL_PATH, "canary_score": prob})


@app.route("/score", methods=["POST"])
def score():
    if _state["model"] is None:
        return jsonify({"error": "model not loaded"}), 503

    payload = request.get_json(force=True)
    missing = [c for c in FEATURE_COLUMNS if c not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    prob = _predict(payload)
    return jsonify({"score": prob, "spike": prob >= 0.5})


if __name__ == "__main__":
    try:
        load_model()
    except FileNotFoundError as e:
        _state["load_error"] = str(e)
        print(f"[score_service] WARNING: no model found at {MODEL_PATH}. "
              f"Run models/train_baseline.py first. Starting anyway: /health stays 200 (liveness) but /ready returns 503.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
