"""Evaluate every model on the same held-out test split (Bitext 20% + CLINC150 out-of-scope) (R3, R4).

Outputs results/results.csv, results/results.md, results/thresholds.json, results/errors.csv.
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from src.common import RESULTS, load_data
from src.models import load_all

LATENCY_RUNS = 200


def route(probs: np.ndarray, oos: int, tau: float) -> np.ndarray:
    """Predicted out-of-scope or max probability below tau -> human agent (oos)."""
    pred = probs.argmax(-1)
    pred[probs.max(-1) < tau] = oos
    return pred


def pick_threshold(probs, y, oos) -> float:
    """Overall accuracy on validation, where 13% of the messages are out-of-scope (as in test)."""
    grid = np.round(np.r_[np.arange(0.0, 0.96, 0.05), 0.97, 0.98, 0.99], 2)
    acc = {t: (route(probs, oos, t) == y).mean() for t in grid}
    best = max(acc.values())
    # Validation is in Bitext's style, where almost every threshold looks the same. Among the
    # thresholds within 0.2 points of the best, take the lowest: real messages are phrased more
    # loosely and get lower confidence, and the out-of-scope class already catches unrelated ones.
    return float(min(t for t in grid if acc[t] >= best - 0.002))


def metrics(pred, y, oos):
    ins = y != oos
    return {
        "accuracy": (pred == y).mean(),
        "in-scope acc": (pred[ins] == y[ins]).mean(),
        "OOS recall": (pred[~ins] == oos).mean(),
    }


def latency_ms(model, texts) -> float:
    for t in texts[:10]:
        model.predict_proba([t])
    times = []
    for i in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        model.predict_proba([texts[i % len(texts)]])
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times)


def hardware() -> str:
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        chip = platform.processor()
    return f"{chip} CPU, {torch.get_num_threads()} torch threads, batch size 1"


def main():
    splits, names, oos, _ = load_data()
    val_x, val_y = splits["validation"][0], np.array(splits["validation"][1])
    test_x, test_y = splits["test"][0], np.array(splits["test"][1])

    rows, thresholds, preds = [], {}, {}
    for model in load_all(with_ablations=True):
        key = f"{model.name} [{model.precision}]"
        print(f"evaluating {key}", flush=True)
        tau = pick_threshold(model.predict_proba(val_x), val_y, oos)
        pred = route(model.predict_proba(test_x), oos, tau)
        preds[key] = pred
        thresholds[key] = tau
        rows.append({
            "model": model.name,
            "precision": model.precision,
            **{k: round(float(v) * 100, 2) for k, v in metrics(pred, test_y, oos).items()},
            "size (MB)": round(model.size_mb(), 1),
            "latency (ms)": round(latency_ms(model, test_x), 2),
            "threshold": tau,
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "results.csv", index=False)
    hw = hardware()
    (RESULTS / "results.md").write_text(
        f"Test split: Bitext 20% + CLINC150 out-of-scope, {len(test_y)} messages "
        f"({(test_y == oos).sum()} out-of-scope). "
        f"Latency: mean of {LATENCY_RUNS} single-message "
        f"calls incl. preprocessing, {hw}.\n\n" + df.to_markdown(index=False) + "\n"
    )
    (RESULTS / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    final = list(preds)[-1]
    wrong = np.flatnonzero(preds[final] != test_y)
    pd.DataFrame({
        "text": [test_x[i] for i in wrong],
        "true": [names[test_y[i]] for i in wrong],
        **{k: [names[preds[k][i]] for i in wrong] for k in preds},
    }).to_csv(RESULTS / "errors.csv", index=False)
    print(df.to_string(index=False))
    print(hw)


if __name__ == "__main__":
    main()
