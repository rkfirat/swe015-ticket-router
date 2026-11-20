"""Evaluate every model on the same held-out CLINC150 test split (R3, R4).

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

from src.common import RESULTS, load_clinc
from src.models import load_all

LATENCY_RUNS = 200


def route(probs: np.ndarray, oos: int, tau: float) -> np.ndarray:
    """Low confidence or explicit out-of-scope -> human agent (oos)."""
    pred = probs.argmax(-1)
    pred[probs.max(-1) < tau] = oos
    return pred


def pick_threshold(probs, y, oos) -> float:
    """Validation has only 100 OOS messages out of 3,100, so plain accuracy would ignore them;
    maximize the mean of in-scope accuracy and OOS recall instead."""
    grid = np.round(np.arange(0.0, 0.96, 0.05), 2)

    def balanced(t):
        m = metrics(route(probs, oos, t), y, oos)
        return (m["in-scope acc"] + m["OOS recall"]) / 2

    return float(max(grid, key=balanced))


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
    ds, names, oos = load_clinc()
    val_x, val_y = ds["validation"]["text"], np.array(ds["validation"]["intent"])
    test_x, test_y = ds["test"]["text"], np.array(ds["test"]["intent"])

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
        f"Test split: CLINC150 'plus' test, {len(test_y)} messages "
        f"({(test_y == oos).sum()} out-of-scope). Latency: mean of {LATENCY_RUNS} single-message "
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
