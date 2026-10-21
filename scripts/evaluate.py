"""Score a predictions CSV against the frozen test set and write reports/<name>.json.

The CSV needs `true` and `pred` columns, one row per test complaint in test order.
Optional columns: `latency_ms` (or `time`), `input_tokens`, `output_tokens`.

    python -m scripts.evaluate data/llm_test_results.csv --name zero_shot_llm --deterministic no
    python -m scripts.evaluate --compare
"""

import argparse
import json

import numpy as np
import pandas as pd

from src import config
from src.data import load_data
from src.labels import HIGH_RISK
from src.metrics import summarize


def evaluate_csv(path, name, deterministic):
    df = pd.read_csv(path)
    _, test = load_data()
    if len(df) != len(test) or not (df["true"].values == test["label"].values).all():
        raise SystemExit(f"{path}: rows do not line up with the test set")

    summary, report = summarize(df["true"], df["pred"])
    summary["invalid"] = int((df["pred"] == "invalid").sum())

    latency_col = "latency_ms" if "latency_ms" in df else "time" if "time" in df else None
    if latency_col:
        summary["median_ms"] = float(np.median(df[latency_col].dropna()))
        summary["p95_ms"] = float(np.percentile(df[latency_col].dropna(), 95))

    if {"input_tokens", "output_tokens"} <= set(df.columns):
        cost = (df["input_tokens"].sum() * config.LLM_INPUT_PRICE_PER_M
                + df["output_tokens"].sum() * config.LLM_OUTPUT_PRICE_PER_M) / 1e6
        summary["cost_per_1k_usd"] = float(cost / len(df) * 1000)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    out = config.REPORTS_DIR / f"{name}.json"
    out.write_text(json.dumps({"model_version": name, "deterministic": deterministic,
                               "summary": summary, "report": report}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


def compare():
    """Print the comparison table across every report in reports/."""
    cols = ["macro_f1"] + [f"recall_{c}" for c in HIGH_RISK] + ["p95_ms", "cost_per_1k_usd"]
    rows = []
    for path in sorted(config.REPORTS_DIR.glob("*.json")):
        r = json.loads(path.read_text())
        if "summary" not in r:  # e.g. the retrieval eval report
            continue
        s = r["summary"]
        rows.append([path.stem] + [s.get(c) for c in cols] + [r.get("deterministic", "yes")])

    header = ["approach"] + cols + ["deterministic"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in sorted(rows, key=lambda r: -(r[1] or 0)):
        cells = [row[0]] + [f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))
                            for v in row[1:]]
        print("| " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--name")
    parser.add_argument("--deterministic", default="yes")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare()
    elif args.csv and args.name:
        evaluate_csv(args.csv, args.name, args.deterministic)
    else:
        parser.error("give a CSV and --name, or --compare")


if __name__ == "__main__":
    main()
