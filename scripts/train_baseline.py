"""Train the TF-IDF + logistic regression classifier, evaluate on the frozen test set,
log the run to MLflow and save the serving artifact.

    python -m scripts.train_baseline
"""

import json
import subprocess
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
from sklearn.model_selection import cross_val_predict

from src import config
from src.baseline import build_pipeline
from src.data import DATASET, load_data
from src.metrics import summarize
from src.utils import benchmark_latency


def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, cwd=config.ROOT, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def choose_threshold(pipeline, train, escalation_rate):
    """Pick the confidence cut-off on train with 5-fold cross-validation.

    The threshold is the confidence below which `escalation_rate` of
    out-of-fold train predictions fall. Test data is never used here.
    """
    proba = cross_val_predict(pipeline, train["text"], train["label"], cv=5, method="predict_proba")
    return float(np.quantile(proba.max(axis=1), escalation_rate))


def main():
    train, test = load_data()
    pipeline = build_pipeline()

    threshold = choose_threshold(pipeline, train, config.TARGET_ESCALATION_RATE)
    pipeline.fit(train["text"], train["label"])

    y_pred = pipeline.predict(test["text"])
    confidence = pipeline.predict_proba(test["text"]).max(axis=1)
    summary, report = summarize(test["label"], y_pred)

    # What the human-routing rule does on test: share escalated, accuracy of the rest
    auto = confidence >= threshold
    summary["test_escalation_rate"] = float(1 - auto.mean())
    summary["auto_accepted_accuracy"] = float((y_pred[auto] == test["label"].values[auto]).mean())

    texts = test["text"].head(200).tolist()
    summary.update(benchmark_latency(lambda t: pipeline.predict_proba([t]), texts))

    model_version = f"tfidf-lr-{datetime.now(timezone.utc):%Y%m%d}-{git_sha()}"
    params = {
        "dataset": DATASET,
        "model": "tfidf+logreg",
        "max_iter": 1000,
        "target_escalation_rate": config.TARGET_ESCALATION_RATE,
        "threshold": round(threshold, 4),
    }

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold, "model_version": model_version},
                config.CLASSIFIER_PATH)

    config.REPORTS_DIR.mkdir(exist_ok=True)
    report_path = config.REPORTS_DIR / "tfidf_lr.json"
    report_path.write_text(json.dumps(
        {"model_version": model_version, "params": params, "summary": summary, "report": report},
        indent=2))

    mlflow.set_tracking_uri(f"sqlite:///{config.ROOT / 'mlflow.db'}")
    mlflow.set_experiment("categorization")
    with mlflow.start_run(run_name=model_version):
        mlflow.log_params(params)
        mlflow.log_metrics(summary)
        mlflow.log_artifact(str(report_path))

    print(json.dumps({"model_version": model_version, **params, **summary}, indent=2))


if __name__ == "__main__":
    main()
