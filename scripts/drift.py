"""Compare classifications in the audit log with the reference distribution.

The reference is the deployed model's predictions on the frozen test set
(train predictions would be over-confident). Live data is every `classify`
event in the audit log.

    python -m scripts.drift
"""

import joblib

from src import config
from src.audit import AuditLog
from src.data import load_data
from src.monitoring import confidence_shares, label_shares, psi, status


def main(min_events=50):
    events = AuditLog().events(step="classify")
    if len(events) < min_events:
        print(f"only {len(events)} classify events in the audit log; need {min_events} for a stable estimate")
        return

    artifact = joblib.load(config.CLASSIFIER_PATH)
    pipeline = artifact["pipeline"]
    _, test = load_data()
    ref_proba = pipeline.predict_proba(test["text"])
    classes = list(pipeline.classes_)
    ref_labels = [classes[i] for i in ref_proba.argmax(axis=1)]

    live_labels = [e["payload"]["label"] for e in events]
    live_conf = [e["payload"]["confidence"] for e in events]
    versions = sorted({e["model_version"] for e in events})

    label_psi = psi(label_shares(ref_labels, classes), label_shares(live_labels, classes))
    conf_psi = psi(confidence_shares(ref_proba.max(axis=1)), confidence_shares(live_conf))
    print(f"events: {len(events)}, model versions seen: {versions}")
    print(f"label mix PSI:  {label_psi:.3f} ({status(label_psi)})")
    print(f"confidence PSI: {conf_psi:.3f} ({status(conf_psi)})")


if __name__ == "__main__":
    main()
