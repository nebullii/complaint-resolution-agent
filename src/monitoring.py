"""Drift watch: compare live predictions in the audit log against the training baseline.

Population Stability Index (PSI) on two distributions:
  - predicted label shares (did the complaint mix change?)
  - confidence (is the model less sure than it used to be?)
Rule of thumb: PSI < 0.1 stable, 0.1 to 0.25 watch, > 0.25 investigate.
"""

import numpy as np

CONFIDENCE_BINS = np.linspace(0, 1, 11)


def psi(expected, actual, eps=1e-4):
    """PSI between two arrays of shares that each sum to 1."""
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def label_shares(labels, classes):
    labels = list(labels)
    return np.array([labels.count(c) / max(len(labels), 1) for c in classes])


def confidence_shares(confidences):
    counts, _ = np.histogram(confidences, bins=CONFIDENCE_BINS)
    return counts / max(counts.sum(), 1)


def status(value):
    return "stable" if value < 0.1 else "watch" if value < 0.25 else "investigate"
