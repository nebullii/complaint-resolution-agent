import time

import numpy as np


def benchmark_latency(predict_fn, samples, n_warmup=10):
    """Benchmarks prediction latency, returning median and p95 in milliseconds."""
    # Warmup run to avoid cold-start overhead skewing metrics
    for text in samples[:n_warmup]:
        predict_fn(text)

    times = []
    for text in samples:
        start = time.perf_counter()
        predict_fn(text)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "median_ms": round(float(np.median(times)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
    }
