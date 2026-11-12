"""Per-stage request latency recorder, exposed on GET /metrics.

Each request records one sample per stage it went through (embed, ann, bm25,
fusion, total, ...). Samples live in a bounded ring buffer per stage, so the
percentiles describe recent traffic and memory stays constant under load.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

import numpy as np


class LatencyRecorder:
    def __init__(self, window: int = 20_000):
        self._window = window
        self._samples: dict[str, deque] = {}
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, stage: str, ms: float) -> None:
        with self._lock:
            buf = self._samples.get(stage)
            if buf is None:
                buf = self._samples[stage] = deque(maxlen=self._window)
            buf.append(ms)
            self._counts[stage] = self._counts.get(stage, 0) + 1

    def record_many(self, timings: dict[str, float]) -> None:
        for stage, ms in timings.items():
            self.record(stage, ms)

    def incr(self, counter: str, n: int = 1) -> None:
        with self._lock:
            self._counts[counter] = self._counts.get(counter, 0) + n

    def snapshot(self) -> dict:
        with self._lock:
            samples = {k: np.fromiter(v, dtype=float) for k, v in self._samples.items()}
            counts = dict(self._counts)
        stages = {}
        for k, a in samples.items():
            if len(a) == 0:
                continue
            p50, p95, p99 = np.percentile(a, [50, 95, 99])
            stages[k] = {"n": counts.get(k, len(a)), "window": len(a),
                         "p50_ms": round(float(p50), 3), "p95_ms": round(float(p95), 3),
                         "p99_ms": round(float(p99), 3), "mean_ms": round(float(a.mean()), 3)}
        return {"stages": stages,
                "counters": {k: v for k, v in counts.items() if k not in samples}}

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()
            self._counts.clear()


class StageTimer:
    """Collects named stage timings for one request.

        t = StageTimer()
        with t.stage("embed"): ...
        with t.stage("ann"): ...
        t.timings  -> {"embed": 12.1, "ann": 3.4}
    """

    def __init__(self):
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[name] = self.timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000

    def add(self, name: str, ms: float) -> None:
        self.timings[name] = self.timings.get(name, 0.0) + ms

    def server_timing_header(self) -> str:
        """RFC-style Server-Timing value, visible in browser devtools."""
        return ", ".join(f"{k};dur={v:.2f}" for k, v in self.timings.items())
