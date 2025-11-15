# src/eval/cost.py
# ----------------
"""
Đo chi phí & hiệu năng (Cost-Performance) cho S-NAP.

- measure_latency: đo thời gian chạy predict cho 1 batch prefix nhiều lần.
- estimate_cpr: tính "Cost-Performance Ratio" đơn giản:
    CPR = main_metric / avg_latency_sec   (metric per second)

Tuỳ nhu cầu, bạn có thể thay công thức CPR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence


PredictFn = Callable[[Sequence[str]], Any]


@dataclass
class LatencyStats:
    avg_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    raw_ms: List[float]


def measure_latency(
    predict_one: PredictFn,
    prefixes: List[Sequence[str]],
    warmup: int = 3,
    repeat: int = 10,
) -> LatencyStats:
    """
    Đo latency (ms) cho predict_one(prefix).

    Parameters
    ----------
    predict_one:
        Hàm nhận 1 prefix (list[str]) và trả result (bất kỳ).
        Ví dụ: lambda p: hybrid.predict_one(p)
    prefixes:
        Danh sách prefix dùng để benchmark (sẽ lấy lần lượt, vòng tròn).
    warmup:
        Số lần chạy "warmup" (không tính vào kết quả).
    repeat:
        Số lần đo (tổng số call được đo).

    Returns
    -------
    LatencyStats:
        avg/p50/p90/p95 và danh sách thô (ms).
    """
    if not prefixes:
        raise ValueError("prefixes rỗng, không đo latency được.")
    if repeat <= 0:
        raise ValueError("repeat phải > 0.")

    # warmup
    idx = 0
    for _ in range(warmup):
        pref = prefixes[idx % len(prefixes)]
        predict_one(pref)
        idx += 1

    times_ms: List[float] = []
    for _ in range(repeat):
        pref = prefixes[idx % len(prefixes)]
        t0 = time.perf_counter()
        predict_one(pref)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        idx += 1

    times_ms_sorted = sorted(times_ms)
    n = len(times_ms_sorted)

    def _percentile(arr: List[float], q: float) -> float:
        if not arr:
            return 0.0
        pos = (len(arr) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(arr) - 1)
        if lo == hi:
            return arr[lo]
        return arr[lo] + (arr[hi] - arr[lo]) * (pos - lo)

    avg = sum(times_ms_sorted) / n if n > 0 else 0.0
    p50 = _percentile(times_ms_sorted, 0.5)
    p90 = _percentile(times_ms_sorted, 0.9)
    p95 = _percentile(times_ms_sorted, 0.95)

    return LatencyStats(
        avg_ms=avg,
        p50_ms=p50,
        p90_ms=p90,
        p95_ms=p95,
        raw_ms=times_ms,
    )


def estimate_cpr(
    main_metric: float,
    latency_stats: LatencyStats,
) -> float:
    """
    CPR (Cost-Performance Ratio) đơn giản:

        CPR = main_metric / (avg_latency_ms / 1000)

    = "điểm F1 (hoặc metric chính) trên mỗi giây inference"
    """
    if latency_stats.avg_ms <= 0:
        return 0.0
    avg_sec = latency_stats.avg_ms / 1000.0
    return main_metric / avg_sec
