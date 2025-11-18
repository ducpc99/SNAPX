# src/eval/runner.py
# -------------------
"""
Runner đánh giá S-NAPX Hybrid.

Nhiệm vụ
--------
- Nhận danh sách EvalSample (prefix, y_true, case_id, optional gold_explanation).
- Gọi model (HybridSNAP) để lấy:
    + y_pred (nhãn cuối cùng)
    + ranked_labels (danh sách label được xếp hạng)
    + meta (chứa explanation, cost, v.v. nếu có)
- Tính các metric:
    + Accuracy
    + Macro-F1
    + MRR
    + NDCG@k
    + Explain Metrics (Reason-Pass, length, token-overlap với gold_explanation, ...)
    + Latency (nếu measure_cost=True)
    + CPR = Macro-F1 / avg_latency_sec (nếu có latency)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import time
from collections import Counter, defaultdict

from .explain_metrics import (
    ExplanationRecord,
    summarize_explain_metrics,
)


# ------------------------------------------------
# Dataclasses
# ------------------------------------------------
@dataclass
class EvalSample:
    """
    Một sample dùng cho evaluation.

    Attributes
    ----------
    case_id:
        ID duy nhất cho sample (ví dụ: model_id_revision_prefixLen).
    prefix:
        Chuỗi hoạt động đã xảy ra (list[str]).
    y_true:
        Nhãn đúng (activity tiếp theo).
    gold_explanation:
        (Tuỳ chọn) giải thích "chuẩn" từ dataset IT (cột output).
        Nếu chạy trên S-NAP csv thường → None.
    """

    case_id: str
    prefix: List[str]
    y_true: str
    gold_explanation: Optional[str] = None


@dataclass
class LatencyStats:
    avg_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    mrr: float
    ndcg_at_k: float
    explain_metrics: Dict[str, float]
    latency: Optional[LatencyStats]
    cpr: Optional[float]


# ------------------------------------------------
# Helpers: classification metrics
# ------------------------------------------------
def _compute_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def _compute_macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """
    Tự implement Macro-F1 để không phụ thuộc sklearn.
    """
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0

    # đếm TP, FP, FN cho từng label
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    f1s: List[float] = []
    for lab in labels:
        tp_l = tp[lab]
        fp_l = fp[lab]
        fn_l = fn[lab]

        if tp_l == 0 and fp_l == 0 and fn_l == 0:
            # label không xuất hiện, bỏ qua
            continue

        prec = tp_l / float(tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0
        rec = tp_l / float(tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
        if prec + rec == 0.0:
            f1 = 0.0
        else:
            f1 = 2.0 * prec * rec / (prec + rec)
        f1s.append(f1)

    if not f1s:
        return 0.0
    return sum(f1s) / float(len(f1s))


def _compute_mrr_and_ndcg(
    y_true: Sequence[str],
    ranked_lists: Sequence[Sequence[str]],
    ndcg_k: int,
) -> Tuple[float, float]:
    """
    Tính MRR và NDCG@k, với giả định mỗi sample chỉ có 1 nhãn đúng.

    - MRR: trung bình reciprocal rank của nhãn đúng trong ranked_list.
    - NDCG@k: do chỉ 1 nhãn đúng, IDCG = 1, nên NDCG = DCG = 1/log2(rank+1)
      nếu nhãn đúng nằm trong top-k, ngược lại 0.
    """
    if not y_true or not ranked_lists:
        return 0.0, 0.0

    rr_list: List[float] = []
    ndcg_list: List[float] = []

    for t, ranked in zip(y_true, ranked_lists):
        try:
            pos = ranked.index(t)
        except ValueError:
            pos = -1

        # MRR component
        if pos >= 0:
            rr_list.append(1.0 / float(pos + 1))
        else:
            rr_list.append(0.0)

        # NDCG@k component
        if 0 <= pos < ndcg_k:
            # DCG với gain=1 duy nhất tại position pos
            dcg = 1.0 / math.log2(pos + 2.0)
            ndcg_list.append(dcg)  # IDCG=1
        else:
            ndcg_list.append(0.0)

    mrr = sum(rr_list) / float(len(rr_list)) if rr_list else 0.0
    ndcg_k_avg = sum(ndcg_list) / float(len(ndcg_list)) if ndcg_list else 0.0
    return mrr, ndcg_k_avg


# ------------------------------------------------
# Helpers: latency
# ------------------------------------------------
def _compute_latency_stats(latencies_ms: List[float]) -> Optional[LatencyStats]:
    if not latencies_ms:
        return None

    xs = sorted(latencies_ms)
    n = len(xs)

    def _percentile(p: float) -> float:
        if n == 1:
            return xs[0]
        k = (n - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return xs[int(k)]
        d0 = xs[f] * (c - k)
        d1 = xs[c] * (k - f)
        return d0 + d1

    avg_ms = sum(xs) / float(n)
    p50 = _percentile(0.5)
    p90 = _percentile(0.9)
    p95 = _percentile(0.95)

    return LatencyStats(avg_ms=avg_ms, p50_ms=p50, p90_ms=p90, p95_ms=p95)


# ------------------------------------------------
# Core runner
# ------------------------------------------------
def run_evaluation(
    samples: List[EvalSample],
    model: Any,
    activities_vocab: List[str],
    measure_cost: bool = False,
    ndcg_k: int = 5,
) -> EvalResult:
    """
    Chạy đánh giá trên list EvalSample với model (HybridSNAP).

    Parameters
    ----------
    samples:
        Danh sách sample đã chuẩn hóa (prefix, y_true, case_id, gold_explanation).
    model:
        Model đã khởi tạo (thường là HybridSNAP).
        Yêu cầu có method:
            predict(prefix, activities_vocab, measure_cost=False) ->
                (y_pred: str, ranked_labels: List[str], meta: Dict[str, Any])
    activities_vocab:
        Tập toàn bộ activity trong process.
    measure_cost:
        Nếu True → đo latency cho từng sample.
    ndcg_k:
        K cho NDCG@k.

    Returns
    -------
    EvalResult
    """

    if not samples:
        raise ValueError("run_evaluation: samples rỗng.")

    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    ranked_lists_all: List[List[str]] = []
    latencies_ms: List[float] = []
    explain_records: List[ExplanationRecord] = []

    for s in samples:
        # đo thời gian nếu cần
        t0 = time.perf_counter()
        y_pred, ranked_labels, meta = model.predict(
            prefix=s.prefix,
            activities_vocab=activities_vocab,
            measure_cost=measure_cost,
        )
        t1 = time.perf_counter()

        y_true_all.append(s.y_true)
        y_pred_all.append(y_pred)
        ranked_lists_all.append(list(ranked_labels))

        if measure_cost:
            lat_ms = (t1 - t0) * 1000.0
            latencies_ms.append(lat_ms)

        # Lấy meta explanation (nếu có)
        expl_meta: Dict[str, Any] = meta.get("explanation", {}) if isinstance(meta, dict) else {}
        expl_text = str(expl_meta.get("text", "")).strip()
        reason_pass = bool(expl_meta.get("reason_pass_local", False))

        # gold_explanation từ sample (nếu được điền khi tạo EvalSample)
        gold_expl = s.gold_explanation

        explain_records.append(
            ExplanationRecord(
                case_id=s.case_id,
                y_true=s.y_true,
                y_pred=y_pred,
                explanation_text=expl_text,
                reason_pass_local=reason_pass,
                human_score=None,
                gold_explanation=gold_expl,
            )
        )

    # 1) Classification metrics
    accuracy = _compute_accuracy(y_true_all, y_pred_all)
    macro_f1 = _compute_macro_f1(y_true_all, y_pred_all)
    mrr, ndcg_at_k = _compute_mrr_and_ndcg(y_true_all, ranked_lists_all, ndcg_k=ndcg_k)

    # 2) Explain metrics
    explain_metrics = summarize_explain_metrics(explain_records, pairs=None)

    # 3) Latency & CPR
    latency_stats = _compute_latency_stats(latencies_ms) if measure_cost else None
    cpr: Optional[float] = None
    if latency_stats is not None and latency_stats.avg_ms > 0.0:
        avg_sec = latency_stats.avg_ms / 1000.0
        cpr = macro_f1 / avg_sec

    return EvalResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        mrr=mrr,
        ndcg_at_k=ndcg_at_k,
        explain_metrics=explain_metrics,
        latency=latency_stats,
        cpr=cpr,
    )
