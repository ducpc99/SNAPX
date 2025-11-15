# src/eval/runner.py
# ------------------
"""
Eval Runner cho S-NAP:
- Accuracy, Macro-F1
- MRR, NDCG@k
- Explain metrics (Reason-Pass, Consistency stub)
- Cost-Performance (latency + CPR)
- Thống kê Guard (tần suất dùng, số candidate bị chặn trung bình)

Thiết kế đơn giản, dễ nối với script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional

from src.hybrid.pipeline import HybridSNAP
from src.eval.cost import measure_latency, estimate_cpr, LatencyStats
from src.eval.explain_metrics import (
    ExplanationRecord,
    summarize_explain_metrics,
)


@dataclass
class EvalSample:
    case_id: str
    prefix: Sequence[str]
    y_true: str


@dataclass
class EvalResult:
    accuracy: float
    macro_f1: float
    mrr: float
    ndcg_at_k: float
    explain_metrics: Dict[str, float]
    latency: Optional[LatencyStats]
    cpr: Optional[float]


# ---------- Classification metrics ----------

def _compute_accuracy(preds: List[str], labels: List[str]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    return correct / len(labels)


def _compute_macro_f1(preds: List[str], labels: List[str]) -> float:
    if not labels:
        return 0.0
    # F1 per class
    classes = sorted(set(labels))
    f1s: List[float] = []
    for c in classes:
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)

        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            # class không xuất hiện trong label
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0.0


def _compute_mrr(
    ranked_candidates_list: List[List[str]],
    labels: List[str],
) -> float:
    """
    Mean Reciprocal Rank (MRR):
    - Với mỗi mẫu:
        rank = vị trí của y_true trong candidate list (tính từ 1)
        rr = 1/rank nếu tìm thấy, else 0
    """
    if not labels:
        return 0.0
    rrs: List[float] = []
    for candidates, y in zip(ranked_candidates_list, labels):
        if y in candidates:
            rank = candidates.index(y) + 1
            rrs.append(1.0 / rank)
        else:
            rrs.append(0.0)
    return sum(rrs) / len(rrs)


def _compute_ndcg_at_k(
    ranked_candidates_list: List[List[str]],
    labels: List[str],
    k: int,
) -> float:
    """
    NDCG@k: đơn giản với ground truth 1 nhãn:
    - Nếu y_true nằm ở vị trí r (1-based) <= k:
        DCG = 1 / log2(r+1)
    - iDCG = 1 (vì ideal là ở vị trí 1)
    """
    import math

    if not labels or k <= 0:
        return 0.0

    scores: List[float] = []
    for candidates, y in zip(ranked_candidates_list, labels):
        if y in candidates:
            rank = candidates.index(y) + 1
            if rank <= k:
                dcg = 1.0 / math.log2(rank + 1)
            else:
                dcg = 0.0
        else:
            dcg = 0.0
        scores.append(dcg)  # iDCG = 1
    return sum(scores) / len(scores)


# ---------- Main eval ----------

def run_evaluation(
    samples: List[EvalSample],
    model: HybridSNAP,
    activities_vocab: Sequence[str],
    measure_cost: bool = False,
    ndcg_k: int = 5,
) -> EvalResult:
    """
    Chạy eval cho 1 model (HybridSNAP).

    Parameters
    ----------
    samples:
        Danh sách EvalSample (case_id, prefix, y_true).
    model:
        Đối tượng HybridSNAP đã setup đầy đủ (seq + graph + sem + guard).
    activities_vocab:
        Vocab activity dùng truyền vào predict_one.
    measure_cost:
        Nếu True → đo thêm latency & CPR.
    ndcg_k:
        K cho NDCG@k.

    Returns
    -------
    EvalResult:
        Gồm classification metrics, explain metrics, cost metrics.
    """
    y_true_list: List[str] = []
    y_pred_list: List[str] = []
    candidates_list: List[List[str]] = []
    expl_records: List[ExplanationRecord] = []

    # thống kê nhẹ về Guard
    guard_used_cnt = 0
    guard_total_blocked = 0

    for s in samples:
        out = model.predict_one(prefix=s.prefix, activities=activities_vocab)
        y_true_list.append(s.y_true)

        y_pred = str(out.get("prediction", ""))
        y_pred_list.append(y_pred)
        candidates_list.append(list(out.get("candidates", [])))

        # Meta block
        meta: Dict[str, Any] = out.get("meta", {}) or {}

        # Explanation metrics
        expl_meta = meta.get("explanation")
        if isinstance(expl_meta, dict) and "text" in expl_meta:
            expl_records.append(
                ExplanationRecord(
                    case_id=s.case_id,
                    y_true=s.y_true,
                    y_pred=y_pred,
                    explanation_text=str(expl_meta.get("text", "")),
                    reason_pass_local=bool(expl_meta.get("reason_pass_local", False)),
                    human_score=None,  # có thể bổ sung từ anotator sau
                )
            )

        # Guard stats (nếu có)
        guard_meta = meta.get("guard", {})
        if isinstance(guard_meta, dict) and guard_meta.get("used", False):
            guard_used_cnt += 1
            num_blocked = guard_meta.get("num_blocked", 0)
            try:
                guard_total_blocked += int(num_blocked)
            except Exception:
                # nếu field kì lạ thì bỏ qua
                pass

    # Classification metrics
    acc = _compute_accuracy(y_pred_list, y_true_list)
    macro_f1 = _compute_macro_f1(y_pred_list, y_true_list)
    mrr = _compute_mrr(candidates_list, y_true_list)
    ndcg = _compute_ndcg_at_k(candidates_list, y_true_list, ndcg_k)

    # Explain metrics
    explain_metrics = summarize_explain_metrics(expl_records)

    # Bổ sung guard metrics vào explain_metrics (cho tiện xem chung 1 chỗ)
    n = len(samples) if samples else 1
    guard_used_rate = guard_used_cnt / float(n)
    guard_avg_blocked = (
        guard_total_blocked / float(guard_used_cnt) if guard_used_cnt > 0 else 0.0
    )
    explain_metrics = dict(explain_metrics)
    explain_metrics["guard_used_rate"] = guard_used_rate
    explain_metrics["guard_avg_blocked"] = guard_avg_blocked

    # Cost-performance
    latency_stats: Optional[LatencyStats] = None
    cpr: Optional[float] = None
    if measure_cost:
        prefixes = [s.prefix for s in samples]
        latency_stats = measure_latency(
            predict_one=lambda p: model.predict_one(p, activities=activities_vocab),
            prefixes=prefixes,
            warmup=3,
            repeat=min(20, max(5, len(prefixes))),
        )
        cpr = estimate_cpr(main_metric=macro_f1, latency_stats=latency_stats)

    return EvalResult(
        accuracy=acc,
        macro_f1=macro_f1,
        mrr=mrr,
        ndcg_at_k=ndcg,
        explain_metrics=explain_metrics,
        latency=latency_stats,
        cpr=cpr,
    )
