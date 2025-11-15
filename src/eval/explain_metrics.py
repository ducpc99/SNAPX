# src/eval/explain_metrics.py
# ---------------------------
"""
Metrics cho Explainability:
- Reason-Pass (local)
- Consistency (so sánh dự đoán với/không với explain)

Thiết kế đơn giản, linh hoạt:
- Input là list các dict, tương thích với meta["explanation"]
  và output runner của HybridSNAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ExplanationRecord:
    """
    Thông tin 1 sample để đánh giá explain.

    Attributes
    ----------
    case_id:
        ID duy nhất (vd. trace_id + prefix_len).
    y_true:
        Nhãn đúng (nếu có ground truth).
    y_pred:
        Nhãn mà model dự đoán.
    explanation_text:
        Chuỗi giải thích cuối cùng.
    reason_pass_local:
        Cờ local check (từ generate_explanation).
    human_score:
        Điểm đánh giá của annotator (1-5) nếu có, None nếu chưa.
    """

    case_id: str
    y_true: Optional[str]
    y_pred: str
    explanation_text: str
    reason_pass_local: bool
    human_score: Optional[int] = None


def compute_reason_pass_rate(records: List[ExplanationRecord]) -> float:
    """
    Tính tỉ lệ Reason-Pass (dựa trên reason_pass_local).

    Returns
    -------
    float:
        Giá trị trong [0,1]. 0.0 nếu không có record.
    """
    if not records:
        return 0.0
    num_ok = sum(1 for r in records if r.reason_pass_local)
    return num_ok / len(records)


def compute_avg_length(records: List[ExplanationRecord]) -> float:
    """Độ dài trung bình (số ký tự) của explanation."""
    if not records:
        return 0.0
    total_len = sum(len(r.explanation_text) for r in records)
    return total_len / len(records)


def compute_usefulness_from_human(records: List[ExplanationRecord]) -> float:
    """
    Tính điểm hữu ích trung bình từ human_score (1-5).

    Nếu không có record nào có human_score != None → trả 0.0.
    """
    filtered = [r.human_score for r in records if r.human_score is not None]
    if not filtered:
        return 0.0
    return sum(filtered) / len(filtered)


@dataclass
class ConsistencyPair:
    """
    Cặp kết quả dự đoán để đo Consistency.

    Attributes
    ----------
    case_id:
        ID duy nhất cho sample.
    y_pred_without_exp:
        Dự đoán khi **không** bật explanation (hoặc mode baseline).
    y_pred_with_exp:
        Dự đoán khi bật explanation (hoặc mode explain).
    """

    case_id: str
    y_pred_without_exp: str
    y_pred_with_exp: str


def compute_consistency(pairs: List[ConsistencyPair]) -> float:
    """
    Tỉ lệ Consistency:
    - Bao nhiêu % sample có y_pred_without_exp == y_pred_with_exp.
    """
    if not pairs:
        return 0.0
    same = sum(1 for p in pairs if p.y_pred_without_exp == p.y_pred_with_exp)
    return same / len(pairs)


def summarize_explain_metrics(
    records: List[ExplanationRecord],
    pairs: Optional[List[ConsistencyPair]] = None,
) -> Dict[str, float]:
    """
    Gom Reason-Pass + Avg length + Usefulness + Consistency (nếu có) thành dict.
    """
    result = {
        "reason_pass_rate": compute_reason_pass_rate(records),
        "avg_explanation_length": compute_avg_length(records),
        "avg_usefulness_human": compute_usefulness_from_human(records),
    }
    if pairs is not None:
        result["consistency_rate"] = compute_consistency(pairs)
    return result
