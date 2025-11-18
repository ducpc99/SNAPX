# src/eval/explain_metrics.py
# ---------------------------
"""
Metrics cho Explainability:
- Reason-Pass (local)
- Consistency (so sánh dự đoán với/không với explain)
- (Mới) Token-level overlap giữa predicted explanation và gold_explanation (IT output)

Thiết kế đơn giản, linh hoạt:
- Input là list ExplanationRecord, tương thích với meta["explanation"]
  và output runner của HybridSNAP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import re


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
        Chuỗi giải thích cuối cùng (model sinh ra).
    reason_pass_local:
        Cờ local check (từ generate_explanation).
    human_score:
        Điểm đánh giá của annotator (1-5) nếu có, None nếu chưa.
    gold_explanation:
        (Mới) Giải thích “chuẩn” từ dataset IT (cột `output`), nếu có.
        Nếu không có (chạy trên S-NAP thường) → None.
    """

    case_id: str
    y_true: Optional[str]
    y_pred: str
    explanation_text: str
    reason_pass_local: bool
    human_score: Optional[int] = None
    gold_explanation: Optional[str] = None  # NEW


@dataclass
class ConsistencyPair:
    """
    Cặp để đo Consistency.

    - y_pred_no_explain: dự đoán của model khi không cung cấp explanation.
    - y_pred_with_explain: dự đoán của model khi có explanation (student-mode).
    """

    case_id: str
    y_pred_no_explain: str
    y_pred_with_explain: str


# ------------------------------------------------
# Reason-Pass / Length / Human usefulness
# ------------------------------------------------
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
    """
    Tính độ dài trung bình của explanation_text (tính theo số ký tự).

    Nếu không có record → trả 0.0.
    """
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


# ------------------------------------------------
# (Mới) Token-overlap giữa predicted vs gold_explanation
# ------------------------------------------------
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize_simple(text: str) -> List[str]:
    """
    Tokenize đơn giản:
    - Lower-case.
    - Lấy các chuỗi \w+ (hỗ trợ Unicode).
    """
    if not text:
        return []
    text = text.lower()
    return _TOKEN_RE.findall(text)


def compute_token_overlap_scores(records: List[ExplanationRecord]) -> Dict[str, float]:
    """
    Tính precision / recall / F1 trung bình (token-level) giữa
    explanation_text (model) và gold_explanation (IT output).

    Chỉ tính trên những record có gold_explanation không rỗng.
    Nếu không có record đủ điều kiện → trả tất cả 0.0.
    """
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for r in records:
        if not r.gold_explanation:
            continue

        pred_tokens = _tokenize_simple(r.explanation_text)
        gold_tokens = _tokenize_simple(r.gold_explanation)

        if not pred_tokens or not gold_tokens:
            continue

        pred_set = set(pred_tokens)
        gold_set = set(gold_tokens)
        inter = pred_set & gold_set

        if not inter:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            continue

        p = len(inter) / float(len(pred_set))
        q = len(inter) / float(len(gold_set))
        if p + q > 0.0:
            f1 = 2.0 * p * q / (p + q)
        else:
            f1 = 0.0

        precisions.append(p)
        recalls.append(q)
        f1s.append(f1)

    if not precisions:
        return {
            "token_precision": 0.0,
            "token_recall": 0.0,
            "token_f1": 0.0,
        }

    def _avg(xs: List[float]) -> float:
        return sum(xs) / float(len(xs)) if xs else 0.0

    return {
        "token_precision": _avg(precisions),
        "token_recall": _avg(recalls),
        "token_f1": _avg(f1s),
    }


# ------------------------------------------------
# Consistency metrics (stub)
# ------------------------------------------------
def compute_consistency(pairs: List[ConsistencyPair]) -> float:
    """
    Tính tỉ lệ consistency:
    - 1 nếu y_pred_no_explain == y_pred_with_explain
    - 0 nếu khác.

    Nếu không có pair → trả 0.0.
    """
    if not pairs:
        return 0.0
    num_same = sum(1 for p in pairs if p.y_pred_no_explain == p.y_pred_with_explain)
    return num_same / len(pairs)


# ------------------------------------------------
# Summarize
# ------------------------------------------------
def summarize_explain_metrics(
    records: List[ExplanationRecord],
    pairs: Optional[List[ConsistencyPair]] = None,
) -> Dict[str, float]:
    """
    Gom Reason-Pass + Avg length + Usefulness + Consistency (nếu có)
    + (mới) token-overlap metrics thành dict.
    """
    result: Dict[str, float] = {
        "reason_pass_rate": compute_reason_pass_rate(records),
        "avg_explanation_length": compute_avg_length(records),
        "avg_usefulness_human": compute_usefulness_from_human(records),
    }

    # Token-level overlap với gold_explanation (nếu có)
    overlap_scores = compute_token_overlap_scores(records)
    result.update(overlap_scores)

    # Consistency (nếu có)
    if pairs is not None:
        result["consistency_rate"] = compute_consistency(pairs)

    return result
