# src/semantics/instruction_memory.py
# -----------------------------------
"""
InstructionMemory cho S-NAPX (IT few-shot).

Ý tưởng
-------
- Đọc dữ liệu từ S-NAP_instructions.csv (thông qua data.loader.load_instructions_data).
- Chuẩn hoá thành danh sách ITExample:
    + prefix: list[str]
    + next_activity: str (nếu có cột 'next')
    + unique_activities: list[str]
    + instruction: str
    + output: str
    + model_id / revision_id: str (nếu có)
- Hỗ trợ truy vấn theo prefix:
    + query_by_prefix(prefix, k): trả về top-k ví dụ giống nhất.
- Mỗi lần truy vấn dùng heuristic similarity đơn giản:
    + Kết hợp Longest Common Prefix (LCP) + Jaccard set của activities.

Module này không tự động gắn vào SemanticsLLM;
việc "nhúng few-shot IT vào prompt" sẽ được thực hiện ở lớp cao hơn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

import pandas as pd

from src.data.loader import load_instructions_data


# ----------------------------------------------------------------------
# Helpers nội bộ (copy nhẹ từ data.loader để độc lập)
# ----------------------------------------------------------------------
def _safe_eval_list_local(x: Any) -> List[str]:
    """Parse list-like safely từ str/list/tuple; cũng chấp nhận 'a,b,c'."""
    import ast

    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") or s.startswith("("):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(t) for t in v]
        except Exception:
            pass
    # fallback: comma-separated
    return [t.strip() for t in s.split(",") if t.strip()]


def _drop_end_tokens_local(seq: Sequence[str]) -> List[str]:
    """Remove end tokens like [END], None, etc."""
    END_TOKENS = {"[END]", "END", "<END>", "None", None, ""}
    return [t for t in seq if str(t).strip() not in END_TOKENS]


def _pick_column(df: pd.DataFrame, key: str) -> Optional[str]:
    """Tìm cột tương ứng theo nhóm synonym đơn giản (copy từ data.loader)."""
    COLUMN_SYNONYMS = {
        "unique_activities": ["unique_activities", "activities", "set_of_activities", "activity_set"],
        "prefix": ["prefix", "partial_trace", "sequence", "seq", "history"],
        "next": ["next", "gold_answer", "next_activity", "target", "label_next"],
        "model_id": ["model_id", "process_id", "domain_id"],
        "revision_id": ["revision_id", "rev_id", "split_rev"],
        "instruction": ["instruction", "prompt", "query"],
        "output": ["output", "answer", "response"],
    }
    for cand in COLUMN_SYNONYMS.get(key, []):
        if cand in df.columns:
            return cand
    return None


# ----------------------------------------------------------------------
# Dataclass ITExample
# ----------------------------------------------------------------------
@dataclass
class ITExample:
    """Một ví dụ instruction-tuning đơn cho S-NAP."""

    instruction: str
    output: str
    prefix: List[str]
    next_activity: Optional[str]
    unique_activities: List[str]
    model_id: Optional[str]
    revision_id: Optional[str]


# ----------------------------------------------------------------------
# InstructionMemory
# ----------------------------------------------------------------------
class InstructionMemory:
    """
    Bộ nhớ IT đơn giản cho S-NAP.

    Thao tác chính:
    - from_csv(...) → build memory từ S-NAP_instructions.csv.
    - query_by_prefix(prefix, k) → lấy top-k ví dụ giống nhất.
    """

    def __init__(self, examples: List[ITExample]) -> None:
        self.examples: List[ITExample] = examples

    # ---------------------- Factory ----------------------
    @classmethod
    def from_csv(
        cls,
        dataset_path: str,
        limit: Optional[int] = None,
        max_examples: Optional[int] = None,
        min_prefix_len: int = 1,
    ) -> "InstructionMemory":
        """
        Build InstructionMemory từ file S-NAP_instructions.csv.

        Parameters
        ----------
        dataset_path:
            Đường dẫn tới file CSV S-NAP_instructions.csv.
        limit:
            Nếu không None → chỉ load trước N dòng từ CSV (ở tầng loader).
        max_examples:
            Nếu không None → sau khi load, chỉ giữ ngẫu nhiên tối đa N ví dụ trong memory.
        min_prefix_len:
            Bỏ các ví dụ có prefix ngắn hơn giá trị này.
        """
        df = load_instructions_data(dataset_path=dataset_path, limit=limit)

        # Chuẩn bị tên cột
        col_acts = _pick_column(df, "unique_activities") or "unique_activities"
        col_instr = _pick_column(df, "instruction") or "instruction"
        col_out = _pick_column(df, "output") or "output"
        col_prefix = _pick_column(df, "prefix")
        col_next = _pick_column(df, "next")
        col_model = _pick_column(df, "model_id")
        col_rev = _pick_column(df, "revision_id")

        if col_prefix is None:
            # Nếu không có prefix → không đủ cho S-NAP few-shot
            raise ValueError(
                "Không tìm thấy cột prefix trong S-NAP_instructions.csv; "
                "InstructionMemory cần prefix để tính similarity."
            )

        # Chuyển đổi/sạch dữ liệu
        examples: List[ITExample] = []

        for _, row in df.iterrows():
            instr = str(row.get(col_instr, "")).strip()
            out = str(row.get(col_out, "")).strip()
            if not instr or not out:
                continue

            # prefix
            raw_prefix = row.get(col_prefix)
            prefix_list = _safe_eval_list_local(raw_prefix)
            prefix_list = _drop_end_tokens_local(prefix_list)
            if len(prefix_list) < min_prefix_len:
                continue

            # next activity (nếu có)
            next_act: Optional[str] = None
            if col_next is not None:
                nv = row.get(col_next)
                if pd.notna(nv):
                    next_act = str(nv).strip() or None

            # unique_activities
            raw_acts = row.get(col_acts)
            ua_list = _safe_eval_list_local(raw_acts)
            ua_list = _drop_end_tokens_local(ua_list)

            # model / revision
            model_id = str(row.get(col_model)).strip() if col_model is not None else None
            if model_id == "nan":
                model_id = None
            revision_id = str(row.get(col_rev)).strip() if col_rev is not None else None
            if revision_id == "nan":
                revision_id = None

            ex = ITExample(
                instruction=instr,
                output=out,
                prefix=prefix_list,
                next_activity=next_act,
                unique_activities=ua_list,
                model_id=model_id,
                revision_id=revision_id,
            )
            examples.append(ex)

        # Nếu cần, random giảm kích thước bộ nhớ
        if max_examples is not None and len(examples) > max_examples:
            import random

            random.shuffle(examples)
            examples = examples[:max_examples]

        return cls(examples=examples)

    # ---------------------- Query ----------------------
    def query_by_prefix(
        self,
        prefix: Sequence[str],
        k: int = 3,
        model_id: Optional[str] = None,
    ) -> List[ITExample]:
        """
        Lấy top-k ví dụ trong memory có prefix giống nhất với query prefix.

        Parameters
        ----------
        prefix:
            Trace prefix cần tìm ví dụ tương tự (list[str]).
        k:
            Số lượng ví dụ trả về.
        model_id:
            Nếu không None → ưu tiên ví dụ cùng model_id khi tính similarity.
        """
        q_pref = [str(a) for a in prefix]
        if not q_pref or not self.examples:
            return []

        scored: List[Tuple[float, ITExample]] = []
        for ex in self.examples:
            score = self._prefix_similarity(q_pref, ex.prefix)

            # bonus nhẹ nếu model_id trùng
            if model_id is not None and ex.model_id is not None and model_id == ex.model_id:
                score += 0.05

            scored.append((score, ex))

        # sort desc theo score
        scored.sort(key=lambda t: t[0], reverse=True)
        return [ex for score, ex in scored[: max(k, 0)] if score > 0.0]

    # ---------------------- Similarity heuristic ----------------------
    @staticmethod
    def _prefix_similarity(q: Sequence[str], ex: Sequence[str]) -> float:
        """
        Heuristic similarity giữa 2 prefix.

        Thành phần:
        - LCP (Longest Common Prefix) normalized.
        - Jaccard similarity giữa 2 tập activity trong prefix.
        - score = 0.7 * lcp_norm + 0.3 * jaccard

        Giá trị trả về ∈ [0,1].
        """
        if not q or not ex:
            return 0.0

        # LCP
        lcp = 0
        for a, b in zip(q, ex):
            if a == b:
                lcp += 1
            else:
                break
        lcp_norm = lcp / float(max(len(q), len(ex)))

        # Jaccard
        set_q = set(q)
        set_ex = set(ex)
        inter = len(set_q & set_ex)
        union = len(set_q | set_ex) or 1
        jacc = inter / float(union)

        score = 0.7 * lcp_norm + 0.3 * jacc
        # đảm bảo nằm [0,1] (có thể dư số do float)
        return max(0.0, min(1.0, score))


__all__ = ["ITExample", "InstructionMemory"]
