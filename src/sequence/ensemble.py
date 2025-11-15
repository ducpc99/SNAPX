# src/sequence/ensemble.py

"""
Ensemble giữa Markov và Prefix-Tree.

Ý tưởng
-------
- MarkovModel cho P_M(next | context)
- PrefixTree cho P_T(next | prefix)
- Ensemble score:
    score_hybrid = alpha * P_M + (1 - alpha) * P_T

Chiến lược:
- Nếu cả hai đều rỗng → không dự đoán được → trả {}.
- Nếu có nhưng tất cả score_hybrid = 0 → fallback về nguồn có dữ liệu (Markov hoặc PrefixTree).
"""

from __future__ import annotations

from collections.abc import Sequence as Seq
from typing import Dict, List

from .markov import MarkovModel
from .prefix_tree import PrefixTree


class SequenceEnsemble:
    """Kết hợp Markov + PrefixTree để tạo candidate cho S-NAP."""

    def __init__(
        self,
        alpha: float = 0.5,
        markov_order: int = 2,
        min_count: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        alpha:
            Trọng số cho Markov (0.0 → chỉ dùng PrefixTree, 1.0 → chỉ dùng Markov).
        markov_order:
            Bậc Markov (thường dùng 1 hoặc 2).
        min_count:
            Ngưỡng min_count chung cho cả Markov và PrefixTree.
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha phải nằm trong [0.0, 1.0]")

        self.alpha = alpha
        self.markov = MarkovModel(order=markov_order, min_count=min_count)
        self.prefix_tree = PrefixTree(min_count=min_count)

    # ------------------------------------------------------------------
    # Huấn luyện
    # ------------------------------------------------------------------
    def fit(self, traces: Seq[List[str]]) -> None:
        """Fit cả Markov và PrefixTree trên danh sách traces."""
        self.markov.fit(traces)
        self.prefix_tree.fit(traces)

    # ------------------------------------------------------------------
    # Suy luận
    # ------------------------------------------------------------------
    def propose_candidates(self, prefix: Seq[str], k: int) -> Dict[str, float]:
        """
        Kết hợp Markov & PrefixTree để lấy top-k ứng viên tiếp theo.

        Parameters
        ----------
        prefix:
            Chuỗi hoạt động đã xảy ra.
        k:
            Số lượng candidate tối đa muốn lấy.

        Returns
        -------
        dict[activity, prob]:
            Xác suất đã chuẩn hoá (tổng ~ 1.0) trên top-k candidate.
        """
        if k <= 0:
            return {}

        markov_probs = self.markov.get_probs(prefix)
        tree_probs = self.prefix_tree.get_top_k(prefix, k=k)

        # Nếu cả hai đều rỗng → không có gì để suy luận
        if not markov_probs and not tree_probs:
            return {}

        candidates = set(markov_probs.keys()) | set(tree_probs.keys())
        hybrid_scores: Dict[str, float] = {}

        for act in candidates:
            pm = markov_probs.get(act, 0.0)
            pt = tree_probs.get(act, 0.0)
            score = self.alpha * pm + (1.0 - self.alpha) * pt
            hybrid_scores[act] = score

        # Nếu tất cả score = 0 → fallback về một nguồn có dữ liệu
        sum_scores = sum(hybrid_scores.values())
        if sum_scores <= 0:
            # Ưu tiên Markov nếu có
            if markov_probs:
                return self._normalize_and_cut(markov_probs, k)
            return self._normalize_and_cut(tree_probs, k)

        # Lấy top-k theo hybrid_scores
        return self._normalize_and_cut(hybrid_scores, k)

    # ------------------------------------------------------------------
    # Helper nội bộ
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_and_cut(scores: Dict[str, float], k: int) -> Dict[str, float]:
        """Chọn top-k theo score và chuẩn hoá lại thành phân phối xác suất."""
        if not scores:
            return {}

        # Sắp xếp theo score giảm dần
        items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top = items[:k]

        total = sum(s for _, s in top)
        if total <= 0:
            # Nếu tổng ≤ 0, giữ nguyên score (coi như tần suất thô)
            return dict(top)

        return {a: s / total for a, s in top}
