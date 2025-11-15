# src/graph_reasoning/reasoner.py
# -------------------------------
"""
GraphReasoner
=============

Mục tiêu
--------
- Dùng DFG để "gắn prior" lên danh sách ứng viên top-k từ SequenceEnsemble.
- Hỗ trợ:
    • hard_mask=True:
        - Loại bỏ hẳn những ứng viên không có cạnh prev→cand trong DFG.
    • hard_mask=False:
        - Không loại bỏ, chỉ cộng thêm soft boost dựa trên P(cand|prev).
- Trả về:
    • new_rank:   list tên ứng viên đã tái xếp hạng
    • new_scores: dict[ứng viên, điểm mới]
    • meta:       dict để logging/giải thích

API chính
---------
GraphReasoner.rerank_candidates(prev, ranked, scores)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dfg import DFG


RankedLabels = List[str]


@dataclass
class GraphReasonerConfig:
    """
    Cấu hình cho GraphReasoner.

    Attributes
    ----------
    enabled:
        Cho phép/bật module này hay không.
    hard_mask:
        Nếu True, loại bỏ ứng viên không có cạnh trong DFG.
        Nếu False, không loại bỏ, chỉ cộng soft boost.
    boost_factor:
        Hệ số tăng điểm dựa trên P(b|a) trong DFG.
        new_score = old_score + boost_factor * P(b|a).
    min_prob_keep:
        Khi hard_mask=True, nếu P(b|a) < min_prob_keep → coi như cạnh không tồn tại.
    """

    enabled: bool = True
    hard_mask: bool = True
    boost_factor: float = 1.0
    min_prob_keep: float = 0.0


class GraphReasoner:
    """Lớp điều phối suy luận đồ thị cho lớp Sequence."""

    def __init__(
        self,
        dfg: Optional[DFG],
        cfg: Optional[GraphReasonerConfig] = None,
    ) -> None:
        """
        Parameters
        ----------
        dfg:
            Đối tượng DFG (đã fit). Nếu None → GraphReasoner sẽ không làm gì.
        cfg:
            GraphReasonerConfig. Nếu None → dùng giá trị mặc định.
        """
        self._dfg = dfg
        self.cfg = cfg or GraphReasonerConfig()

    # ------------------------------------------------------------------
    # API chính
    # ------------------------------------------------------------------
    def rerank_candidates(
        self,
        prev: Optional[str],
        ranked: RankedLabels,
        scores: Dict[str, float],
    ) -> Tuple[RankedLabels, Dict[str, float], Dict[str, Any]]:
        """
        Áp dụng prior DFG để tái xếp hạng danh sách ứng viên.

        Args
        ----
        prev:
            Hoạt động liền trước (prefix[-1]) hoặc None nếu prefix rỗng.
        ranked:
            Danh sách ứng viên đã xếp hạng từ Sequence layer (list activity id).
        scores:
            Map activity id → điểm số hiện tại từ Sequence layer.

        Returns
        -------
        new_ranked:
            Danh sách ứng viên sau khi áp DFG.
        new_scores:
            Điểm số đã cập nhật (có thể bị cộng boost / loại bỏ).
        meta:
            Thông tin diễn giải:
            - 'used': bool
            - 'dropped': list ứng viên bị loại (nếu hard_mask=True)
            - 'boosted': dict[ứng viên, boost_value]
            - 'config': cấu hình sử dụng
        """
        # Nếu tắt hoặc không có DFG → trả về nguyên trạng
        if (not self.cfg.enabled) or (self._dfg is None) or (not ranked):
            return ranked, dict(scores), {"used": False}

        prev = str(prev) if prev is not None else None
        new_scores: Dict[str, float] = dict(scores)

        dropped: List[str] = []
        boosted: Dict[str, float] = {}

        # Duyệt từng candidate theo thứ tự hiện tại
        for cand in ranked:
            c = str(cand)
            base_score = new_scores.get(c, 0.0)
            if base_score is None:
                base_score = 0.0

            # P(cand|prev) trong DFG
            p = self._dfg.prob(prev, c)

            if self.cfg.hard_mask:
                # Nếu cạnh không tồn tại hoặc quá nhỏ → loại
                if p < self.cfg.min_prob_keep:
                    new_scores.pop(c, None)
                    dropped.append(c)
                    continue

            # Soft boost (kể cả khi hard_mask=True nhưng p>=min_prob_keep)
            if p > 0.0 and self.cfg.boost_factor != 0.0:
                boost = self.cfg.boost_factor * p
                new_scores[c] = base_score + boost
                boosted[c] = boost
            else:
                new_scores[c] = base_score

        # Sắp xếp lại theo điểm mới (giảm dần), bỏ những cand đã bị drop
        filtered_items = sorted(new_scores.items(), key=lambda x: x[1], reverse=True)
        new_ranked = [a for a, _ in filtered_items]

        meta: Dict[str, Any] = {
            "used": True,
            "dropped": dropped,
            "boosted": boosted,
            "config": {
                "hard_mask": self.cfg.hard_mask,
                "boost_factor": self.cfg.boost_factor,
                "min_prob_keep": self.cfg.min_prob_keep,
            },
        }
        return new_ranked, new_scores, meta

    # ------------------------------------------------------------------
    # Helpers (tiện ích)
    # ------------------------------------------------------------------
    def allow(self, a: Optional[str], b: str) -> bool:
        """Tra cứu nhanh tính hợp lệ của cạnh a→b trong DFG."""
        if self._dfg is None:
            return True
        return self._dfg.allow(a, b)

    def prob(self, a: Optional[str], b: str) -> float:
        """Lấy P(b|a) trong DFG (0.0 nếu không tồn tại)."""
        if self._dfg is None:
            return 0.0
        return self._dfg.prob(a, b)


__all__ = ["GraphReasoner", "GraphReasonerConfig"]
