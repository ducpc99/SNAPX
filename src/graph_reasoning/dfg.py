# src/graph_reasoning/dfg.py
# --------------------------
"""
Directed-Flow-Graph (DFG) builder dùng cho GraphReasoner & Guard.

Mục tiêu
--------
- Xây DFG từ traces:
    • Đếm số lần a→b xuất hiện liên tiếp (directly-follows).
    • Tính P(b | a) = count(a→b) / sum_b' count(a→b').
- Hỗ trợ:
    • min_count: ngưỡng lọc cạnh quá hiếm.
    • min_ratio: ngưỡng lọc theo tỉ lệ count(a→b) / sum(a→*).
- Cung cấp API:
    • prob(a, b)    → P(b|a) (0.0 nếu không có cạnh).
    • allow(a, b)   → bool, cạnh có tồn tại sau khi lọc không.
    • outgoing(a)   → dict[b, prob] cho mọi b thỏa ngưỡng.

Ghi chú
-------
- a = None: hiểu là "không có ràng buộc" → allow = True, prob = 0.0.
  (dùng cho trường hợp prefix rỗng).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence as Seq
from typing import Dict, List, Optional, Tuple


Edge = Tuple[str, str]
ProbDist = Dict[str, float]


class DFG:
    """Lớp DFG nhẹ cho S-NAP."""

    def __init__(self, min_count: int = 1, min_ratio: float = 0.0) -> None:
        """
        Parameters
        ----------
        min_count:
            Số lần tối thiểu một cạnh a→b phải xuất hiện để được giữ lại.
        min_ratio:
            Ngưỡng tỉ lệ count(a→b) / sum(a→*) tối thiểu để cạnh được giữ lại.
        """
        if min_count < 1:
            raise ValueError("min_count phải >= 1")
        if not (0.0 <= min_ratio <= 1.0):
            raise ValueError("min_ratio phải nằm trong [0.0, 1.0]")

        self.min_count = min_count
        self.min_ratio = min_ratio

        # Đếm thô
        self._edge_counts: Dict[Edge, int] = defaultdict(int)
        self._outgoing_totals: Dict[str, int] = defaultdict(int)

        # Sau khi fit:
        self._edge_probs: Dict[Edge, float] = {}
        self._outgoing_probs: Dict[str, ProbDist] = {}

    # ------------------------------------------------------------------
    # Xây dựng từ traces
    # ------------------------------------------------------------------
    def fit(self, traces: Seq[List[str]]) -> None:
        """Học DFG từ danh sách traces."""
        self._edge_counts.clear()
        self._outgoing_totals.clear()
        self._edge_probs.clear()
        self._outgoing_probs.clear()

        # 1) Đếm trực tiếp a→b
        for trace in traces:
            if not trace:
                continue
            for a, b in zip(trace, trace[1:]):
                a = str(a)
                b = str(b)
                self._edge_counts[(a, b)] += 1
                self._outgoing_totals[a] += 1

        # 2) Lọc & tính xác suất
        for (a, b), cnt in self._edge_counts.items():
            if cnt < self.min_count:
                continue
            total = self._outgoing_totals.get(a, 0)
            if total <= 0:
                continue
            ratio = cnt / total
            if ratio < self.min_ratio:
                continue

            prob = ratio
            self._edge_probs[(a, b)] = prob
            if a not in self._outgoing_probs:
                self._outgoing_probs[a] = {}
            self._outgoing_probs[a][b] = prob

    # ------------------------------------------------------------------
    # API truy vấn
    # ------------------------------------------------------------------
    def prob(self, a: Optional[str], b: str) -> float:
        """
        Lấy P(b|a) trong DFG sau khi lọc.

        - Nếu a=None → không ràng buộc → trả 0.0.
        - Nếu cạnh không tồn tại → 0.0.
        """
        if a is None or b is None:
            return 0.0
        return self._edge_probs.get((str(a), str(b)), 0.0)

    def allow(self, a: Optional[str], b: str) -> bool:
        """
        Kiểm tra cạnh a→b có được giữ trong DFG sau khi lọc không.

        - Nếu a=None hoặc b=None → True (không ràng buộc).
        """
        if a is None or b is None:
            return True
        return (str(a), str(b)) in self._edge_probs

    def outgoing(self, a: str) -> ProbDist:
        """
        Trả về phân phối outgoing P(* | a) sau khi lọc.

        Nếu a không có trong DFG → {}.
        """
        return dict(self._outgoing_probs.get(str(a), {}))

    # ------------------------------------------------------------------
    # Tiện ích
    # ------------------------------------------------------------------
    @classmethod
    def from_traces(
        cls,
        traces: Seq[List[str]],
        min_count: int = 1,
        min_ratio: float = 0.0,
    ) -> "DFG":
        """Hàm tiện ích: tạo & fit DFG từ traces."""
        dfg = cls(min_count=min_count, min_ratio=min_ratio)
        dfg.fit(traces)
        return dfg


__all__ = ["DFG"]
