# src/sequence/prefix_tree.py

"""
Prefix-Tree (trie) cho S-NAP.

Ý tưởng chính
-------------
- Xây một cây prefix từ tất cả traces:
  + Mỗi node đại diện cho một prefix (chuỗi hoạt động).
  + Mỗi node giữ thống kê: sau prefix này, immediate next activity là gì và xuất hiện bao nhiêu lần.
- Khi dự đoán:
  + Đi xuống cây theo prefix.
  + Nếu node tồn tại, dùng histogram next_counts để lấy top-k hoạt động tiếp theo.

Cấu trúc
--------
- Mỗi node:
    - children: dict[activity, node_con]
    - next_counts: dict[next_activity, count]
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence as Seq
from typing import Dict, List


class _PrefixNode:
    """Node nội bộ của Prefix-Tree."""

    __slots__ = ("children", "next_counts")

    def __init__(self) -> None:
        # activity -> _PrefixNode
        self.children: Dict[str, _PrefixNode] = {}
        # next_activity -> count
        self.next_counts: Dict[str, int] = defaultdict(int)


class PrefixTree:
    """Cây prefix để truy vấn các tiếp diễn khả thi của một prefix."""

    def __init__(self, min_count: int = 1) -> None:
        """
        Parameters
        ----------
        min_count:
            Ngưỡng tối thiểu số lần xuất hiện để giữ 1 activity trong kết quả.
        """
        self.min_count = min_count
        self.root = _PrefixNode()

    # ------------------------------------------------------------------
    # Huấn luyện
    # ------------------------------------------------------------------
    def fit(self, traces: Seq[List[str]]) -> None:
        """
        Xây cây prefix từ danh sách traces.

        Parameters
        ----------
        traces:
            Mỗi phần tử là một trace (list activity id).
        """
        # Reset cây
        self.root = _PrefixNode()

        for trace in traces:
            if not trace:
                continue

            node = self.root
            n = len(trace)

            for i in range(n):
                next_act = trace[i]
                # Với prefix hiện tại (node), "next" sẽ là trace[i]
                node.next_counts[next_act] += 1

                # Di chuyển xuống child tương ứng activity hiện tại để kéo dài prefix
                act = trace[i]
                if act not in node.children:
                    node.children[act] = _PrefixNode()
                node = node.children[act]

    # ------------------------------------------------------------------
    # Suy luận
    # ------------------------------------------------------------------
    def get_top_k(self, prefix: Seq[str], k: int) -> Dict[str, float]:
        """
        Trả về top-k hoạt động tiếp theo và xác suất tương đối (theo tần suất quan sát).

        Parameters
        ----------
        prefix:
            Chuỗi hoạt động đã xảy ra.
        k:
            Số lượng activity tối đa cần trả về.

        Returns
        -------
        dict[next_activity, prob]:
            Xác suất được chuẩn hoá theo node tương ứng với prefix (không liên quan các node khác).
        """
        if k <= 0:
            return {}

        node = self.root
        # Dò theo prefix; nếu không đi được thì trả empty
        for act in prefix:
            child = node.children.get(act)
            if child is None:
                return {}
            node = child

        if not node.next_counts:
            return {}

        # Lọc theo min_count
        filtered = [(a, c) for a, c in node.next_counts.items() if c >= self.min_count]
        if not filtered:
            return {}

        # Sắp xếp theo count giảm dần
        filtered.sort(key=lambda x: x[1], reverse=True)
        top = filtered[:k]

        total = sum(c for _, c in top)
        if total <= 0:
            return {}

        return {a: c / total for a, c in top}
