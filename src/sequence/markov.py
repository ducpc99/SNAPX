# src/sequence/markov.py

"""
Mô hình Markov cho dự đoán hoạt động kế tiếp (next-activity).

Ý tưởng chính
-------------
- Học phân phối chuyển tiếp P(next | context) từ các trace huấn luyện.
- Hỗ trợ Markov bậc 1, 2, ... (thực tế ta chỉ cần đến bậc 2 là đủ cho S-NAP).
- Khi dự đoán:
  + Ưu tiên dùng context dài nhất (bậc cao nhất) có trong mô hình,
  + Nếu không có thì back-off xuống bậc thấp hơn (1, rồi 0).

Cấu trúc dữ liệu
----------------
- transition_counts:
    dict[context_tuple, dict[next_activity, count]]
- transition_probs:
    dict[context_tuple, dict[next_activity, prob]]
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence as Seq
from typing import Dict, List, Tuple


Context = Tuple[str, ...]
ProbDist = Dict[str, float]


class MarkovModel:
    """Mô hình Markov đơn giản cho next-activity prediction."""

    def __init__(self, order: int = 1, min_count: int = 1) -> None:
        """
        Parameters
        ----------
        order:
            Bậc của mô hình Markov (1 = chỉ nhìn hoạt động ngay trước).
        min_count:
            Ngưỡng tối thiểu số lần xuất hiện để chấp nhận một cạnh.
        """
        if order < 1:
            raise ValueError("order phải >= 1")

        self.order = order
        self.min_count = min_count

        # Đếm số lần chuyển tiếp (context -> next)
        self._transition_counts: Dict[Context, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Phân phối xác suất P(next | context)
        self._transition_probs: Dict[Context, ProbDist] = {}

    # ------------------------------------------------------------------
    # Huấn luyện
    # ------------------------------------------------------------------
    def fit(self, traces: Seq[List[str]]) -> None:
        """
        Học phân phối chuyển tiếp từ danh sách traces.

        Parameters
        ----------
        traces:
            Mỗi phần tử là một trace (list các activity id).
        """
        self._transition_counts.clear()
        self._transition_probs.clear()

        for trace in traces:
            if not trace:
                continue

            n = len(trace)
            # Với mỗi vị trí i, xem nó là "next" và context là các hoạt động trước đó.
            for i in range(n):
                next_act = trace[i]

                # Với mỗi bậc từ 1..order
                for o in range(1, self.order + 1):
                    if i - o < 0:
                        continue
                    context = tuple(trace[i - o : i])
                    self._transition_counts[context][next_act] += 1

        # Chuyển sang xác suất
        for context, counts in self._transition_counts.items():
            total = sum(counts.values())
            # Nếu tổng số lần < min_count thì bỏ qua context này
            if total < self.min_count:
                continue

            probs: ProbDist = {}
            for act, c in counts.items():
                if c >= self.min_count:
                    probs[act] = c / total

            if probs:
                self._transition_probs[context] = probs

    # ------------------------------------------------------------------
    # Suy luận
    # ------------------------------------------------------------------
    def get_probs(self, context_seq: Seq[str]) -> ProbDist:
        """
        Lấy phân phối P(next | context) với chiến lược back-off.

        Parameters
        ----------
        context_seq:
            Dãy hoạt động đã xảy ra (prefix).

        Returns
        -------
        ProbDist:
            dict[next_activity, prob] hoặc {} nếu không có dữ liệu.
        """
        if not self._transition_probs or not context_seq:
            return {}

        # Thử từ bậc cao nhất xuống thấp nhất
        ctx_len = min(len(context_seq), self.order)
        for L in range(ctx_len, 0, -1):
            context = tuple(context_seq[-L:])
            probs = self._transition_probs.get(context)
            if probs:
                return dict(probs)  # copy nhẹ

        # Không tìm được context nào trong mô hình
        return {}
