"""
src/graph_reasoning/priors.py
-----------------------------
GraphPriors: dùng Directed Flow Graph (DFG) để sinh "prior" (ưu tiên) cho
các ứng viên top-k trong Sequence Layer.

Chức năng chính
---------------
- Lấy xác suất chuyển tiếp P(b|a) từ DFG làm prior.
- Nếu bật `hard_mask`, loại bỏ ứng viên không hợp lệ (không có cạnh a→b trong DFG).
- Nếu tắt `hard_mask`, chỉ cộng điểm "prior_boost * P(b|a)" để điều chỉnh độ tin cậy.
- Kết quả được dùng để tái xếp hạng (re-rank) danh sách ứng viên top-k trước khi
  đưa vào Semantics LLM.

Luồng dữ liệu:
--------------
Sequence scores  ─► GraphPriors.reweight() ─► Re-ranked scores ─► Semantics LLM

API chính:
----------
GraphPriors(dfg: DFGBuilder, hard_mask=True, prior_boost=0.3)

• prior(prev, cand) -> float
    Trả về xác suất P(cand|prev) từ DFG.

• reweight(ranked, scores, prev)
    Áp dụng prior cho danh sách ứng viên (ranked) dựa trên DFG(prev→cand).
    Trả về (new_ranked, new_scores, meta).

Meta trả về:
------------
{
  "used": True,
  "hard_mask": bool,
  "prior_boost": float,
  "dropped": [...],
  "hard_mask_triggered": bool
}
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Any
from .dfg import DFGBuilder


class GraphPriors:
    """
    Lớp GraphPriors dùng Directed Flow Graph (DFG) để sinh prior cho các ứng viên trong top-k của Sequence Layer.
    
    Chức năng chính:
    - Lấy xác suất chuyển tiếp P(b|a) từ DFG làm prior cho các ứng viên.
    - Nếu bật `hard_mask`, loại bỏ các ứng viên không hợp lệ (không có cạnh a→b trong DFG).
    - Nếu tắt `hard_mask`, chỉ cộng thêm điểm "prior_boost * P(b|a)" để điều chỉnh độ tin cậy của các ứng viên.
    - Sau khi tái xếp hạng, các ứng viên được đưa vào Semantics LLM để chọn ứng viên cuối cùng.

    Luồng dữ liệu:
    - Sequence scores ─► GraphPriors.reweight() ─► Re-ranked scores ─► Semantics LLM
    """

    def __init__(self, dfg: DFGBuilder, hard_mask: bool = True, prior_boost: float = 0.3):
        """
        Khởi tạo lớp GraphPriors.

        Args:
            dfg: Đối tượng DFGBuilder đã được fit từ các traces.
            hard_mask: Nếu True, loại bỏ các ứng viên không có cạnh hợp lệ (prev→cand) trong DFG.
            prior_boost: Hệ số cộng thêm cho P(b|a), khoảng 0.2–0.5 là hợp lý.
        """
        self.dfg = dfg  # Đối tượng DFGBuilder chứa thông tin về các chuyển tiếp và xác suất của chúng
        self.hard_mask = bool(hard_mask)  # Nếu True, ứng viên không hợp lệ sẽ bị loại bỏ hoàn toàn
        self.prior_boost = float(prior_boost)  # Hệ số cộng thêm cho P(b|a)

    # ------------------------------------------------------------------ #
    def prior(self, prev: str | None, cand: str) -> float:
        """
        Trả về xác suất P(cand|prev) từ DFG (0.0 nếu không tồn tại).

        Args:
            prev: Hoạt động trước đó trong chuỗi.
            cand: Ứng viên đang được xét.

        Returns:
            Xác suất P(cand|prev) từ DFG.
        """
        return self.dfg.prob(prev, cand)

    # ------------------------------------------------------------------ #
    def reweight(
        self,
        ranked: List[str],
        scores: Dict[str, float],
        prev: str | None,
    ) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
        """
        Tái xếp hạng danh sách ứng viên (ranked) dựa trên prior từ DFG.
        - Nếu hard_mask=True: loại bỏ các ứng viên không có cạnh hợp lệ từ prev → cand.
        - Nếu hard_mask=False: vẫn giữ lại ứng viên, nhưng cộng thêm điểm dựa vào xác suất P(b|a).

        Args:
            ranked: Danh sách các ứng viên đã được xếp hạng.
            scores: Điểm ban đầu của các ứng viên.
            prev: Hoạt động trước đó trong chuỗi (None nếu là hoạt động đầu tiên).

        Returns:
            - Danh sách các ứng viên đã được tái xếp hạng.
            - Điểm mới của các ứng viên.
            - Metadata bao gồm thông tin về việc áp dụng hard_mask và prior_boost.
        """
        new_scores: Dict[str, float] = {}  # Lưu trữ điểm mới sau khi cộng thêm prior
        dropped: List[str] = []  # Lưu các ứng viên bị loại bỏ

        for a in ranked:
            base = float(scores.get(a, 0.0))  # Điểm ban đầu của ứng viên
            p = self.prior(prev, a)  # Lấy prior từ DFG (P(b|a))

            if self.hard_mask and prev is not None and not self.dfg.allow(prev, a):
                # Nếu hard_mask=True và không có cạnh hợp lệ (prev→a) trong DFG, loại bỏ ứng viên
                dropped.append(a)
                continue

            # Cộng thêm điểm prior nếu hard_mask=False, sử dụng prior_boost
            new_scores[a] = base + self.prior_boost * p

        # Nếu tất cả ứng viên bị loại bỏ, trả lại danh sách gốc
        if not new_scores:
            new_scores = dict(scores)
            out_ranked = list(ranked)
            meta = {"hard_mask_triggered": True, "dropped": dropped}
        else:
            # Sắp xếp lại các ứng viên theo điểm mới
            out_ranked = sorted(new_scores.keys(), key=lambda x: -new_scores[x])
            meta = {"hard_mask_triggered": False, "dropped": dropped}

        # Thêm metadata về hard_mask và prior_boost vào kết quả
        meta.update({
            "hard_mask": self.hard_mask,
            "prior_boost": self.prior_boost,
            "used": True,
        })
        return out_ranked, new_scores, meta
