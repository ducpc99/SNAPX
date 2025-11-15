from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import ast
from collections import Counter

import pandas as pd

Pair = Tuple[str, str]


@dataclass
class ActivityGuardConfig:
    """
    Cấu hình cho ActivityGuard (guard theo cặp activity (prev, next)).
    """

    # Số lần tối thiểu quan sát cặp (prev,next)
    min_count: int = 1
    # Tỷ lệ tối thiểu của cặp này trong tổng số lần xuất hiện prev
    min_ratio: float = 0.0
    # Ngưỡng majority vote: >= threshold -> allowed
    threshold: float = 0.5
    # Nếu True, cặp chưa từng thấy vẫn được phép (guard chỉ chặn những cặp chắc chắn xấu)
    keep_unseen: bool = True


@dataclass
class ActivityGuard:
    """
    Guard ở mức activity, học từ A-SAD_instructions.

    allowed_pairs: map (prev,next) -> score (xác suất / tần suất được xem là hợp lệ).
    counts_by_prev: đếm số lần mỗi prev xuất hiện (để debug / thống kê).
    """

    config: ActivityGuardConfig
    allowed_pairs: Dict[Pair, float]
    counts_by_prev: Dict[str, int]

    # ---------- Build từ CSV (A-SAD_instructions) ----------
    @classmethod
    def from_csv(cls, path: str, cfg: Optional[ActivityGuardConfig] = None) -> "ActivityGuard":
        """
        Khởi tạo ActivityGuard từ file CSV.

        Hỗ trợ 2 schema:

        1) Dataset đã aggregate: có cột prev, next, label (0/1).
        2) Dataset instruction A-SAD: có cột `eventually_follows`, `instruction_type`,
           và `is_valid`. Ta sẽ:
              - parse eventually_follows -> (prev,next)
              - dùng instruction_type: pos_inv -> label=1, neg_inv -> label=0
              - chỉ dùng hàng is_valid == True
              - majority vote theo (prev,next) rồi filter theo min_count, min_ratio, threshold.
        """
        if cfg is None:
            cfg = ActivityGuardConfig()

        df = pd.read_csv(path)

        allowed_pairs: Dict[Pair, float] = {}
        counts_prev: Counter[str] = Counter()
        pair_counts: Counter[Pair] = Counter()
        pos_counts: Counter[Pair] = Counter()

        # Case 1: dataset đã có prev/next/label
        if {"prev", "next", "label"}.issubset(df.columns):
            for _, row in df.iterrows():
                prev = str(row["prev"])
                nxt = str(row["next"])
                label = row["label"]

                if isinstance(label, str):
                    y = 1 if label.lower() in ("1", "true", "yes") else 0
                else:
                    y = int(bool(label))

                pair = (prev, nxt)
                pair_counts[pair] += 1
                counts_prev[prev] += 1
                if y == 1:
                    pos_counts[pair] += 1

        # Case 2: raw A-SAD_instructions (như file anh đang dùng)
        elif "eventually_follows" in df.columns:
            for _, row in df.iterrows():
                ev = row["eventually_follows"]
                if not isinstance(ev, str) or not ev:
                    continue
                try:
                    pair = ast.literal_eval(ev)
                except Exception:
                    continue
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue

                prev = str(pair[0])
                nxt = str(pair[1])

                inst_type = str(row.get("instruction_type", ""))
                # Chỉ dùng invariants: pos_inv / neg_inv
                if inst_type not in ("pos_inv", "neg_inv"):
                    # Các kiểu instruction khác (vd: câu hỏi next activity) không dùng cho guard
                    continue

                # Chỉ lấy những dòng được đánh dấu là valid (mô hình trả lời đúng)
                is_valid = bool(row.get("is_valid", True))
                if not is_valid:
                    continue

                # pos_inv → cặp (prev,next) thường xuất hiện → label=1
                # neg_inv → cặp (prev,next) hiếm/không xuất hiện → label=0
                y = 1 if inst_type == "pos_inv" else 0

                p = (prev, nxt)
                pair_counts[p] += 1
                counts_prev[prev] += 1
                if y == 1:
                    pos_counts[p] += 1
        else:
            # Schema không hỗ trợ → guard rỗng (không chặn gì)
            return cls(config=cfg, allowed_pairs={}, counts_by_prev={})

        # Aggregate + lọc theo config
        for (prev, nxt), tot in pair_counts.items():
            if tot < cfg.min_count:
                continue

            prev_total = counts_prev[prev] if counts_prev[prev] > 0 else tot
            if prev_total > 0 and (tot / prev_total) < cfg.min_ratio:
                continue

            pos = pos_counts.get((prev, nxt), 0)
            score = pos / float(tot)
            if score >= cfg.threshold:
                allowed_pairs[(prev, nxt)] = score

        return cls(
            config=cfg,
            allowed_pairs=allowed_pairs,
            counts_by_prev=dict(counts_prev),
        )

    # ---------- API runtime ----------
    def is_allowed(self, prev: Optional[str], cand: str) -> bool:
        """
        Trả về True nếu cặp (prev,cand) được xem là hợp lệ.

        Nếu prev=None (prefix rỗng) → luôn cho phép.

        Nếu cặp không có trong allowed_pairs:
            - nếu keep_unseen=True → cho phép
            - nếu keep_unseen=False → xem như bị chặn.
        """
        if prev is None:
            return True

        key: Pair = (str(prev), str(cand))
        if key in self.allowed_pairs:
            return True

        # cặp chưa từng thấy trong dữ liệu huấn luyện
        return self.config.keep_unseen
