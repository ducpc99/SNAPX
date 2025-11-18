# src/semantics/sem_prior_builder.py
# ----------------------------------
# Xây Semantics-Prior cho S-NAPX từ dữ liệu S-NAP / S-NAP_instructions:
#   - Activity prior:   P_sem(a)       từ cột `next` (next_activity).
#   - Pair prior:       P_sem(b | a)   từ cặp (prev, next) với prev là activity cuối của prefix.
#
# Ý tưởng:
#   - Dùng chính bộ dữ liệu đã được Instruction-Tuning (S-NAP_instructions.csv)
#     để học ra phân phối "ưu tiên ngữ nghĩa" của các activity / cặp activity.
#   - Sau đó SemanticsPrior sẽ pha nhẹ prior này với phân phối Sequence/Graph:
#       P_final = (1 - λ) * P_seq + λ * P_prior
#
# Các hàm public:
#   - build_activity_prior_from_snap(...)
#   - build_pair_prior_from_snap(...)
#   - save_prior_map(...)

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import json
import pandas as pd  # giữ lại nếu sau này cần debug trực tiếp

from src.data.loader import load_task_csv


# ------------------------------------------------
# Activity prior: P_sem(a) từ S-NAP / S-NAP_instructions
# ------------------------------------------------
def build_activity_prior_from_snap(
    dataset_path: str,
    limit: Optional[int] = None,
    drop_end: bool = True,
    min_activities: int = 2,
) -> Dict[str, float]:
    """
    Xây prior theo ACTIVITY từ S-NAP (hoặc S-NAP_instructions):

    - Đọc CSV bằng loader với task="next_activity".
    - Đếm tần suất nhãn `next`.
    - Chuẩn hoá thành xác suất P_sem(a).

    Args:
        dataset_path:
            Đường dẫn tới S-NAP_instructions.csv
            (hoặc S-NAP.csv đã chuẩn hoá cùng format).
        limit:
            Nếu không None thì chỉ lấy N dòng đầu (debug).
        drop_end:
            Bỏ token kết thúc ([END]/END/...) nếu còn sót trong prefix.
        min_activities:
            Loại các process có ít hơn N activity duy nhất.

    Returns:
        dict {activity: probability}
    """
    df = load_task_csv(
        task="next_activity",
        dataset_path=dataset_path,
        limit=limit,
        drop_end=drop_end,
        invert_labels=False,
        min_activities=min_activities,
        split_file=None,
        return_splits_if_available=False,
    )

    if df.empty:
        return {}

    cnt: Counter[str] = Counter()
    for _, row in df.iterrows():
        label = str(row["next"]).strip()
        if not label:
            continue
        cnt[label] += 1

    total = float(sum(cnt.values())) or 1.0
    prior: Dict[str, float] = {act: float(freq) / total for act, freq in cnt.items()}
    return prior


# ------------------------------------------------
# Pair prior: P_sem(b | a) từ S-NAP / S-NAP_instructions
# ------------------------------------------------
def build_pair_prior_from_snap(
    dataset_path: str,
    limit: Optional[int] = None,
    drop_end: bool = True,
    min_activities: int = 2,
    min_pair_count: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Xây prior theo CẶP (prev, next) trực tiếp từ dữ liệu S-NAP / S-NAP_instructions.

    Ý tưởng:
        - Với mỗi dòng:
            prefix = [a1, a2, ..., ak]
            next   = b
          ta coi prev = ak, rồi đếm số lần xuất hiện (prev, b).
        - Sau đó chuẩn hoá theo từng prev:
              P_sem(b | prev) = count(prev, b) / sum_b' count(prev, b').

    Args:
        dataset_path:
            Đường dẫn tới S-NAP_instructions.csv
            (hoặc S-NAP.csv chuẩn hoá).
        limit:
            Nếu không None thì chỉ lấy N dòng đầu (debug).
        drop_end:
            Bỏ token kết thúc trong prefix (nếu có).
        min_activities:
            Loại process có ít hơn N activity duy nhất (dựa trên unique_activities).
        min_pair_count:
            Chỉ giữ các cặp (prev, next) có tần suất >= ngưỡng này
            để tránh nhiễu từ các cặp xuất hiện quá hiếm.

    Returns:
        prior[prev][next] = P_sem(next | prev)
    """
    df = load_task_csv(
        task="next_activity",
        dataset_path=dataset_path,
        limit=limit,
        drop_end=drop_end,
        invert_labels=False,
        min_activities=min_activities,
        split_file=None,
        return_splits_if_available=False,
    )

    if df.empty:
        return {}

    pair_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        prefix = row.get("prefix", [])
        nxt = str(row.get("next", "")).strip()

        if not isinstance(prefix, (list, tuple)):
            # Đề phòng trường hợp prefix vẫn còn ở dạng string
            # (load_task_csv chuẩn rồi thì không rơi vào nhánh này).
            try:
                if isinstance(prefix, str):
                    prefix = json.loads(prefix)
                else:
                    prefix = []
            except Exception:
                prefix = []

        prefix = [str(a).strip() for a in prefix if str(a).strip()]
        if not prefix or not nxt:
            continue

        prev = prefix[-1]
        pair_counts[prev][nxt] += 1

    prior: Dict[str, Dict[str, float]] = {}
    for prev, dests in pair_counts.items():
        # Lọc theo min_pair_count
        filtered = {b: c for b, c in dests.items() if c >= min_pair_count}
        if not filtered:
            continue

        total = float(sum(filtered.values())) or 1.0
        prior[prev] = {b: float(c) / total for b, c in filtered.items()}

    return prior


# ------------------------------------------------
# Helper: lưu prior map
# ------------------------------------------------
def save_prior_map(prior_map: Any, output_path: str) -> None:
    """
    Lưu prior_map (dict) thành file JSON.

    Args:
        prior_map:
            dict (có thể lồng nhau) chứa P_sem.
        output_path:
            Đường dẫn file JSON đầu ra.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior_map, f, ensure_ascii=False, indent=2)


__all__ = [
    "build_activity_prior_from_snap",
    "build_pair_prior_from_snap",
    "save_prior_map",
]
