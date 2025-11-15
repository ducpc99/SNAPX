# src/semantics/sem_prior_builder.py
# ----------------------------------
# Xây Semantics-Prior cho S-NAPX:
#   - Activity prior:   P_sem(a)     từ S-NAP (next_activity).
#   - Pair prior:       P_sem(b|a)   ưu tiên lấy từ S-PMD.csv (cột 'dfg').
#
# Các hàm public:
#   - build_activity_prior_from_snap(...)
#   - build_pair_prior_from_dfg(...)   # tên giữ nguyên cho tương thích script
#   - save_prior_map(...)

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import ast
import json

import pandas as pd

from src.data.loader import load_task_csv


# ------------------------------------------------
# Activity prior: P_sem(a) từ S-NAP (next_activity)
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
    - Đếm tần suất nhãn 'next'.
    - Chuẩn hoá thành xác suất P_sem(a).

    Args:
        dataset_path: đường dẫn tới S-NAP_instructions.csv (hoặc S_NAP.csv chuẩn hoá).
        limit: nếu không None thì chỉ lấy N dòng đầu (debug).
        drop_end: bỏ token kết thúc ([END]/END/...) nếu còn sót.
        min_activities: loại process có ít hơn N activity duy nhất.

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

    # cột chuẩn hoá của loader là 'next'
    next_series = df["next"].astype(str)
    cnt = Counter(next_series.tolist())
    total = float(sum(cnt.values())) or 1.0

    prior: Dict[str, float] = {act: float(freq) / total for act, freq in cnt.items()}
    return prior


# ------------------------------------------------
# Pair prior: P_sem(b|a) – ưu tiên S-PMD.csv
# ------------------------------------------------
def _parse_dfg_value(x: Any) -> list:
    """
    Parse một ô 'dfg' thành list các cạnh.

    Giao diện mong đợi:
        - str: biểu diễn list/tuple Python, ví dụ: "[('A','B'), ('B','C')]"
        - list/tuple: đã là list cạnh.
    """
    if isinstance(x, (list, tuple)):
        return list(x)

    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s:
        return []

    # cố gắng literal_eval trước
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return list(v)
    except Exception:
        pass

    # fallback: không parse được thì trả list rỗng
    return []


def build_pair_prior_from_dfg(
    dataset_path: str,
    limit: Optional[int] = None,
    drop_end: bool = True,
    min_activities: int = 2,
) -> Dict[str, Dict[str, float]]:
    """
    Xây prior theo CẶP (prev, next) – tên hàm giữ là '..._from_dfg' để
    tương thích với script cũ, nhưng ưu tiên:

        1) Nếu file có cột 'dfg' (S-PMD.csv):
            - 'dfg' là list các cạnh (a,b) ⇒ dùng trực tiếp.

        2) Nếu KHÔNG có 'dfg':
            - Fallback 1: load_task_csv(task="dfg") → cột 'pairs'.
            - Fallback 2: load_task_csv(task="activity_anomaly") → act1, act2, label=1.

    Kết quả:
        prior[a][b] = P_sem(b | a)
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {dataset_path}")

    pair_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # --------------------------------------------
    # 1) Thử đọc trực tiếp bằng pandas, ưu tiên 'dfg'
    # --------------------------------------------
    use_pmd = False
    df: Optional[pd.DataFrame] = None

    try:
        df = pd.read_csv(path)
        if "dfg" in df.columns:
            use_pmd = True
    except Exception:
        df = None

    if use_pmd and df is not None:
        # Case S-PMD.csv: có cột 'dfg'
        if limit and limit != "all":
            df = df.head(int(limit))

        df["_dfg_list"] = df["dfg"].apply(_parse_dfg_value)

        for _, row in df.iterrows():
            for edge in row["_dfg_list"]:
                # edge có thể là tuple/list ('A','B') hoặc string "A->B" / "A,B"
                a: Optional[str] = None
                b: Optional[str] = None

                if isinstance(edge, (list, tuple)) and len(edge) == 2:
                    a, b = edge[0], edge[1]
                else:
                    s = str(edge)
                    if "->" in s:
                        a, b = s.split("->", 1)
                    elif "," in s:
                        a, b = s.split(",", 1)
                    else:
                        continue

                a = str(a).strip()
                b = str(b).strip()
                if not a or not b:
                    continue

                pair_counts[a][b] += 1

    else:
        # --------------------------------------------
        # 2) Fallback: dùng loader chuẩn hoá
        # --------------------------------------------
        try:
            # 2a) Thử dạng DFG aggregate (task="dfg")
            df = load_task_csv(
                task="dfg",
                dataset_path=dataset_path,
                limit=limit,
                drop_end=drop_end,
                invert_labels=False,
                min_activities=min_activities,
                split_file=None,
                return_splits_if_available=False,
            )
            for _, row in df.iterrows():
                for a, b in row["pairs"]:
                    a = str(a).strip()
                    b = str(b).strip()
                    if not a or not b:
                        continue
                    pair_counts[a][b] += 1
        except Exception:
            # 2b) Thử dạng edge-level kiểu A-SAD (task="activity_anomaly")
            df = load_task_csv(
                task="activity_anomaly",
                dataset_path=dataset_path,
                limit=limit,
                drop_end=drop_end,
                invert_labels=False,
                min_activities=min_activities,
                split_file=None,
                return_splits_if_available=False,
            )
            for _, row in df.iterrows():
                # chỉ lấy cạnh hợp lệ (label=1)
                try:
                    lbl = int(row["label"])
                except Exception:
                    lbl = 0
                if lbl != 1:
                    continue
                a = str(row["act1"]).strip()
                b = str(row["act2"]).strip()
                if not a or not b:
                    continue
                pair_counts[a][b] += 1

    # --------------------------------------------
    # Chuẩn hoá: P_sem(b | a) cho từng nguồn a
    # --------------------------------------------
    prior: Dict[str, Dict[str, float]] = {}
    for a, dests in pair_counts.items():
        total = float(sum(dests.values())) or 1.0
        prior[a] = {b: float(c) / total for b, c in dests.items()}

    return prior


# ------------------------------------------------
# Helper: lưu prior map
# ------------------------------------------------
def save_prior_map(prior_map: Any, output_path: str) -> None:
    """
    Lưu prior_map (dict) thành file JSON.

    Args:
        prior_map: dict (có thể lồng nhau) chứa P_sem.
        output_path: đường dẫn file JSON đầu ra.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior_map, f, ensure_ascii=False, indent=2)
