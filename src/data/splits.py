from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd


# ----------------------------- helpers ----------------------------- #

def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return set(cols).issubset(df.columns)

def _join_id_series(df: pd.DataFrame) -> pd.Series:
    """
    Ưu tiên ghép bằng model_id + revision_id (giống repo tham khảo).
    Nếu thiếu revision_id → fallback chỉ dùng model_id.
    Cuối cùng fallback index (ít khuyến nghị).
    """
    if _has_cols(df, ["model_id", "revision_id"]):
        return df["model_id"].astype(str) + "_" + df["revision_id"].astype(str)
    if "model_id" in df.columns:
        return df["model_id"].astype(str)
    return df.index.astype(str)

def _stratified_group_split(
    groups: pd.Series,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Chia theo NHÓM (group-level) để tránh rò rỉ giữa train/val/test.
    Mỗi group là một giá trị duy nhất của join_id (hoặc model_id).
    """
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(groups.unique()))
    rng.shuffle(uniq)

    n = len(uniq)
    n_test = int(round(test_ratio * n))
    n_val  = int(round(val_ratio * n))
    n_train = max(0, n - n_test - n_val)

    test_ids = uniq[:n_test].tolist()
    val_ids  = uniq[n_test:n_test + n_val].tolist()
    train_ids = uniq[n_test + n_val:].tolist()
    return train_ids, val_ids, test_ids


# ----------------------------- bucket reports ----------------------------- #

def _bucket_process_size(n: int) -> str:
    if n <= 1: return "≤1"
    if 2 <= n <= 4: return "2–4"
    if 5 <= n <= 7: return "5–7"
    if 8 <= n <= 10: return "8–10"
    return "≥11"

def _bucket_prefix_len(n: int) -> str:
    if n <= 2: return "≤2"
    if 3 <= n <= 5: return "3–5"
    if 6 <= n <= 8: return "6–8"
    return "≥9"

def _bucket_trace_len(n: int) -> str:
    if n <= 3: return "≤3"
    if 4 <= n <= 7: return "4–7"
    if 8 <= n <= 10: return "8–10"
    return "≥11"

def _report_counts(series: pd.Series, all_buckets: List[str]) -> Dict[str, int]:
    cnt = series.value_counts().to_dict()
    return {b: int(cnt.get(b, 0)) for b in all_buckets}

def make_bucket_report(
    task: str,
    df: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    """
    Trả về report số lượng mẫu theo bucket:
      - process_size: số activity duy nhất (từ 'unique_activities' hoặc 'activities')
      - prefix_len: (S-NAP) độ dài prefix
      - trace_len: (T-SAD) độ dài trace
    """
    report: Dict[str, Dict[str, int]] = {}

    # process size
    acts_col = "unique_activities" if "unique_activities" in df.columns else "activities" if "activities" in df.columns else None
    if acts_col is not None:
        proc_buckets = ["≤1", "2–4", "5–7", "8–10", "≥11"]
        series = df[acts_col].apply(lambda xs: _bucket_process_size(len(xs)))
        report["process_size"] = _report_counts(series, proc_buckets)

    # prefix len for S-NAP
    if task == "next_activity" and "prefix" in df.columns:
        pref_buckets = ["≤2", "3–5", "6–8", "≥9"]
        series = df["prefix"].apply(lambda xs: _bucket_prefix_len(len(xs)))
        report["prefix_len"] = _report_counts(series, pref_buckets)

    # trace len for T-SAD
    if task == "trace_anomaly" and "trace" in df.columns:
        tr_buckets = ["≤3", "4–7", "8–10", "≥11"]
        series = df["trace"].apply(lambda xs: _bucket_trace_len(len(xs)))
        report["trace_len"] = _report_counts(series, tr_buckets)

    return report


# ----------------------------- core API ----------------------------- #

def make_or_load_splits(
    df: pd.DataFrame,
    task: str,
    split_file: Union[str, Path],
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    overwrite: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    - Nếu split_file tồn tại & hợp lệ: áp vào df để trả về {'train','val','test'}.
    - Ngược lại: tạo split theo group (ưu tiên group = model_id+revision_id),
      lưu ra pickle (list các group id), rồi trả bộ dữ liệu đã chia.
    Đồng thời in báo cáo bucket theo split để hỗ trợ phân tích dữ liệu.
    """
    split_path = Path(split_file)

    # group key ưu tiên: model_id + revision_id
    join_id = _join_id_series(df)
    df = df.copy()
    df["_join_id"] = join_id

    def _subset(ids: List[str]) -> pd.DataFrame:
        s = set(map(str, ids))
        return df[df["_join_id"].isin(s)].drop(columns=["_join_id"]).reset_index(drop=True)

    # 1) Load có sẵn
    if split_path.exists() and not overwrite:
        try:
            with open(split_path, "rb") as f:
                train_ids, val_ids, test_ids = pickle.load(f)
            splits = {
                "train": _subset(train_ids),
                "val":   _subset(val_ids),
                "test":  _subset(test_ids),
            }
            _print_split_summary(task, splits, note=f"Loaded {split_path}")
            return splits
        except Exception:
            # nếu lỗi pickle → sẽ tạo mới
            pass

    # 2) Tạo mới
    train_ids, val_ids, test_ids = _stratified_group_split(
        groups=join_id, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )
    # lưu danh sách group ids (không phải index hàng)
    with open(split_path, "wb") as f:
        pickle.dump((train_ids, val_ids, test_ids), f)

    splits = {
        "train": _subset(train_ids),
        "val":   _subset(val_ids),
        "test":  _subset(test_ids),
    }
    _print_split_summary(task, splits, note=f"Created {split_path}")
    return splits


def _print_split_summary(task: str, splits: Dict[str, pd.DataFrame], note: str = "") -> None:
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"[splits] {note} | sizes → train={sizes.get('train',0)}, val={sizes.get('val',0)}, test={sizes.get('test',0)}")

    for name, part in splits.items():
        rep = make_bucket_report(task, part)
        if not rep:
            continue
        print(f"[splits] {name} bucket report:")
        for key, dist in rep.items():
            pretty = ", ".join([f"{k}:{v}" for k, v in dist.items()])
            print(f"  - {key}: {pretty}")
