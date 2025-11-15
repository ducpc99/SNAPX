"""
src/data/splits.py
------------------
Tạo và quản lý train/val/test splits theo NHÓM (ưu tiên 'model_id' + 'revision_id', 
fallback 'model_id') để tránh rò rỉ thông tin giữa các miền khi đánh giá.

Phương pháp luận (methodology):
- Chia theo nhóm (group-wise split): toàn bộ mẫu thuộc cùng một nhóm sẽ
  nằm trong cùng một split (train/val/test). Cách này phản ánh đúng kịch bản
  tổng quát hoá theo domain/process, thay vì trộn mẫu cùng quy trình sang nhiều splits.
- Ưu tiên định danh nhóm bằng 'model_id' + 'revision_id' (nếu có) nhằm tái lập
  thí nghiệm tương thích với bộ tham chiếu; nếu thiếu 'revision_id', dùng 'model_id'.

Chức năng chính:
- make_or_load_splits: nếu có file pickle (danh sách group-id) thì load, 
  nếu không thì tạo mới theo tỷ lệ (mặc định 0.7/0.1/0.2), lưu lại rồi trả về.
- Báo cáo bucket sau khi chia: thống kê theo quy mô quy trình (số activity),
  theo độ dài prefix (S-NAP) hoặc độ dài trace (T-SAD) để hỗ trợ phân tích thiên lệch dữ liệu.

Lưu ý:
- Pickle lưu danh sách group-id, không lưu chỉ số hàng, giúp tái lập splits
  khi dữ liệu được nạp lại theo cùng định danh nhóm.
- Tỷ lệ chia có thể điều chỉnh; việc chia là ngẫu nhiên trên không gian nhóm
  với seed cố định, đảm bảo tái lập.

Ví dụ sử dụng:
    from src.data.loader import load_task_csv
    from src.data.splits import make_or_load_splits

    df = load_task_csv("next_activity", "datasets/S_NAP.csv")
    splits = make_or_load_splits(
        df, task="next_activity",
        split_file="datasets/train_val_test.pkl",
        val_ratio=0.10, test_ratio=0.20, seed=42,
    )
"""

from __future__ import annotations
import ast
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import pickle

# ============ helpers ============

END_TOKENS = {"[END]", "END", "<END>", "None", None, ""}

def _safe_eval_list(x: Any) -> List[str]:
    """Parse list-like safely from str/list/tuple; also accept 'a,b,c'."""
    if isinstance(x, (list, tuple)):
        return [str(t) for t in x]
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    if s.startswith("[") or s.startswith("("):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(t) for t in v]
        except Exception:
            pass
    # fallback: comma-separated
    return [t.strip() for t in s.split(",") if t.strip()]

def _safe_eval_pair_list(x: Any) -> List[Tuple[str, str]]:
    """Parse list of pairs from str/list; accept ['A','B'], ('A','B'), 'A->B', 'A,B'."""
    items = _safe_eval_list(x)
    out: List[Tuple[str, str]] = []
    for it in items:
        if isinstance(it, (list, tuple)) and len(it) == 2:
            out.append((str(it[0]), str(it[1])))
        else:
            s = str(it)
            if "->" in s:
                a, b = s.split("->", 1)
                out.append((a.strip(), b.strip()))
            elif "," in s:
                a, b = s.split(",", 1)
                out.append((a.strip(), b.strip()))
    return out

def _drop_end_tokens(seq: List[str]) -> List[str]:
    return [t for t in seq if str(t).strip() not in END_TOKENS]

def _hash_activities(acts: Iterable[str]) -> str:
    s = "|".join(sorted({str(a).strip() for a in acts if str(a).strip()}))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

# ============ schema mapping ============

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "unique_activities": ["unique_activities", "activities", "set_of_activities", "activity_set"],
    "activities":        ["activities", "unique_activities", "set_of_activities", "activity_set"],
    "prefix":            ["prefix", "partial_trace", "sequence", "seq", "history"],
    "next":              ["next", "gold_answer", "next_activity", "target", "label_next"],
    "trace":             ["trace", "execution", "full_trace"],
    "label":             ["label", "ds_labels", "y", "is_anomaly", "valid"],
    "act1":              ["act1", "a1", "src", "from", "first_activity"],
    "act2":              ["act2", "a2", "dst", "to", "second_activity"],
    "pairs":             ["pairs", "dfg_pairs", "directly_follows", "direct_pairs", "eventually_follows"],
    "model_id":          ["model_id", "process_id", "domain_id"],
    "revision_id":       ["revision_id", "rev_id", "split_rev"],
}

def _pick(df: pd.DataFrame, key: str) -> Optional[str]:
    for cand in COLUMN_SYNONYMS.get(key, []):
        if cand in df.columns:
            return cand
    return None

def _ensure_model_id(df: pd.DataFrame, acts_col: str) -> pd.Series:
    col = _pick(df, "model_id")
    if col:
        return df[col].astype(str)
    return df[acts_col].apply(lambda xs: _hash_activities(xs))

# ============ core loader ============

def load_task_csv(
    task: str,
    dataset_path: Union[str, Path],
    limit: Optional[Union[int, str]] = None,
    drop_end: bool = True,
    invert_labels: bool = False,
    min_activities: int = 2,
    split_file: Optional[Union[str, Path]] = None,
    return_splits_if_available: bool = True,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Đọc CSV cho một task và chuẩn hoá schema.

    Args:
        task: 'next_activity' | 'activity_anomaly' | 'trace_anomaly' | 'dfg'
        dataset_path: đường dẫn CSV
        limit: 'all' hoặc số dòng đầu để debug
        drop_end: bỏ các token kết thúc ([END]/END/<END>/None) trong prefix/trace/next
        invert_labels: đảo nhãn anomaly (True<->False) để tương thích dữ liệu tham khảo
        min_activities: loại process có ít hơn N activity duy nhất
        split_file: đường dẫn tới pickle train/val/test (vd: datasets/train_val_test.pkl)
        return_splits_if_available: nếu có split_file hợp lệ thì trả dict {'train','val','test'}

    Returns:
        - DataFrame chuẩn hoá (mặc định)
        - HOẶC dict {'train','val','test'} nếu split_file tồn tại & có thể map
    """
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)
    if limit and limit != "all":
        df = df.head(int(limit))

    # parse sớm các cột list/pairs có thể sử dụng
    for key in ["unique_activities", "activities", "prefix", "trace", "pairs"]:
        col = _pick(df, key)
        if not col:
            continue
        if key == "pairs":
            df[col] = df[col].apply(_safe_eval_pair_list)
        else:
            df[col] = df[col].apply(_safe_eval_list)

    rev_col = _pick(df, "revision_id")

    if task == "next_activity":
        acts_col = _pick(df, "unique_activities") or _pick(df, "activities")
        pref_col = _pick(df, "prefix")
        next_col = _pick(df, "next")
        if not (acts_col and pref_col and next_col):
            raise ValueError(f"CSV thiếu cột cho S-NAP (found: {df.columns.tolist()})")

        df["_acts"] = df[acts_col].apply(lambda xs: sorted(set(map(str, xs))))
        df["_pref"] = df[pref_col].apply(lambda xs: _drop_end_tokens([str(t) for t in xs]) if drop_end else [str(t) for t in xs])
        df["_next"] = df[next_col].astype(str)
        if drop_end:
            df = df[~df["_next"].isin(END_TOKENS)]
        # loại process quá nhỏ
        df = df[df["_acts"].apply(lambda x: len(x) >= min_activities)]
        df["_model_id"] = _ensure_model_id(df, "_acts")

        out = pd.DataFrame({
            "unique_activities": df["_acts"],
            "prefix": df["_pref"],
            "next": df["_next"],
            "model_id": df["_model_id"],
        })
        if rev_col:
            out["revision_id"] = df[rev_col].astype(str)
        base_df = out.reset_index(drop=True)

    elif task == "activity_anomaly":
        acts_col = _pick(df, "activities") or _pick(df, "unique_activities")
        a1 = _pick(df, "act1")
        a2 = _pick(df, "act2")
        lab = _pick(df, "label")
        if not (acts_col and a1 and a2 and lab):
            raise ValueError("CSV thiếu cột cho A-SAD (activities, act1, act2, label)")

        df["_acts"] = df[acts_col].apply(lambda xs: sorted(set(map(str, xs))))
        df["_a1"] = df[a1].astype(str)
        df["_a2"] = df[a2].astype(str)
        # chuẩn hoá nhãn -> {0,1}
        df["_lbl"] = df[lab].apply(lambda v: 1 if str(v).strip().lower() in {"1", "true", "yes", "valid"} else 0)
        if invert_labels:
            df["_lbl"] = 1 - df["_lbl"]
        df["_model_id"] = _ensure_model_id(df, "_acts")

        out = pd.DataFrame({
            "activities": df["_acts"],
            "act1": df["_a1"],
            "act2": df["_a2"],
            "label": df["_lbl"].astype(int),
            "model_id": df["_model_id"],
        })
        if rev_col:
            out["revision_id"] = df[rev_col].astype(str)
        base_df = out.reset_index(drop=True)

    elif task == "trace_anomaly":
        acts_col = _pick(df, "activities") or _pick(df, "unique_activities")
        trc = _pick(df, "trace")
        lab = _pick(df, "label")
        if not (acts_col and trc and lab):
            raise ValueError("CSV thiếu cột cho T-SAD (activities, trace, label)")

        df["_acts"] = df[acts_col].apply(lambda xs: sorted(set(map(str, xs))))
        df["_trc"] = df[trc].apply(lambda xs: _drop_end_tokens([str(t) for t in xs]) if drop_end else [str(t) for t in xs])
        df["_lbl"] = df[lab].apply(lambda v: 1 if str(v).strip().lower() in {"1", "true", "yes", "valid"} else 0)
        if invert_labels:
            df["_lbl"] = 1 - df["_lbl"]
        df["_model_id"] = _ensure_model_id(df, "_acts")

        out = pd.DataFrame({
            "activities": df["_acts"],
            "trace": df["_trc"],
            "label": df["_lbl"].astype(int),
            "model_id": df["_model_id"],
        })
        if rev_col:
            out["revision_id"] = df[rev_col].astype(str)
        base_df = out.reset_index(drop=True)

    elif task == "dfg":
        acts_col = _pick(df, "activities") or _pick(df, "unique_activities")
        prs = _pick(df, "pairs")
        if not (acts_col and prs):
            raise ValueError("CSV thiếu cột cho DFG (activities, pairs)")

        df["_acts"] = df[acts_col].apply(lambda xs: sorted(set(map(str, xs))))
        df["_prs"] = df[prs].apply(_safe_eval_pair_list)
        df["_model_id"] = _ensure_model_id(df, "_acts")

        out = pd.DataFrame({
            "activities": df["_acts"],
            "pairs": df["_prs"],
            "model_id": df["_model_id"],
        })
        if rev_col:
            out["revision_id"] = df[rev_col].astype(str)
        base_df = out.reset_index(drop=True)

    else:
        raise ValueError(f"task không hỗ trợ: {task}")

    # ============ optional split mapping ============
    if return_splits_if_available and split_file:
        split_path = Path(split_file)
        if split_path.exists():
            try:
                with open(split_path, "rb") as f:
                    train_ids, val_ids, test_ids = pickle.load(f)
                # yêu cầu có cả model_id & revision_id để ghép chắc chắn
                if {"model_id", "revision_id"}.issubset(set(base_df.columns)):
                    base_df["_join_id"] = base_df["model_id"].astype(str) + "_" + base_df["revision_id"].astype(str)
                    def _subset(ids):
                        return base_df[base_df["_join_id"].isin(set(map(str, ids)))].drop(columns=["_join_id"]).reset_index(drop=True)
                    return {
                        "train": _subset(train_ids),
                        "val":   _subset(val_ids),
                        "test":  _subset(test_ids),
                    }
                # fallback: không có revision_id thì trả nguyên DF
            except Exception:
                pass

    return base_df
