import pandas as pd
import ast
import hashlib
from typing import Union, Optional, Dict, Any
from pathlib import Path
import pickle

# ----------------------------- helpers ----------------------------- #

def _safe_eval_list(x: Union[str, list, tuple]) -> list:
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


def _safe_eval_pair_list(x: Union[str, list, tuple]) -> list:
    """Parse list of pairs from str/list; accept ['A','B'], ('A','B'), 'A->B', 'A,B'."""
    items = _safe_eval_list(x)
    out: list = []
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


def _drop_end_tokens(seq: list) -> list:
    """Remove end tokens like [END], None, etc."""
    END_TOKENS = {"[END]", "END", "<END>", "None", None, ""}
    return [t for t in seq if str(t).strip() not in END_TOKENS]


def _pick(df: pd.DataFrame, key: str) -> Optional[str]:
    """Pick the first valid column matching the key or synonym."""
    COLUMN_SYNONYMS = {
        "unique_activities": ["unique_activities", "activities", "set_of_activities", "activity_set"],
        "activities": ["activities", "unique_activities", "set_of_activities", "activity_set"],
        "prefix": ["prefix", "partial_trace", "sequence", "seq", "history"],
        "next": ["next", "gold_answer", "next_activity", "target", "label_next"],
        "trace": ["trace", "execution", "full_trace"],
        "label": ["label", "ds_labels", "y", "is_anomaly", "valid"],
        "act1": ["act1", "a1", "src", "from", "first_activity"],
        "act2": ["act2", "a2", "dst", "to", "second_activity"],
        "pairs": ["pairs", "dfg_pairs", "directly_follows", "direct_pairs", "eventually_follows"],
        "model_id": ["model_id", "process_id", "domain_id"],
        "revision_id": ["revision_id", "rev_id", "split_rev"],
    }
    for cand in COLUMN_SYNONYMS.get(key, []):
        if cand in df.columns:
            return cand
    return None


def _ensure_model_id(df: pd.DataFrame, acts_col: str) -> pd.Series:
    """Ensure that each row has a valid model_id."""
    col = _pick(df, "model_id")
    if col:
        return df[col].astype(str)
    return df[acts_col].apply(lambda xs: _hash_activities(xs))


def _hash_activities(acts: list) -> str:
    """Generate a hash for a list of activities."""
    s = "|".join(sorted({str(a).strip() for a in acts if str(a).strip()}))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


# ----------------------------- Data Loading Functions ----------------------------- #

SNAP_CSV_PATH = "datasets/S-NAP.csv"
INSTRUCTIONS_CSV_PATH = "datasets/S-NAP_instructions.csv"


def load_snap_data(
    dataset_path: str = SNAP_CSV_PATH,
    limit: Optional[int] = None,
    drop_end: bool = True,
) -> pd.DataFrame:
    """
    Tải và xử lý dữ liệu S-NAP.csv, trả về DataFrame chuẩn hóa.
    """
    df = pd.read_csv(dataset_path)

    # list-like columns
    df["unique_activities"] = df["unique_activities"].apply(_safe_eval_list)
    df["prefix"] = df["prefix"].apply(
        lambda x: _safe_eval_list(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )
    df["next"] = df["next"].astype(str)

    if drop_end:
        df = df[~df["next"].isin(["[END]", "END", "<END>", "None", None, ""])]

    # bỏ activities rác "None"
    df = df[df["unique_activities"].apply(lambda x: "None" not in x and None not in x)]

    if limit is not None:
        df = df.head(limit)

    return df


def load_instructions_data(
    dataset_path: str = INSTRUCTIONS_CSV_PATH,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Tải và xử lý dữ liệu S-NAP_instructions.csv, trả về DataFrame chuẩn hóa.
    """
    df = pd.read_csv(dataset_path)

    df["unique_activities"] = df["unique_activities"].apply(_safe_eval_list)
    df["instruction"] = df["instruction"].astype(str)
    df["output"] = df["output"].astype(str)

    if limit is not None:
        df = df.head(limit)

    return df


def load_data(
    dataset_path: str = SNAP_CSV_PATH,
    data_type: str = "S-NAP",
    limit: Optional[int] = None,
    drop_end: bool = True,
) -> pd.DataFrame:
    """
    Wrapper đơn giản: chọn loader theo loại dữ liệu.
    """
    if data_type == "S-NAP":
        return load_snap_data(dataset_path=dataset_path, limit=limit, drop_end=drop_end)
    elif data_type == "S-NAP_instructions":
        return load_instructions_data(dataset_path=dataset_path, limit=limit)
    else:
        raise ValueError("Invalid data type. Use 'S-NAP' or 'S-NAP_instructions'.")


# ----------------------------- Kaggle helper cho run_local_eval.py ----------------------------- #

def load_task_csv(
    task: str,
    dataset_path: str,
    limit: Optional[int] = None,
    drop_end: bool = True,
    invert_labels: bool = False,      # giữ cho tương thích, hiện không dùng
    min_activities: int = 1,
    split_file: Optional[str] = None,
    return_splits_if_available: bool = False,
) -> Any:
    """
    Loader chuẩn cho run_local_eval.py.

    - Nếu `split_file` tồn tại và `return_splits_if_available=True`:
        → đọc .pkl và trả về dict {'train','val','test'} (như Kaggle dataset).
    - Ngược lại:
        → đọc CSV `dataset_path` và trả về 1 DataFrame (train=test=val trong script).
    """
    if task != "next_activity":
        raise ValueError(f"Hiện chỉ hỗ trợ task 'next_activity', nhận được: {task!r}")

    # 1) Thử load từ split_file (.pkl) nếu được yêu cầu
    if split_file and return_splits_if_available:
        try:
            p = Path(split_file)
            if p.exists():
                with p.open("rb") as f:
                    obj = pickle.load(f)
                # kỳ vọng dict chứa train/val/test
                if isinstance(obj, dict) and all(k in obj for k in ("train", "val", "test")):
                    return obj
                else:
                    print(f"[load_task_csv] split_file={split_file} không phải dict train/val/test, dùng CSV thay thế.")
        except Exception as e:
            print(f"[load_task_csv] Không đọc được split_file={split_file}: {e}. Fallback sang CSV.")

    # 2) Fallback: đọc trực tiếp CSV
    df = load_snap_data(dataset_path=dataset_path, limit=limit, drop_end=drop_end)

    # lọc theo min_activities nếu cần
    if min_activities is not None and min_activities > 0:
        df = df[df["unique_activities"].apply(lambda acts: len(acts) >= min_activities)]

    # invert_labels (chủ yếu cho anomaly task) – ở đây chỉ stub cho đủ tham số
    if invert_labels and "label" in df.columns:
        df["label"] = df["label"].apply(lambda x: 0 if int(x) == 1 else 1)

    return df
