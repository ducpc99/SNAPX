<<<<<<< HEAD
import pandas as pd
import ast
import hashlib
from typing import Union, Optional
from pathlib import Path

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

# Định nghĩa đường dẫn cố định
SNAP_CSV_PATH = "datasets/S-NAP.csv"
INSTRUCTIONS_CSV_PATH = "datasets/S-NAP_instructions.csv"

def load_snap_data(dataset_path: str = SNAP_CSV_PATH, limit: Optional[int] = None, drop_end: bool = True) -> pd.DataFrame:
    """
    Tải và xử lý dữ liệu S-NAP.csv, trả về DataFrame chuẩn hóa.
    """
    df = pd.read_csv(dataset_path)
    
    # Chuyển các cột danh sách thành set
    df["unique_activities"] = df["unique_activities"].apply(_safe_eval_list)
    df["prefix"] = df["prefix"].apply(lambda x: _safe_eval_list(x) if isinstance(x, str) else x)
    df["next"] = df["next"].astype(str)

    if drop_end:
        df = df[~df["next"].isin(["[END]", "END", "<END>", "None", None])]
    
    # Đảm bảo dữ liệu không có hoạt động "None"
    df = df[df["unique_activities"].apply(lambda x: "None" not in x and None not in x)]
    
    return df

def load_instructions_data(dataset_path: str = INSTRUCTIONS_CSV_PATH, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Tải và xử lý dữ liệu S-NAP_instructions.csv, trả về DataFrame chuẩn hóa.
    """
    df = pd.read_csv(dataset_path)
    
    # Chuyển cột `unique_activities` thành set
    df["unique_activities"] = df["unique_activities"].apply(_safe_eval_list)
    
    # Xử lý các cột cho instructions
    df["instruction"] = df["instruction"].astype(str)
    df["output"] = df["output"].astype(str)
    
    return df

def load_task_csv(file_path: str) -> pd.DataFrame:
    """
    Tải dữ liệu từ CSV và trả về dưới dạng DataFrame.
    """
    return pd.read_csv(file_path)

def load_data(dataset_path: str = SNAP_CSV_PATH, data_type: str = "S-NAP", limit: Optional[int] = None, drop_end: bool = True) -> pd.DataFrame:
    """
    Tải dữ liệu từ CSV, xử lý dữ liệu tùy theo loại dữ liệu (S-NAP hoặc S-NAP_instructions).
    """
    if data_type == "S-NAP":
        return load_snap_data(dataset_path, limit, drop_end)
    elif data_type == "S-NAP_instructions":
        return load_instructions_data(dataset_path, limit)
    else:
        raise ValueError("Invalid data type. Use 'S-NAP' or 'S-NAP_instructions'.")

=======
import pandas as pd
import ast
import hashlib
from typing import Union, Optional
from pathlib import Path

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

# Định nghĩa đường dẫn cố định
SNAP_CSV_PATH = "datasets/S-NAP.csv"
INSTRUCTIONS_CSV_PATH = "datasets/S-NAP_instructions.csv"

def load_snap_data(dataset_path: str = SNAP_CSV_PATH, limit: Optional[int] = None, drop_end: bool = True) -> pd.DataFrame:
    """
    Tải và xử lý dữ liệu S-NAP.csv, trả về DataFrame chuẩn hóa.
    """
    df = pd.read_csv(dataset_path)
    
    # Chuyển các cột danh sách thành set
    df["unique_activities"] = df["unique_activities"].apply(_safe_eval_list)
    df["prefix"] = df["prefix"].apply(lambda x: _safe_eval_list(x) if isinstance(x, str) else x)
    df["next"] = df["next"].astype(str)

    if drop_end:
        df = df[~df["next"].isin(["[END]", "END", "<END>", "None", None])]
    
    # Đảm bảo dữ liệu không có hoạt động "None"
    df = df[df["unique_activities"].apply(lambda x: "None" not in x and None not in x)]
    
    return df

def load_instructions_data(dataset_path: str = INSTRUCTIONS_CSV_PATH, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Tải và xử lý dữ liệu S-NAP_instructions.csv, trả về DataFrame chuẩn hóa.
    """
    df = pd.read_csv(dataset_path)
    
    # Chuyển cột `unique_activities` thành set
    df["unique_activities"] = df["unique_activities"].apply(_safe_eval_list)
    
    # Xử lý các cột cho instructions
    df["instruction"] = df["instruction"].astype(str)
    df["output"] = df["output"].astype(str)
    
    return df

def load_task_csv(file_path: str) -> pd.DataFrame:
    """
    Tải dữ liệu từ CSV và trả về dưới dạng DataFrame.
    """
    return pd.read_csv(file_path)

def load_data(dataset_path: str = SNAP_CSV_PATH, data_type: str = "S-NAP", limit: Optional[int] = None, drop_end: bool = True) -> pd.DataFrame:
    """
    Tải dữ liệu từ CSV, xử lý dữ liệu tùy theo loại dữ liệu (S-NAP hoặc S-NAP_instructions).
    """
    if data_type == "S-NAP":
        return load_snap_data(dataset_path, limit, drop_end)
    elif data_type == "S-NAP_instructions":
        return load_instructions_data(dataset_path, limit)
    else:
        raise ValueError("Invalid data type. Use 'S-NAP' or 'S-NAP_instructions'.")
