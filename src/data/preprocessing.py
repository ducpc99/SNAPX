import pandas as pd
from typing import Optional, Union

# ----------------------------- Helper Functions ----------------------------- #

def _remove_invalid_activities(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ các hoạt động không hợp lệ (ví dụ: None hoặc token kết thúc)."""
    df = df[df["unique_activities"].apply(lambda x: "None" not in x and None not in x)]
    return df

def _remove_end_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Loại bỏ các token kết thúc như [END], None, <END>."""
    END_TOKENS = {"[END]", "END", "<END>", "None", None, ""}
    df = df[~df["next"].isin(END_TOKENS)]
    return df

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa các cột để đảm bảo tính nhất quán trong DataFrame."""
    # Chuyển cột `unique_activities` thành set
    df["unique_activities"] = df["unique_activities"].apply(lambda x: list(set(x)) if isinstance(x, list) else x)
    
    # Chuẩn hóa các cột `prefix` và `trace`
    df["prefix"] = df["prefix"].apply(lambda x: x if isinstance(x, list) else [])
    df["trace"] = df["trace"].apply(lambda x: x if isinstance(x, list) else [])
    
    return df

# ----------------------------- Data Preprocessing ----------------------------- #

def preprocess_snap_data(df: pd.DataFrame, drop_end: bool = True) -> pd.DataFrame:
    """
    Xử lý dữ liệu từ S-NAP.csv, trả về DataFrame đã được chuẩn hóa.
    """
    # Loại bỏ các token kết thúc nếu yêu cầu
    if drop_end:
        df = _remove_end_tokens(df)
    
    # Loại bỏ các hoạt động không hợp lệ
    df = _remove_invalid_activities(df)
    
    # Chuẩn hóa các cột để đảm bảo tính nhất quán
    df = _standardize_columns(df)
    
    return df

def preprocess_instructions_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý dữ liệu từ S-NAP_instructions.csv, chuẩn hóa cột và loại bỏ các giá trị không hợp lệ.
    """
    # Chuẩn hóa các cột cho instructions
    df["instruction"] = df["instruction"].astype(str)
    df["output"] = df["output"].astype(str)
    
    # Loại bỏ các hoạt động không hợp lệ
    df = _remove_invalid_activities(df)
    
    # Chuẩn hóa các cột
    df = _standardize_columns(df)
    
    return df

def preprocess_data(dataset_path: str, data_type: str, drop_end: bool = True) -> pd.DataFrame:
    """
    Hàm chính để xử lý dữ liệu từ CSV, tuỳ theo loại dữ liệu (S-NAP hoặc S-NAP_instructions).
    """
    if data_type == "S-NAP":
        df = pd.read_csv(dataset_path)
        return preprocess_snap_data(df, drop_end)
    elif data_type == "S-NAP_instructions":
        df = pd.read_csv(dataset_path)
        return preprocess_instructions_data(df)
    else:
        raise ValueError("Invalid data type. Use 'S-NAP' or 'S-NAP_instructions'.")
