import pandas as pd
import hashlib
from typing import List, Dict, Any

# ----------------------------- Utility Functions ----------------------------- #

def remove_duplicates(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Loại bỏ các dòng trùng lặp dựa trên các cột chỉ định.
    """
    return df.drop_duplicates(subset=columns)

def _hash_activities(acts: List[str]) -> str:
    """
    Tạo hash từ danh sách các hoạt động (activities) để tạo ID duy nhất.
    """
    s = "|".join(sorted({str(a).strip() for a in acts if str(a).strip()}))
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def check_required_columns(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Kiểm tra xem DataFrame có chứa tất cả các cột yêu cầu hay không.
    """
    return all(col in df.columns for col in required_cols)

def get_column_values(df: pd.DataFrame, column: str) -> List[Any]:
    """
    Lấy tất cả giá trị từ một cột cụ thể trong DataFrame dưới dạng danh sách.
    """
    if column in df.columns:
        return df[column].tolist()
    else:
        raise ValueError(f"Cột '{column}' không tồn tại trong DataFrame.")

def split_data(df: pd.DataFrame, ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu thành 2 phần theo tỷ lệ cho trước (train/val).
    """
    train_size = int(len(df) * ratio)
    train_df = df[:train_size]
    val_df = df[train_size:]
    return train_df, val_df

def save_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Lưu DataFrame vào file CSV.
    """
    df.to_csv(file_path, index=False)

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Đọc dữ liệu từ file CSV và trả về DataFrame.
    """
    return pd.read_csv(file_path)

def calculate_stats(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Tính toán các thống kê cơ bản cho một cột trong DataFrame.
    """
    stats = {
        "mean": df[column].mean(),
        "median": df[column].median(),
        "min": df[column].min(),
        "max": df[column].max(),
        "std": df[column].std(),
    }
    return stats
