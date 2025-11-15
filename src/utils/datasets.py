# src/utils/datasets.py
# ---------------------
from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd

from src.eval.runner import EvalSample


def parse_prefix(raw: str, sep: str = ";") -> List[str]:
    """
    Parse prefix từ chuỗi CSV.

    - Dạng mặc định: "A;B;C" → ["A", "B", "C"].
    - Nếu bạn đổi format, chỉ cần sửa lại hàm này.
    """
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        return [tok.strip() for tok in s.split(sep) if tok.strip()]
    elif isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def load_eval_samples(
    csv_path: str,
    case_id_col: str,
    prefix_col: str,
    label_col: str,
    prefix_sep: str = ";",
) -> List[EvalSample]:
    """Đọc file CSV eval & trả về list EvalSample."""
    df = pd.read_csv(csv_path)
    samples: List[EvalSample] = []
    for _, row in df.iterrows():
        case_id = str(row[case_id_col])
        prefix = parse_prefix(row[prefix_col], sep=prefix_sep)
        label = str(row[label_col])
        samples.append(EvalSample(case_id=case_id, prefix=prefix, y_true=label))
    return samples


def build_traces_from_samples(samples: List[EvalSample]) -> List[List[str]]:
    """
    Xây traces đơn giản từ EvalSample:

    pseudo-trace = prefix + [label]

    TODO: sau này bạn có log train thật thì thay thế cho chuẩn.
    """
    traces: List[List[str]] = []
    for s in samples:
        if s.prefix:
            tr = list(s.prefix) + [s.y_true]
        else:
            tr = [s.y_true]
        traces.append(tr)
    return traces
