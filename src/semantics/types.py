# src/semantics/types.py
# -----------------------
"""
Các kiểu dữ liệu chung (dataclass) dùng trong Semantics Layer.

- PromptTemplate:
    Đại diện cho 1 template prompt (IT, predict, explain...).
"""

from dataclasses import dataclass
from typing import Literal


PromptTask = Literal[
    "trace_anomaly",
    "activity_anomaly",
    "next_activity",
    "dfg",
    "process_tree",
    "snap_predict",   # next-activity từ top-k candidate (đặc thù S-NAP)
    "snap_explain",   # giải thích prediction (đặc thù S-NAP)
]


@dataclass
class PromptTemplate:
    """
    Mô tả 1 prompt template.

    Attributes
    ----------
    id:
        Mã định danh duy nhất (ví dụ: "it_next_v1", "snap_predict_strict_v1").
    task:
        Loại nhiệm vụ (next_activity, dfg, snap_predict, snap_explain...).
    variant_group:
        Nhóm biến thể:
        - "it_general": prompt dùng cho IT process mining tổng quát.
        - "snap_predict": prompt dùng cho infer chọn nhãn từ top-k.
        - "snap_explain": prompt dùng cho sinh giải thích.
        - "guard": (tuỳ chọn) prompt hỗ trợ anomaly-guard.
    description:
        Mô tả ngắn gọn mục đích / ngữ cảnh dùng template này.
    template:
        Chuỗi prompt có placeholder kiểu {activities}, {prefix}, {candidates}, ...
    """

    id: str
    task: PromptTask
    variant_group: str
    description: str
    template: str


__all__ = ["PromptTemplate", "PromptTask"]
