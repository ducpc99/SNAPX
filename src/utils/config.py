# src/utils/config.py
# -------------------
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

import yaml


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Gộp 2 dict theo kiểu: override đè lên base (đệ quy với dict con).
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dict(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(paths: List[str]) -> Dict[str, Any]:
    """
    Đọc & merge nhiều file YAML theo thứ tự.

    Ví dụ:
        cfg = load_config(["config/base.yaml", "config/local_eval.yaml"])

    File phía sau sẽ override file phía trước.
    """
    cfg: Dict[str, Any] = {}
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file config: {path}")
        with open(path, "r", encoding="utf-8") as f:
            part = yaml.safe_load(f) or {}
        cfg = _merge_dict(cfg, part)
    return cfg
