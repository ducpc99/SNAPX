# tests/conftest.py
# -----------------
# Đảm bảo có thể import được "src.*" trong các file test.

import sys
from pathlib import Path

# Thư mục gốc của project (chứa thư mục "src", "scripts", "config", "tests", ...)
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
