# scripts/build_sem_prior.py
# --------------------------
# Script tạo Semantics-Prior JSON từ dữ liệu IT:
#   - Activity prior  : P_sem(a)     từ S-NAP_instructions.csv
#   - Pair prior      : P_sem(b|a)   từ chính các cặp (prev, next) trong S-NAP_instructions.csv
#
# Kết quả:
#   - --out-activity → JSON {activity: prob}
#   - --out-pair     → JSON {prev: {next: prob}}

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép import src.*
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.semantics.sem_prior_builder import (  # type: ignore
    build_activity_prior_from_snap,
    build_pair_prior_from_snap,
    save_prior_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Semantics-Prior maps (activity/pair) cho S-NAPX từ dữ liệu IT."
    )

    parser.add_argument(
        "--snap-path",
        type=str,
        default="datasets/S-NAP_instructions.csv",
        help="Đường dẫn tới S-NAP_instructions.csv (hoặc S-NAP.csv cùng format).",
    )
    parser.add_argument(
        "--out-activity",
        type=str,
        default="datasets/sem_prior_activity.json",
        help="File JSON xuất prior theo activity.",
    )
    parser.add_argument(
        "--out-pair",
        type=str,
        default="datasets/sem_prior_pairs.json",
        help="File JSON xuất prior theo cặp (prev, next).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Giới hạn số dòng để debug (None = dùng toàn bộ).",
    )
    parser.add_argument(
        "--min-activities",
        type=int,
        default=2,
        help="Loại process có ít hơn N activity duy nhất.",
    )
    parser.add_argument(
        "--min-pair-count",
        type=int,
        default=1,
        help="Chỉ giữ cặp (prev, next) xuất hiện ít nhất N lần trong IT dataset.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("🔧 Xây Semantics-Prior từ dữ liệu IT (S-NAP_instructions)...\n")

    # 1) Activity prior from S-NAP / S-NAP_instructions
    print(f"➡️  Đọc S-NAP/S-NAP_instructions từ: {args.snap_path}")
    act_prior = build_activity_prior_from_snap(
        dataset_path=args.snap_path,
        limit=args.limit,
        drop_end=True,
        min_activities=args.min_activities,
    )
    print(f"   Số activity trong prior: {len(act_prior)}")
    save_prior_map(act_prior, args.out_activity)
    print(f"✅ Đã lưu activity prior → {args.out_activity}\n")

    # 2) Pair prior from S-NAP / S-NAP_instructions
    print(f"➡️  Xây pair prior P_sem(next | prev) từ: {args.snap_path}")
    pair_prior = build_pair_prior_from_snap(
        dataset_path=args.snap_path,
        limit=args.limit,
        drop_end=True,
        min_activities=args.min_activities,
        min_pair_count=args.min_pair_count,
    )
    num_pairs = sum(len(v) for v in pair_prior.values())
    print(f"   Số cặp (prev,next) trong prior: {num_pairs}")
    save_prior_map(pair_prior, args.out_pair)
    print(f"✅ Đã lưu pair prior → {args.out_pair}\n")

    print("🎉 Hoàn tất build Semantics-Prior.")


if __name__ == "__main__":
    main()
