# scripts/build_sem_prior.py
# --------------------------
# Script tạo Semantics-Prior JSON từ dữ liệu IT:
# - Activity prior  : từ S-NAP_instructions.csv
# - (tuỳ chọn) Pair prior: từ một CSV kiểu DFG/S-DFD

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép import src.*
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.semantics.sem_prior_builder import (  # type: ignore
    build_activity_prior_from_snap,
    build_pair_prior_from_dfg,
    save_prior_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Semantics-Prior maps (activity/pair) for S-NAPX.")

    parser.add_argument(
        "--snap-path",
        type=str,
        default="datasets/S-NAP_instructions.csv",
        help="Đường dẫn tới S-NAP_instructions.csv",
    )
    parser.add_argument(
        "--dfg-path",
        type=str,
        default=None,
        help="(Tuỳ chọn) CSV kiểu DFG/S-DFD để build pair prior.",
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
        help="File JSON xuất prior theo pair (prev,next). Chỉ dùng nếu --dfg-path được cung cấp.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("🔧 Xây Semantics-Prior từ dữ liệu IT...\n")

    # 1) Activity prior from S-NAP
    print(f"➡️  Đọc S-NAP từ: {args.snap_path}")
    act_prior = build_activity_prior_from_snap(
        dataset_path=args.snap_path,
        limit=args.limit,
        drop_end=True,
        min_activities=args.min_activities,
    )
    print(f"   Số activity trong prior: {len(act_prior)}")
    save_prior_map(act_prior, args.out_activity)
    print(f"✅ Đã lưu activity prior → {args.out_activity}\n")

    # 2) Pair prior from DFG/S-DFD (optional)
    if args.dfg_path:
        print(f"➡️  Đọc DFG/S-DFD từ: {args.dfg_path}")
        pair_prior = build_pair_prior_from_dfg(
            dataset_path=args.dfg_path,
            limit=args.limit,
            drop_end=True,
            min_activities=args.min_activities,
        )
        num_pairs = sum(len(v) for v in pair_prior.values())
        print(f"   Số cặp (prev,next) trong prior: {num_pairs}")
        save_prior_map(pair_prior, args.out_pair)
        print(f"✅ Đã lưu pair prior → {args.out_pair}\n")
    else:
        print("ℹ️  Không cung cấp --dfg-path, bỏ qua pair prior.\n")

    print("🎉 Hoàn tất build Semantics-Prior.")


if __name__ == "__main__":
    main()
