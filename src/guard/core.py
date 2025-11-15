from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from src.guard.config import GuardConfig
from src.guard.activity_guard import ActivityGuard, ActivityGuardConfig


@dataclass
class Guard:
    """
    Guard tổng hợp cho pipeline HybridSNAP.

    Hiện tại chỉ implement ActivityGuard (guard theo cặp (prev,next)).
    Sau này có thể mở rộng thêm TraceGuard mà không thay đổi giao diện bên ngoài.
    """

    cfg: GuardConfig
    activity_guard: Optional[ActivityGuard] = None
    trace_guard: Optional[Any] = None  # placeholder nếu sau này cần guard theo trace

    # -------------- Factory từ dataset --------------
    @classmethod
    def from_datasets(cls, cfg: GuardConfig) -> "Guard":
        """
        Khởi tạo Guard từ các dataset A-SAD / T-SAD.

        - Nếu cfg.use_activity_guard=True và có đường dẫn cfg.activity_dataset:
            → build ActivityGuard.from_csv(...)

        Nếu file không tồn tại hoặc lỗi đọc, method sẽ raise Exception
        để caller (run_local_eval) bắt và fallback tắt Guard.
        """
        activity_guard: Optional[ActivityGuard] = None
        trace_guard: Optional[Any] = None  # chưa dùng

        if cfg.enabled and cfg.use_activity_guard and cfg.activity_dataset:
            path = Path(cfg.activity_dataset)
            if not path.exists():
                raise FileNotFoundError(f"ActivityGuard dataset not found: {path}")

            # map GuardConfig -> ActivityGuardConfig
            act_cfg = ActivityGuardConfig(
                min_count=getattr(cfg, "min_count", 1),
                min_ratio=getattr(cfg, "min_ratio", 0.0),
                threshold=getattr(cfg, "threshold", 0.5),
                keep_unseen=True,
            )
            activity_guard = ActivityGuard.from_csv(str(path), cfg=act_cfg)

        # (Tuỳ chọn) trace_guard có thể được build tương tự nếu sau này cần
        return cls(cfg=cfg, activity_guard=activity_guard, trace_guard=trace_guard)

    # -------------- Apply lên phân phối ứng viên --------------
    def apply(
        self,
        prefix: List[str],
        ranked: List[str],
        scores: Dict[str, float],
    ) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
        """
        Áp dụng Guard lên danh sách ứng viên.

        Parameters
        ----------
        prefix:
            Chuỗi activity trước đó (trace prefix).
        ranked:
            Danh sách label được sắp xếp theo score giảm dần (top-k).
            -> đây chỉ là list activity-id, KHÔNG gồm score.
        scores:
            Dict {label: score} tương ứng với `ranked`.

        Returns
        -------
        new_rank:
            Danh sách label sau khi điều chỉnh bằng Guard.
        new_scores:
            Dict {label: score_mới}.
        meta:
            Thông tin debug: số cặp bị chặn, mode, v.v.
        """
        meta: Dict[str, Any] = {
            "used": False,
            "mode": self.cfg.mode,
            "num_blocked": 0,
        }

        # Nếu Guard không bật hoặc không có activity_guard → giữ nguyên
        if not (self.cfg.enabled and self.activity_guard is not None and ranked):
            return ranked, scores, meta

        # Chỉ áp dụng khi prefix đủ dài
        if len(prefix) < self.cfg.min_prefix_len:
            return ranked, scores, meta

        prev = prefix[-1] if prefix else None
        new_scores: Dict[str, float] = dict(scores)
        kept_labels: List[str] = []
        blocked_labels: List[str] = []

        for label in ranked:
            original_score = scores.get(label, 0.0)
            allowed = self.activity_guard.is_allowed(prev, label)

            if allowed:
                kept_labels.append(label)
                # giữ nguyên score
                new_scores[label] = original_score
            else:
                blocked_labels.append(label)
                meta["num_blocked"] += 1

                if self.cfg.mode == "hard":
                    # loại bỏ hẳn khỏi phân phối
                    new_scores.pop(label, None)
                    continue
                elif self.cfg.mode == "soft":
                    # phạt score nhưng vẫn giữ candidate để phân tích
                    new_scores[label] = original_score * self.cfg.penalty_factor
                    kept_labels.append(label)
                else:
                    # mode lạ → behave như không dùng guard
                    kept_labels.append(label)
                    new_scores[label] = original_score

        # Re-rank theo new_scores (giảm dần)
        new_rank = sorted(kept_labels, key=lambda a: new_scores.get(a, 0.0), reverse=True)

        meta["used"] = True
        meta["blocked_labels"] = blocked_labels
        return new_rank, new_scores, meta
