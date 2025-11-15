# src/graph_reasoning/semantics_prior.py
# --------------------------------------
# Semantics-Prior V2 (pair-based, smart mix).
#
# Mục tiêu:
#   - Không làm "gãy" Sequence/Graph khi prior không hữu ích.
#   - Chỉ "bơm" tri thức ngữ nghĩa nhẹ nhàng vào phân phối xác suất.
#
# Sử dụng:
#   cfg = SemanticsPriorConfig(enabled=True, mode="pair", lambda_mix=0.2, ...)
#   prior = SemanticsPrior(cfg=cfg, prior_map=loaded_json)
#   new_ranked, new_scores, meta = prior.apply(prefix, ranked, scores)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    """Softmax ổn định trên dict {cand: score}."""
    if not scores:
        return {}
    max_s = max(scores.values())
    exps: Dict[str, float] = {}
    for k, v in scores.items():
        exps[k] = math.exp(float(v) - max_s)
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class SemanticsPriorConfig:
    """
    Cấu hình cho SemanticsPrior.

    Thuộc tính:
        enabled       : bật / tắt prior.
        mode          : 'activity' | 'pair' | 'hybrid'.
        default_weight: trọng số mặc định khi không có prior cho candidate.
        lambda_mix    : tỉ lệ pha P_prior vào P_final (0..1).
        floor, ceil   : kẹp biên cho weight trước khi chuẩn hoá.
    """

    enabled: bool = False
    mode: str = "activity"
    default_weight: float = 1.0
    lambda_mix: float = 0.2
    floor: float = 0.7
    ceil: float = 1.3


class SemanticsPrior:
    """
    Semantics-Prior V2.

    prior_map:
        - mode='activity':
            prior_map = {activity: weight}
        - mode='pair':
            prior_map = {prev: {next: weight}}
        - mode='hybrid':
            kết hợp pair + activity.

    Ngoài prior_map, có thể gắn provider động:
        fn(prefix, candidates) -> {cand: weight}
    """

    def __init__(
        self,
        cfg: SemanticsPriorConfig,
        prior_map: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cfg = cfg
        self.enable: bool = bool(cfg.enabled)
        self.mode: str = (cfg.mode or "activity").lower()
        self.default_weight: float = float(cfg.default_weight)
        self.lambda_mix: float = max(0.0, min(1.0, float(cfg.lambda_mix)))
        self.floor: float = float(cfg.floor)
        # đảm bảo ceil >= floor
        self.ceil: float = float(cfg.ceil if cfg.ceil >= cfg.floor else cfg.floor)

        self._prior_map: Dict[str, Any] = dict(prior_map or {})

        # Tự nhận diện prior dạng pair?
        self._is_pair_prior: bool = False
        try:
            any_val = next(iter(self._prior_map.values()))
            self._is_pair_prior = isinstance(any_val, dict)
        except StopIteration:
            self._is_pair_prior = False

        # Provider động (tuỳ chọn)
        self._provider: Optional[
            Callable[[Sequence[str], List[str]], Dict[str, float]]
        ] = None

    # ------------------------------------------------
    # Public API
    # ------------------------------------------------
    def set_provider(
        self,
        fn: Callable[[Sequence[str], List[str]], Dict[str, float]],
    ) -> None:
        """
        Gắn provider động:
            fn(prefix, ranked) -> {candidate: weight}

        Hữu ích nếu muốn dùng LLM/heuristic để gợi ý prior theo context.
        """
        self._provider = fn

    # ------------------------------------------------
    # Internal helpers: activity & pair prior
    # ------------------------------------------------
    def _activity_prior_probs(self, ranked: List[str]) -> Dict[str, float]:
        """
        Tính P_prior(a) từ prior_map (dạng activity):
            weight_raw = clamp(prior_map.get(a, default_weight), floor, ceil)
            P_prior(a) ~ weight_raw được chuẩn hoá.
        """
        if not ranked:
            return {}

        raw: Dict[str, float] = {}
        for cand in ranked:
            w = self._prior_map.get(cand, self.default_weight)
            w = _clamp(float(w), self.floor, self.ceil)
            raw[cand] = w

        z = sum(raw.values()) or 1.0
        return {c: v / z for c, v in raw.items()}

    def _pair_prior_probs(
        self,
        prefix: Sequence[str],
        ranked: List[str],
    ) -> Dict[str, float]:
        """
        Tính P_prior(b | a) với a = last(prefix):
            - Nếu prior_map kiểu pair {a: {b: weight}}:
                dùng tập dests[a].
            - Nếu không có, fallback uniform theo default_weight.
        """
        if not ranked:
            return {}

        raw: Dict[str, float] = {}
        prev: Optional[str] = prefix[-1] if prefix else None

        if (
            self._is_pair_prior
            and prev is not None
            and prev in self._prior_map
        ):
            dests: Dict[str, float] = self._prior_map.get(prev, {}) or {}
            for cand in ranked:
                w = dests.get(cand, self.default_weight)
                raw[cand] = _clamp(float(w), self.floor, self.ceil)
        else:
            # fallback: không có prior cho cặp cụ thể, dùng default_weight
            for cand in ranked:
                raw[cand] = _clamp(self.default_weight, self.floor, self.ceil)

        z = sum(raw.values()) or 1.0
        return {c: v / z for c, v in raw.items()}

    # ------------------------------------------------
    # Core: apply prior vào distribution hiện tại
    # ------------------------------------------------
    def apply(
        self,
        prefix: Sequence[str],
        ranked: List[str],
        scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], Dict[str, float], Dict[str, Any]]:
        """
        Áp dụng Semantics-Prior lên danh sách ứng viên.

        Args:
            prefix : chuỗi đã quan sát (list activity).
            ranked : danh sách ứng viên theo thứ tự hiện tại.
            scores : dict {cand: score} từ Sequence/Graph (log-score hoặc score tuỳ ý).

        Returns:
            new_ranked: danh sách ứng viên sau khi áp prior.
            new_scores: dict {cand: P_final(cand)} sau khi trộn.
            meta      : thông tin debug (mode, source, prior_probs,...)
        """
        if not self.enable or not ranked:
            # Không dùng prior → giữ nguyên
            return ranked, dict(scores or {}), {"used": False}

        # Chuẩn hoá scores hiện tại thành phân phối P_seq
        base_scores: Dict[str, float] = {
            c: float((scores or {}).get(c, 0.0)) for c in ranked
        }
        p_seq = _softmax(base_scores)

        # 1) Nếu có provider động thì ưu tiên dùng
        source = "none"
        prior_probs: Dict[str, float] = {}
        if self._provider is not None:
            try:
                dyn = self._provider(prefix, ranked) or {}
            except Exception:
                dyn = {}
            if dyn:
                raw_dyn = {
                    c: float(dyn.get(c, self.default_weight)) for c in ranked
                }
                z = sum(raw_dyn.values()) or 1.0
                prior_probs = {c: v / z for c, v in raw_dyn.items()}
                source = "dynamic"

        # 2) Nếu không có provider / dyn rỗng → dùng prior_map theo mode
        if not prior_probs:
            mode = self.mode
            if mode == "pair":
                prior_probs = self._pair_prior_probs(prefix, ranked)
                source = "pair_map" if self._is_pair_prior else "pair_uniform"
            elif mode == "hybrid":
                pair_p = self._pair_prior_probs(prefix, ranked)
                act_p = self._activity_prior_probs(ranked)
                prior_probs = {
                    c: 0.5 * pair_p.get(c, 0.0) + 0.5 * act_p.get(c, 0.0)
                    for c in ranked
                }
                source = "hybrid"
            else:
                # mặc định: activity
                prior_probs = self._activity_prior_probs(ranked)
                source = "activity"

        # 3) Trộn: P_final = (1-λ)*P_seq + λ*P_prior
        lam = self.lambda_mix
        new_scores: Dict[str, float] = {}
        for cand in ranked:
            ps = p_seq.get(cand, 0.0)
            pp = prior_probs.get(cand, 0.0)
            pf = (1.0 - lam) * ps + lam * pp
            new_scores[cand] = pf

        # Re-rank theo P_final giảm dần
        new_ranked = sorted(ranked, key=lambda c: -new_scores.get(c, 0.0))

        meta: Dict[str, Any] = {
            "used": True,
            "mode": self.mode,
            "source": source,
            "lambda_mix": lam,
            "prior_probs": {c: prior_probs.get(c, 0.0) for c in ranked},
        }
        return new_ranked, new_scores, meta


__all__ = ["SemanticsPrior", "SemanticsPriorConfig"]
