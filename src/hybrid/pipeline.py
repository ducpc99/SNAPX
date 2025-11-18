# src/hybrid/pipeline.py
# ----------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.sequence.ensemble import SequenceEnsemble
from src.graph_reasoning.reasoner import GraphReasoner
from src.graph_reasoning.semantics_prior import SemanticsPrior
from src.semantics.inference import SemanticsLLM
from src.semantics.prompt_pool import PromptPool
from src.semantics.explanations import generate_explanation
from src.guard.core import Guard  # Guard V2 (apply trên phân phối)

# Kiểu danh sách ứng viên: [(label, score), ...]
RankedCandidates = List[Tuple[str, float]]


@dataclass
class HybridConfig:
    """
    Cấu hình cho HybridSNAP.

    Attributes
    ----------
    top_k:
        Số lượng candidate tối đa lấy từ SequenceEnsemble.
    enable_graph_reasoner:
        Bật / tắt GraphReasoner (DFG).
    enable_semantics_prior:
        Bật / tắt SemanticsPrior.
    enable_guard:
        Bật / tắt Guard (footprint/anomaly soft penalty).
    enable_explanation:
        Nếu True và có LLM + PromptPool → sinh giải thích bằng generate_explanation().
    """

    top_k: int = 5
    enable_graph_reasoner: bool = True
    enable_semantics_prior: bool = True
    enable_guard: bool = True
    enable_explanation: bool = False  # bật để sinh giải thích


class HybridSNAP:
    """
    Pipeline chính cho S-NAP Hybrid:

    1) SequenceEnsemble → top-k candidate + score.
    2) GraphReasoner (DFG) → re-rank soft/hard mask.
    3) SemanticsPrior → inject prior ngữ nghĩa từ JSON.
    4) Guard → penalty các candidate bất thường.
    5) SemanticsLLM → chọn nhãn cuối cùng từ danh sách candidate.
    6) (tuỳ chọn) generate_explanation → sinh giải thích.
    """

    def __init__(
        self,
        cfg: HybridConfig,
        seq_model: SequenceEnsemble,
        graph_reasoner: Optional[GraphReasoner] = None,
        sem_prior: Optional[SemanticsPrior] = None,
        semantics_llm: Optional[SemanticsLLM] = None,
        guard: Optional[Guard] = None,
        prompt_pool: Optional[PromptPool] = None,
        activities_vocab: Optional[Sequence[str]] = None,
    ) -> None:
        if cfg.top_k <= 0:
            raise ValueError("cfg.top_k phải > 0")

        self.cfg = cfg
        self.seq_model = seq_model
        self.graph_reasoner = graph_reasoner
        self.sem_prior = sem_prior
        self.semantics_llm = semantics_llm
        self.guard = guard
        self.prompt_pool = prompt_pool
        # vocab toàn bộ activity (thường lấy từ TRAIN)
        self.activities_vocab = list(activities_vocab) if activities_vocab is not None else None

    # ------------------------------------------------------------------
    # 1) Sequence
    # ------------------------------------------------------------------
    def _sequence_rank(
        self,
        prefix: Sequence[str],
    ) -> Tuple[RankedCandidates, Dict[str, Any]]:
        """
        Gọi SequenceEnsemble để lấy top-k ứng viên.

        Returns
        -------
        ranked:
            List[(activity, score)] sắp xếp giảm dần theo score.
        meta:
            Thông tin debug: scores raw, nguồn model,...
        """
        scores = self.seq_model.propose_candidates(prefix, k=self.cfg.top_k)
        ranked: RankedCandidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        meta = {
            "used": True,
            "scores": scores,
            "source": "sequence_ensemble",
        }
        return ranked, meta

    # ------------------------------------------------------------------
    # 2) GraphReasoner
    # ------------------------------------------------------------------
    def _apply_graph_reasoning(
        self,
        prefix: Sequence[str],
        ranked: RankedCandidates,
    ) -> Tuple[RankedCandidates, Dict[str, float], Dict[str, Any]]:
        """
        Áp dụng GraphReasoner nếu được bật.

        Returns
        -------
        new_ranked:
            Ứng viên sau khi đã re-rank bằng DFG.
        new_scores:
            Dict[label, score] sau re-rank.
        meta:
            Thông tin diễn giải.
        """
        if not (self.cfg.enable_graph_reasoner and self.graph_reasoner is not None and ranked):
            return ranked, {a: s for a, s in ranked}, {"used": False}

        prev = prefix[-1] if prefix else None
        ranking = [a for a, _ in ranked]
        scores_map = {a: s for a, s in ranked}

        new_rank, new_scores, meta = self.graph_reasoner.rerank_candidates(
            prev=prev,
            ranked=ranking,
            scores=scores_map,
        )
        new_ranked: RankedCandidates = [(a, new_scores.get(a, 0.0)) for a in new_rank]
        meta = dict(meta or {})
        meta.setdefault("used", True)
        meta.setdefault("impl", "GraphReasoner")
        return new_ranked, new_scores, meta

    # ------------------------------------------------------------------
    # 3) Semantics-Prior
    # ------------------------------------------------------------------
    def _apply_semantics_prior(
        self,
        prefix: Sequence[str],
        ranked: RankedCandidates,
        scores: Dict[str, float],
    ) -> Tuple[RankedCandidates, Dict[str, float], Dict[str, Any]]:
        """
        Áp dụng SemanticsPrior nếu được bật.
        """
        if not (self.cfg.enable_semantics_prior and self.sem_prior is not None and ranked):
            return ranked, scores, {"used": False}

        ranked_labels = [a for a, _ in ranked]
        new_rank, new_scores, meta = self.sem_prior.apply(
            prefix=list(prefix),
            ranked=ranked_labels,
            scores=dict(scores),
        )
        new_ranked: RankedCandidates = [(a, new_scores.get(a, 0.0)) for a in new_rank]
        meta = dict(meta or {})
        meta.setdefault("used", True)
        return new_ranked, new_scores, meta

    # ------------------------------------------------------------------
    # 4) Guard (V2: soft penalty trên phân phối)
    # ------------------------------------------------------------------
    def _apply_guard(
        self,
        prefix: Sequence[str],
        ranked: RankedCandidates,
        scores: Dict[str, float],
    ) -> Tuple[RankedCandidates, Dict[str, float], Dict[str, Any]]:
        """
        Áp dụng Guard nếu được bật.
        """
        if not (self.cfg.enable_guard and self.guard is not None and ranked):
            return ranked, scores, {"used": False}

        ranking = [a for a, _ in ranked]
        scores_map = dict(scores)

        new_rank, new_scores, meta = self.guard.apply(
            prefix=list(prefix),
            ranked=ranking,
            scores=scores_map,
        )
        new_ranked: RankedCandidates = [(a, new_scores.get(a, 0.0)) for a in new_rank]
        meta = dict(meta or {})
        meta.setdefault("used", True)
        return new_ranked, new_scores, meta

    # ------------------------------------------------------------------
    # 5) API gốc: predict_one (trả dict, dùng cho tests + debug)
    # ------------------------------------------------------------------
    def predict_one(
        self,
        prefix: Sequence[str],
        activities: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy full pipeline cho 1 prefix.

        Parameters
        ----------
        prefix:
            Chuỗi activity đã xảy ra.
        activities:
            Danh sách toàn bộ activity có thể có.
            Nếu None → dùng self.activities_vocab.

        Returns
        -------
        dict:
            {
              "prediction": label cuối cùng (str | None),
              "candidates": [label1, label2, ...],
              "scores": {label: score},
              "meta": {sequence, graph, sem_prior, guard, semantics, explanation}
            }
        """
        pref = list(map(str, prefix))

        if activities is not None:
            all_activities = list(map(str, activities))
        else:
            all_activities = list(self.activities_vocab or [])

        # 1) Sequence
        ranked_seq, seq_meta = self._sequence_rank(pref)
        meta: Dict[str, Any] = {
            "sequence": seq_meta,
        }
        if not ranked_seq:
            # Không có candidate nào từ SequenceEnsemble → dừng luôn.
            return {
                "prediction": None,
                "candidates": [],
                "scores": {},
                "meta": meta,
            }

        # 2) GraphReasoner
        ranked_g, scores_g, graph_meta = self._apply_graph_reasoning(pref, ranked_seq)
        meta["graph"] = graph_meta

        # 3) Semantics-Prior
        ranked_s, scores_s, sem_prior_meta = self._apply_semantics_prior(pref, ranked_g, scores_g)
        meta["sem_prior"] = sem_prior_meta

        # 4) Guard (V2 – footprint/anomaly)
        ranked_guard, scores_guard, guard_meta = self._apply_guard(pref, ranked_s, scores_s)
        meta["guard"] = guard_meta

        candidate_labels = [a for a, _ in ranked_guard]

        # 5) Semantics LLM (Block 3: chọn nhãn cuối cùng)
        if self.semantics_llm is not None and all_activities and candidate_labels:
            chosen_label, sem_meta = self.semantics_llm.predict_from_candidates(
                activities=all_activities,
                prefix=pref,
                candidates=candidate_labels,
            )
            meta["semantics"] = {
                "enabled": True,
                "chosen_label": chosen_label,
                **(sem_meta or {}),
            }
        else:
            # fallback: chọn top-1 sau Guard
            chosen_label = candidate_labels[0] if candidate_labels else None
            meta["semantics"] = {
                "enabled": False,
                "chosen_label": chosen_label,
                "reason": "sequence_graph_sem_prior_guard_only_or_missing_vocab_or_llm_disabled",
            }

        final_label = chosen_label

        # 6) Explanation (Block 4: nếu bật)
        if (
            self.cfg.enable_explanation
            and self.semantics_llm is not None
            and self.prompt_pool is not None
            and all_activities
            and final_label is not None
            and candidate_labels
        ):
            try:
                expl = generate_explanation(
                    semantics_llm=self.semantics_llm,
                    prompt_pool=self.prompt_pool,
                    activities=all_activities,
                    prefix=pref,
                    candidates=candidate_labels,
                    chosen_label=final_label,
                )
                meta["explanation"] = expl
            except Exception as exc:  # không để explain làm crash pipeline
                meta["explanation"] = {
                    "error": str(exc),
                    "prompt_id": None,
                }

        final_scores = {a: scores_guard.get(a, 0.0) for a in candidate_labels}
        return {
            "prediction": final_label,
            "candidates": candidate_labels,
            "scores": final_scores,
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # 6) API mới cho eval: predict(...)
    # ------------------------------------------------------------------
    def predict(
        self,
        prefix: Sequence[str],
        activities_vocab: Optional[Sequence[str]] = None,
        measure_cost: bool = False,
    ) -> Tuple[str, List[str], Dict[str, Any]]:
        """
        API mỏng cho evaluation:

        - Được `run_evaluation` gọi.
        - Trả về (y_pred, ranked_labels, meta).

        Parameters
        ----------
        prefix:
            Chuỗi activity đã xảy ra.
        activities_vocab:
            Danh sách activity dùng cho bước SemanticsLLM.
            Nếu None → dùng self.activities_vocab.
        measure_cost:
            Được runner truyền xuống, nhưng ở đây không dùng (latency được đo ở runner).

        Returns
        -------
        y_pred:
            Nhãn cuối cùng (str hoặc None).
        ranked_labels:
            Danh sách candidate (top-k) sau toàn bộ pipeline (sequence+graph+sem_prior+guard).
        meta:
            Toàn bộ meta như predict_one.
        """
        _ = measure_cost  # hiện tại không dùng, để giữ signature
        res = self.predict_one(prefix=prefix, activities=activities_vocab)
        y_pred = res.get("prediction")
        ranked_labels = list(res.get("candidates", []))
        meta = res.get("meta", {})
        return y_pred, ranked_labels, meta

    # ------------------------------------------------------------------
    # 7) batch_predict (giữ lại cho tiện debug / test)
    # ------------------------------------------------------------------
    def batch_predict(
        self,
        prefixes: List[Sequence[str]],
        activities: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for pref in prefixes:
            res = self.predict_one(prefix=pref, activities=activities)
            results.append(res)
        return results
