# tests/test_hybrid_pipeline.py
# -----------------------------
from typing import Any, Dict, List, Sequence, Tuple

from src.sequence.ensemble import SequenceEnsemble
from src.graph_reasoning.dfg import DFG
from src.graph_reasoning.reasoner import GraphReasoner, GraphReasonerConfig
from src.graph_reasoning.semantics_prior import SemanticsPrior, SemanticsPriorConfig
from src.semantics.prompt_pool import PromptPool, PromptPoolConfig
from src.hybrid.pipeline import HybridSNAP, HybridConfig


class FakeSemanticsLLM:
    """
    Fake LLM dùng cho test:
    - predict_from_candidates: luôn chọn candidate đầu tiên (deterministic).
    Không load model thật, nên chạy rất nhanh, không tốn VRAM.
    """

    def __init__(self, prompt_pool: PromptPool) -> None:
        self.prompt_pool = prompt_pool

    def predict_from_candidates(
        self,
        activities: Sequence[str],
        prefix: Sequence[str],
        candidates: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        chosen = candidates[0]
        meta = {
            "prompt_id": self.prompt_pool.get_prediction_template().id,
            "raw_prompt": "<fake>",
            "mode": "fake",
            "candidate_logprobs": {c: 0.0 for c in candidates},
        }
        return chosen, meta


def test_hybrid_pipeline_simple():
    # Toy traces
    traces = [
        ["A", "B", "C"],
        ["A", "B", "D"],
    ]

    # Sequence
    seq_model = SequenceEnsemble(alpha=0.5, markov_order=2, min_count=1)
    seq_model.fit(traces)

    # DFG + Reasoner
    dfg = DFG.from_traces(traces, min_count=1, min_ratio=0.0)
    gr_cfg = GraphReasonerConfig(
        enabled=True,
        hard_mask=False,
        boost_factor=0.5,
        min_prob_keep=0.0,
    )
    graph_reasoner = GraphReasoner(dfg=dfg, cfg=gr_cfg)

    # Semantics-Prior (tắt, nhưng vẫn truyền object)
    sem_prior = SemanticsPrior(
        SemanticsPriorConfig(enabled=False, mode="activity", default_weight=1.0),
        prior_map={},
    )

    # PromptPool + Fake LLM
    prompt_pool = PromptPool(PromptPoolConfig())
    fake_llm = FakeSemanticsLLM(prompt_pool)

    # Vocab
    activities_vocab = sorted({a for tr in traces for a in tr})

    hyb_cfg = HybridConfig(
        top_k=3,
        enable_graph_reasoner=True,
        enable_semantics_prior=False,
        enable_guard=False,
        enable_explanation=True,  # bật để HybridSNAP gọi fake_llm
    )

    hybrid = HybridSNAP(
        cfg=hyb_cfg,
        seq_model=seq_model,
        graph_reasoner=graph_reasoner,
        sem_prior=sem_prior,
        semantics_llm=fake_llm,
        guard=None,
        prompt_pool=prompt_pool,
        activities_vocab=activities_vocab,
    )

    prefix = ["A", "B"]
    out = hybrid.predict_one(prefix=prefix, activities=activities_vocab)

    assert out["prediction"] in out["candidates"]
    assert "sequence" in out["meta"]
    assert "graph" in out["meta"]
    assert "semantics" in out["meta"]
    # Nếu enable_explanation=True, meta["semantics"] nên có info về fake_llm
    assert out["meta"]["semantics"].get("mode") == "fake"
