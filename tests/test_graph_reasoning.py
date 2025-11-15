# tests/test_graph_reasoning.py
# -----------------------------
from src.graph_reasoning.dfg import DFG
from src.graph_reasoning.reasoner import GraphReasoner, GraphReasonerConfig


def test_graph_reasoner_re_rank():
    # Simple traces: sau B thường là C
    traces = [
        ["A", "B", "C"],
        ["A", "B", "C"],
        ["A", "D", "C"],
    ]

    dfg = DFG.from_traces(traces, min_count=1, min_ratio=0.0)

    cfg = GraphReasonerConfig(
        enabled=True,
        hard_mask=False,
        boost_factor=1.0,
        min_prob_keep=0.0,
    )
    gr = GraphReasoner(dfg=dfg, cfg=cfg)

    prev = "B"
    ranked = ["C", "D"]
    scores = {"C": 0.1, "D": 0.9}  # ban đầu D cao hơn C

    new_rank, new_scores, meta = gr.rerank_candidates(prev, ranked, scores)

    assert meta["used"] is True
    # C phải được boost
    assert new_scores["C"] > scores["C"]
    # rank mới không rỗng và vẫn là tập con của ban đầu
    assert set(new_rank).issubset(set(ranked))
