# tests/test_sequence.py
# ----------------------
from src.sequence.ensemble import SequenceEnsemble


# tests/test_sequence.py
# ----------------------
from src.sequence.ensemble import SequenceEnsemble


def test_sequence_ensemble_basic():
    # Toy traces
    traces = [
        ["A", "B", "C"],
        ["A", "B", "D"],
        ["A", "E"],
    ]

    seq = SequenceEnsemble(alpha=0.5, markov_order=2, min_count=1)
    seq.fit(traces)

    # prefix: ["A"] -> B phải là ứng viên hợp lý
    prefix1 = ["A"]
    scores1 = seq.propose_candidates(prefix1, k=3)
    assert scores1, "SequenceEnsemble không trả candidate nào cho prefix ['A']"
    assert "B" in scores1 and scores1["B"] > 0.0

    # prefix: ["A","B"] → C và D đều có thể xuất hiện
    prefix2 = ["A", "B"]
    scores2 = seq.propose_candidates(prefix2, k=3)
    assert set(scores2.keys()) <= {"C", "D"}
    # ít nhất phải có >=1 candidate
    assert len(scores2) >= 1
