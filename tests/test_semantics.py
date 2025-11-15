# tests/test_semantics.py
# -----------------------
from src.semantics.prompt_pool import PromptPool, PromptPoolConfig

try:
    # Nếu bạn có hàm normalize nội bộ
    from src.semantics.explanations import _normalize_explanation
    HAS_NORMALIZE = True
except Exception:
    HAS_NORMALIZE = False


def test_prompt_pool_has_snap_templates():
    pool = PromptPool(PromptPoolConfig())

    pred_tmpl = pool.get_prediction_template()
    expl_tmpl = pool.get_explanation_template()

    # Kiểm tra id / task mang tên S-NAP (tuỳ bạn đang đặt id)
    assert "snap" in pred_tmpl.task.lower()
    assert "predict" in pred_tmpl.task.lower()
    assert "CANDIDATE" in pred_tmpl.template

    assert "snap" in expl_tmpl.task.lower()
    assert "explain" in expl_tmpl.task.lower()


def test_normalize_explanation_max_two_sentences_if_available():
    if not HAS_NORMALIZE:
        # Nếu bạn chưa implement normalize, bỏ qua test này
        return

    raw = "Explanation: Câu 1 rất dài. Câu 2 cũng được. Câu 3 nên bị cắt."
    norm = _normalize_explanation(raw)
    # chỉ còn tối đa 2 câu
    assert norm.count(".") <= 2
    assert "Câu 3" not in norm
