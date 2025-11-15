# src/semantics/explanations.py
# -----------------------------
"""
Sinh và chuẩn hoá giải thích (explanations) cho dự đoán.

Nhiệm vụ
--------
- Lấy PromptTemplate explain từ PromptPool.
- Build prompt dựa trên activities, prefix, candidates, chosen_label.
- Gọi SemanticsLLM.generate_text(...) để sinh giải thích (≤ 2 câu).
- Chuẩn hoá text (strip, cắt ≤ 2 câu, bỏ prefix thừa).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .inference import SemanticsLLM
from .prompt_pool import PromptPool
from .types import PromptTemplate


def generate_explanation(
    semantics_llm: SemanticsLLM,
    prompt_pool: PromptPool,
    activities: Sequence[str],
    prefix: Sequence[str],
    candidates: List[str],
    chosen_label: str,
) -> Dict[str, Any]:
    """
    Gọi LLM để sinh giải thích cho 1 prediction.

    Returns
    -------
    result:
        Dict gồm:
        - 'text': giải thích cuối cùng (đã chuẩn hoá ≤ 2 câu).
        - 'raw_prompt': prompt full dùng để sinh.
        - 'raw_output': output gốc từ model.
        - 'reason_pass_local': bool (check format đơn giản).
        - 'prompt_id': id template explain dùng.
    """
    template: PromptTemplate = prompt_pool.get_explanation_template()
    prompt = _render_explanation_prompt(
        template=template,
        activities=activities,
        prefix=prefix,
        candidates=candidates,
        chosen_label=chosen_label,
    )

    raw_output = semantics_llm.generate_text(prompt)
    explanation_text = _normalize_explanation(raw_output)
    reason_pass = _reason_pass_local(explanation_text)

    return {
        "text": explanation_text,
        "raw_prompt": prompt,
        "raw_output": raw_output,
        "reason_pass_local": reason_pass,
        "prompt_id": template.id,
    }


def _render_explanation_prompt(
    template: PromptTemplate,
    activities: Sequence[str],
    prefix: Sequence[str],
    candidates: List[str],
    chosen_label: str,
) -> str:
    """Render prompt explain từ PromptTemplate."""
    activities_str = "{ " + ", ".join(map(str, activities)) + " }" if activities else "{}"
    prefix_str = " -> ".join(map(str, prefix)) if prefix else "<EMPTY_PREFIX>"
    candidates_str = "{ " + ", ".join(map(str, candidates)) + " }"

    return template.template.format(
        activities=activities_str,
        prefix=prefix_str,
        candidates=candidates_str,
        chosen_label=chosen_label,
    )


def _normalize_explanation(raw_text: str) -> str:
    """
    Chuẩn hoá chuỗi giải thích:
    - strip khoảng trắng.
    - cắt ≤ 2 câu.
    - loại bỏ prefix thừa kiểu 'Explanation:'.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Bỏ prefix "Explanation:" nếu có
    for prefix in ["Explanation:", "EXPLANATION:", "explanation:"]:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    # Cắt tối đa 2 câu (tách theo dấu chấm)
    # Đây là heuristic đơn giản, đủ dùng cho Reason-Pass local.
    sentences = [s.strip() for s in text.replace("?\n", "?\n").split(".") if s.strip()]
    if len(sentences) > 2:
        sentences = sentences[:2]
    normalized = ". ".join(sentences)
    if normalized and not normalized.endswith("."):
        normalized += "."

    return normalized


def _reason_pass_local(text: str) -> bool:
    """
    Check Reason-Pass local (rất đơn giản):
    - Không rỗng.
    - Không quá dài (<= 3 câu).
    """
    if not text:
        return False
    # đếm câu bằng dấu '.'
    num_sentences = text.count(".")
    return 1 <= num_sentences <= 3


__all__ = ["generate_explanation"]
