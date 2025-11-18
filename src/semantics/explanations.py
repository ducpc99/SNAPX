# src/semantics/explanations.py
# -----------------------------
"""
Sinh và chuẩn hoá giải thích (explanations) cho dự đoán.

Nhiệm vụ
--------
- Lấy PromptTemplate explain từ PromptPool.
- Build prompt dựa trên activities, prefix, candidates, chosen_label.
- Gọi SemanticsLLM.generate_text(...) để sinh giải thích.
- Chuẩn hoá text (strip, cắt ≤ 2 câu, bỏ phần prompt echo).
- Trả ra dict meta dùng được cho EvalRunner + explain_metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .prompt_pool import PromptPool
from .inference import SemanticsLLM
from .types import PromptTemplate


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def generate_explanation(
    semantics_llm: SemanticsLLM,
    prompt_pool: PromptPool,
    activities: Sequence[str],
    prefix: Sequence[str],
    candidates: Sequence[str],
    chosen_label: str,
) -> Dict[str, Any]:
    """
    Sinh giải thích cho dự đoán chosen_label.

    Parameters
    ----------
    semantics_llm:
        Đối tượng SemanticsLLM đã load model.
    prompt_pool:
        PromptPool (đã có template snap_explain).
    activities:
        Toàn bộ activity trong process (vocab).
    prefix:
        Chuỗi hoạt động đã diễn ra.
    candidates:
        Danh sách candidate (top-k) mà model consider.
    chosen_label:
        Nhãn cuối cùng được chọn (final prediction).

    Returns
    -------
    Dict[str, Any]:
        {
            "text": explanation_text,        # giải thích đã chuẩn hoá
            "raw_prompt": prompt,            # full prompt đưa vào LLM
            "raw_output": raw_output,        # raw decode từ model
            "reason_pass_local": bool,       # cờ Reason-Pass đơn giản
            "prompt_id": template.id,        # id template explain
        }
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
    explanation_text = _normalize_explanation(raw_output, prompt)
    reason_pass = _reason_pass_local(explanation_text)

    return {
        "text": explanation_text,
        "raw_prompt": prompt,
        "raw_output": raw_output,
        "reason_pass_local": reason_pass,
        "prompt_id": template.id,
    }


# ----------------------------------------------------------------------
# Helpers: render prompt
# ----------------------------------------------------------------------
def _render_explanation_prompt(
    template: PromptTemplate,
    activities: Sequence[str],
    prefix: Sequence[str],
    candidates: Sequence[str],
    chosen_label: str,
) -> str:
    """
    Render prompt cho explanation từ PromptTemplate.

    Quy ước string:
    - {activities}: { a1, a2, ... }
    - {prefix}: a1 -> a2 -> ... (hoặc <EMPTY_PREFIX> nếu rỗng)
    - {candidates}: { c1, c2, ... }
    - {chosen_label}: tên activity dự đoán.
    """
    if activities:
        activities_str = "{ " + ", ".join(map(str, activities)) + " }"
    else:
        activities_str = "{ }"

    prefix_str = " -> ".join(map(str, prefix)) if prefix else "<EMPTY_PREFIX>"

    if candidates:
        candidates_str = "{ " + ", ".join(map(str, candidates)) + " }"
    else:
        candidates_str = "{ }"

    return template.template.format(
        activities=activities_str,
        prefix=prefix_str,
        candidates=candidates_str,
        chosen_label=str(chosen_label),
    )


# ----------------------------------------------------------------------
# Helpers: normalize explanation text
# ----------------------------------------------------------------------
def _normalize_explanation(raw_output: str, prompt: str) -> str:
    """
    Chuẩn hoá text giải thích từ raw_output.

    Các bước:
    - Bỏ phần prompt echo (nếu model in lại prompt).
    - Strip đầu cuối.
    - Bỏ prefix kiểu 'Answer:' / 'Explanation:' nếu có.
    - Cắt còn tối đa 2 câu.
    - Giới hạn độ dài tối đa (khoảng vài trăm ký tự cho an toàn).
    """
    if not raw_output:
        return ""

    text = raw_output

    # 1) Bỏ phần prompt echo nếu output bắt đầu bằng prompt
    text = text.strip()
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()
    else:
        # fallback: nếu prompt xuất hiện bên trong, cắt theo lần xuất hiện cuối
        idx = text.rfind(prompt)
        if idx != -1:
            text = text[idx + len(prompt) :].strip()

    # 2) Bỏ prefix Answer: / Explanation:
    lowered = text.lower()
    for prefix in ["answer:", "explanation:", "giải thích:", "giai thich:"]:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip()
            break

    # 3) Cắt còn tối đa 2 câu (dùng dấu . ? ! đơn giản)
    sentences = _split_sentences_simple(text)
    if len(sentences) > 2:
        sentences = sentences[:2]
    text = " ".join(s.strip() for s in sentences if s.strip())

    # 4) Giới hạn độ dài (phòng model lảm nhảm)
    max_chars = 600
    if len(text) > max_chars:
        text = text[: max_chars].rstrip()

    return text.strip()


def _split_sentences_simple(text: str) -> Sequence[str]:
    """
    Split câu rất đơn giản theo . ? ! để đủ dùng cho Reason-Pass.
    Không xử lý trường hợp viết tắt phức tạp.
    """
    import re

    if not text:
        return []

    # Tách theo dấu câu kèm khoảng trắng phía sau
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    # Nếu không có dấu câu nào → coi toàn bộ là 1 câu
    if len(parts) == 1:
        return [text]
    return parts


def _reason_pass_local(text: str) -> bool:
    """
    Check Reason-Pass local (rất đơn giản):
    - Không rỗng.
    - Không quá dài (<= 3 câu).
    """
    if not text:
        return False

    # Đếm câu bằng tách đơn giản
    sentences = _split_sentences_simple(text)
    num_sentences = len([s for s in sentences if s.strip()])
    return 1 <= num_sentences <= 3


__all__ = ["generate_explanation"]
