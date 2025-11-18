# src/semantics/prompt_pool.py
# ----------------------------
"""
PromptPool cho S-NAPX.

Nhiệm vụ chính
--------------
- Gom toàn bộ prompt (cả S-NAP prediction, explanation, và IT) vào 1 registry.
- Cung cấp API đơn giản cho:
    + Lấy prompt strict cho S-NAP prediction (chọn nhãn từ top-k).
    + Lấy prompt strict cho S-NAP explanation (≤ 2 câu).
    + (Tuỳ chọn) lấy prompt IT để tạo dữ liệu instruction-tuning.

Thiết kế
--------
- PromptTemplate được định nghĩa trong src/semantics/types.py.
- PromptPoolConfig cho phép:
    + Bật/tắt tự động nạp prompt IT.
    + Chỉ định tên module chứa prompt IT (vd: prompts_all, prompts_*_excluded).
    + Đặt id mặc định cho template prediction / explanation.

Ghi chú
-------
- Việc load 4 file prompt IT là "best effort":
    + Nếu import thành công → thêm vào pool như template IT (variant_group="it_general").
    + Nếu import lỗi (không có file, sai path, v.v.) → chỉ in cảnh báo, pipeline vẫn chạy.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .types import PromptTemplate, PromptTask


# ----------------------------------------------------------------------
# Cấu hình cho PromptPool
# ----------------------------------------------------------------------
@dataclass
class PromptPoolConfig:
    """
    Cấu hình cho PromptPool.

    Attributes
    ----------
    load_it_templates:
        Có cố gắng load prompt IT từ các module Python không.
    it_modules:
        Danh sách tên module Python để load prompt IT.
        Ví dụ: ("prompts_all", "prompts_prediction_excluded", ...)
    default_prediction_id:
        Id PromptTemplate dùng mặc định cho S-NAP prediction.
    default_explanation_id:
        Id PromptTemplate dùng mặc định cho S-NAP explanation.
    """

    load_it_templates: bool = True
    it_modules: Sequence[str] = (
        "prompts_all",
        "prompts_prediction_excluded",
        "prompts_discovery_excluded",
        "prompts_anomaly_excluded",
    )
    default_prediction_id: str = "snap_predict_strict_v1"
    default_explanation_id: str = "snap_explain_reasonable_v1"


# ----------------------------------------------------------------------
# PromptPool
# ----------------------------------------------------------------------
class PromptPool:
    """
    Registry đơn giản quản lý PromptTemplate.

    - self._templates: Dict[id, PromptTemplate]
    """

    def __init__(
        self,
        config: Optional[PromptPoolConfig] = None,
        templates: Optional[Dict[str, PromptTemplate]] = None,
    ) -> None:
        self.config = config or PromptPoolConfig()
        # nếu caller truyền sẵn templates thì dùng, else khởi tạo rỗng
        self._templates: Dict[str, PromptTemplate] = dict(templates or {})

        # luôn có ít nhất 2 template core (predict + explain)
        self._load_default_templates()

        # cố gắng nạp thêm template IT (nếu config cho phép)
        if self.config.load_it_templates:
            self._load_it_templates_best_effort()

    # ------------------------------------------------------------------
    # Default templates (S-NAP specific)
    # ------------------------------------------------------------------
    def _load_default_templates(self) -> None:
        """Khởi tạo PromptTemplate mặc định cho S-NAP prediction & explanation."""

        # 1) Prompt S-NAP prediction (chọn nhãn từ top-k)
        if "snap_predict_strict_v1" not in self._templates:
            snap_predict_template = PromptTemplate(
                id="snap_predict_strict_v1",
                task="snap_predict",  # kiểu logic riêng cho SemanticsLLM
                variant_group="snap_predict",
                description="Chọn chính xác 1 activity tiếp theo từ CANDIDATE list (S-NAP).",
                template=(
                    "You are an advanced AI system specialized in solving process mining tasks.\n"
                    "We have a partial execution trace of activities from a business process.\n"
                    "From the provided CANDIDATE list, which single activity should come next?\n\n"
                    "All process activities: {activities}\n"
                    "Candidate next activities (top-k): {candidates}\n"
                    "So far executed (prefix): {prefix}\n\n"
                    "Answer with exactly ONE activity name from the CANDIDATE list and nothing else.\n"
                    "Answer:"
                ),
            )
            self._templates[snap_predict_template.id] = snap_predict_template

        # 2) Prompt S-NAP explanation (≤ 2 câu)
        if "snap_explain_reasonable_v1" not in self._templates:
            snap_explain_template = PromptTemplate(
                id="snap_explain_reasonable_v1",
                task="snap_explain",
                variant_group="snap_explain",
                description="Giải thích ≤ 2 câu vì sao chosen_label là hợp lý.",
                template=(
                    "You are an advanced AI system specialized in solving process mining tasks.\n"
                    "A model has predicted the next activity for an ongoing process execution.\n\n"
                    "All process activities: {activities}\n"
                    "Candidate next activities (top-k): {candidates}\n"
                    "So far executed (prefix): {prefix}\n"
                    "Predicted next activity: {chosen_label}\n\n"
                    "In at most 2 sentences, explain concisely why this prediction is reasonable "
                    "given the process and the observed trace.\n"
                    "Answer:"
                ),
            )
            self._templates[snap_explain_template.id] = snap_explain_template

    # ------------------------------------------------------------------
    # Load IT templates từ các module prompts_*.py (best effort)
    # ------------------------------------------------------------------
    def _load_it_templates_best_effort(self) -> None:
        """
        Cố gắng import các module prompt IT (prompts_all, prompts_*_excluded, ...).

        Mỗi module (nếu tồn tại) được kỳ vọng có:
        - GENERAL_INTRO: str
        - TASK_PROMPTS_VARIANTS: Dict[str, List[Union[str, Dict]]]
        - TASK_PROMPTS_INVERTED_NEGATIVE: Dict[str, List[Union[str, Dict]]] (tuỳ)
        - TASK_PROMPTS_INVERTED_POSITIVE: Dict[str, List[Union[str, Dict]]] (tuỳ)

        Ta sẽ convert chúng thành PromptTemplate với:
        - task: map trực tiếp các key 'trace_anomaly', 'activity_anomaly', 'next_activity', 'dfg', 'process_tree'.
        - variant_group: 'it_general'.
        - template: GENERAL_INTRO + "\n\n" + template_str (nếu GENERAL_INTRO tồn tại).
        """
        for module_name in self.config.it_modules:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                # Không in quá ồn để tránh spam log khi không dùng IT
                # print(f"[PromptPool] Không import được module IT: {module_name}: {e}")
                continue

            general_intro = getattr(mod, "GENERAL_INTRO", "").strip()

            for dict_name in [
                "TASK_PROMPTS_VARIANTS",
                "TASK_PROMPTS_INVERTED_NEGATIVE",
                "TASK_PROMPTS_INVERTED_POSITIVE",
            ]:
                task_dict = getattr(mod, dict_name, None)
                if not isinstance(task_dict, dict):
                    continue

                for task_name, variants in task_dict.items():
                    # map tên task trong file IT sang PromptTask
                    pm_task: Optional[PromptTask]
                    if task_name in {
                        "trace_anomaly",
                        "activity_anomaly",
                        "next_activity",
                        "dfg",
                        "process_tree",
                    }:
                        pm_task = task_name  # type: ignore[assignment]
                    else:
                        # Task lạ: bỏ qua
                        continue

                    if not isinstance(variants, (list, tuple)):
                        continue

                    for idx, item in enumerate(variants):
                        # Mỗi phần tử có thể là:
                        # - string (template thuần)
                        # - dict có key "template"
                        if isinstance(item, dict):
                            tpl_str = str(item.get("template", "")).strip()
                        else:
                            tpl_str = str(item).strip()

                        if not tpl_str:
                            continue

                        template_id = f"it_{module_name}_{dict_name}_{task_name}_{idx+1}"
                        if template_id in self._templates:
                            continue

                        full_tpl = tpl_str
                        if general_intro:
                            full_tpl = (general_intro + "\n\n" + tpl_str).strip()

                        tmpl = PromptTemplate(
                            id=template_id,
                            task=pm_task,  # type: ignore[arg-type]
                            variant_group="it_general",
                            description=f"IT template from {module_name}.{dict_name}.{task_name}[{idx}]",
                            template=full_tpl,
                        )
                        self._templates[tmpl.id] = tmpl

    # ------------------------------------------------------------------
    # API: lấy template prediction / explanation
    # ------------------------------------------------------------------
    def get_prediction_template(self) -> PromptTemplate:
        """PromptTemplate cho S-NAP prediction (strict)."""
        template_id = self.config.default_prediction_id
        if template_id not in self._templates:
            raise KeyError(
                f"Prediction template id '{template_id}' không tồn tại trong PromptPool."
            )
        return self._templates[template_id]

    def get_explanation_template(self) -> PromptTemplate:
        """PromptTemplate cho S-NAP explanation."""
        template_id = self.config.default_explanation_id
        if template_id not in self._templates:
            raise KeyError(
                f"Explanation template id '{template_id}' không tồn tại trong PromptPool."
            )
        return self._templates[template_id]

    # ------------------------------------------------------------------
    # API cho IT / training / phân tích prompt
    # ------------------------------------------------------------------
    def list_templates(
        self,
        task: Optional[PromptTask] = None,
        variant_group: Optional[str] = None,
    ) -> List[PromptTemplate]:
        """
        Liệt kê PromptTemplate theo bộ lọc (task, variant_group).

        Dùng cho:
        - generator tạo instruction dataset.
        - kiểm thử / phân tích prompt.
        """
        results: List[PromptTemplate] = []
        for tmpl in self._templates.values():
            if task is not None and tmpl.task != task:
                continue
            if variant_group is not None and tmpl.variant_group != variant_group:
                continue
            results.append(tmpl)
        return results

    def sample_it_template(
        self,
        task: PromptTask,
        variant_group: str = "it_general",
    ) -> PromptTemplate:
        """
        Chọn 1 PromptTemplate để dùng trong instruction-tuning.

        Nếu không tìm thấy template phù hợp sẽ raise ValueError.
        """
        candidates = self.list_templates(task=task, variant_group=variant_group)
        if not candidates:
            raise ValueError(
                f"Không tìm thấy IT template cho task='{task}', variant_group='{variant_group}'. "
                "Hãy kiểm tra lại các module prompts_*.py hoặc cấu hình PromptPoolConfig."
            )
        # Để reproducible, chọn phần tử đầu tiên (có thể đổi thành random.choice nếu muốn).
        return candidates[0]


__all__ = ["PromptPoolConfig", "PromptPool"]
