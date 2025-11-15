# src/semantics/prompt_pool.py
# ----------------------------
"""
PromptPool cho S-NAPX.

Nhiệm vụ chính
--------------
- Gom toàn bộ prompt (cả IT chung, cả S-NAP specific) vào 1 registry.
- Cung cấp API đơn giản cho:
    + Lấy prompt strict cho S-NAP prediction (chọn nhãn từ top-k).
    + Lấy prompt strict cho S-NAP explanation (≤ 2 câu).
    + (Tuỳ chọn) lấy prompt IT để tạo dữ liệu instruction-tuning.

Ghi chú
-------
- Để tránh vỡ code nếu thiếu 4 file prompt IT, việc load chúng là "best effort":
    • Nếu import thành công → đưa vào pool.
    • Nếu không → bỏ qua, nhưng S-NAP inference vẫn chạy bình thường.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional

from .types import PromptTemplate, PromptTask


@dataclass
class PromptPoolConfig:
    """
    Cấu hình cho PromptPool.

    Attributes
    ----------
    default_prediction_id:
        id của PromptTemplate dùng mặc định cho S-NAP prediction.
    default_explanation_id:
        id của PromptTemplate dùng mặc định cho S-NAP explanation.
    """

    default_prediction_id: str = "snap_predict_strict_v1"
    default_explanation_id: str = "snap_explain_reasonable_v1"


class PromptPool:
    """
    Ngân hàng prompt cho toàn bộ Semantics Layer.

    Giao diện chính
    ---------------
    - get_prediction_template():
        PromptTemplate cho S-NAP prediction (chọn nhãn từ top-k).
    - get_explanation_template():
        PromptTemplate cho S-NAP explanation.
    - sample_it_template(task, variant_group="it_general"):
        Lấy 1 PromptTemplate cho IT (trace_anomaly, dfg...).
    """

    def __init__(
        self,
        config: Optional[PromptPoolConfig] = None,
        templates: Optional[Dict[str, PromptTemplate]] = None,
    ) -> None:
        self.config = config or PromptPoolConfig()
        # Registry: id → PromptTemplate
        self._templates: Dict[str, PromptTemplate] = templates or {}
        if templates is None:
            self._load_default_templates()

    # ------------------------------------------------------------------
    # Load default templates (S-NAP specific + IT prompts nếu có)
    # ------------------------------------------------------------------
    def _load_default_templates(self) -> None:
        """Khởi tạo một số PromptTemplate mặc định."""

        # 1) Prompt S-NAP prediction (chọn nhãn từ top-k)
        snap_predict_template = PromptTemplate(
            id="snap_predict_strict_v1",
            task="snap_predict",
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

        # 2) Prompt S-NAP explanation
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
                "Observed prefix: {prefix}\n"
                "Chosen next activity: {chosen_label}\n\n"
                "In at most TWO sentences, explain WHY this chosen activity is reasonable "
                "given the observed prefix and typical process behavior.\n"
                "Do NOT restate the full input. Focus on causal or ordering logic.\n"
                "Explanation:"
            ),
        )
        self._templates[snap_explain_template.id] = snap_explain_template

        # 3) IT prompts từ 4 file nếu có (prompts_all/prediction/discovery/anomaly)
        #    Ta không phụ thuộc vào chúng để S-NAP chạy, nên chỉ try/except nhẹ.
        for module_name in [
            "prompts_all",
            "prompts_prediction_excluded",
            "prompts_discovery_excluded",
            "prompts_anomaly_excluded",
        ]:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            general_intro = getattr(module, "GENERAL_INTRO", "")
            task_variants = getattr(module, "TASK_PROMPTS_VARIANTS", {})
            # Chỉ load thành PromptTemplate để dùng khi tạo IT dataset, không dùng trực tiếp cho inference.

            for task, variants in task_variants.items():
                # variants là list[str] hoặc list[dict] tuỳ định nghĩa gốc.
                # Ta coi mỗi phần tử là 1 template string.
                for idx, tpl in enumerate(variants):
                    tpl_str = tpl.strip()
                    template_id = f"it_{module_name}_{task}_{idx+1}"
                    if template_id in self._templates:
                        continue

                    # Map tên task IT sang PromptTask
                    pm_task: PromptTask
                    if task == "trace_anomaly":
                        pm_task = "trace_anomaly"
                    elif task == "activity_anomaly":
                        pm_task = "activity_anomaly"
                    elif task == "next_activity":
                        pm_task = "next_activity"
                    elif task == "dfg":
                        pm_task = "dfg"
                    elif task == "process_tree":
                        pm_task = "process_tree"
                    else:
                        # Task lạ: bỏ qua
                        continue

                    tmpl = PromptTemplate(
                        id=template_id,
                        task=pm_task,
                        variant_group="it_general",
                        description=f"IT template from {module_name}.{task}[{idx}]",
                        template=(general_intro + "\n\n" + tpl_str).strip(),
                    )
                    self._templates[tmpl.id] = tmpl

    # ------------------------------------------------------------------
    # API cho S-NAP inference
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
    # API cho IT / training
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
        candidates = self.list_templat_
