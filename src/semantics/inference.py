# src/semantics/inference.py
# --------------------------
"""
SemanticsLLM
============

Lớp bọc LLM cho Semantics Layer của S-NAPX.

Nhiệm vụ
--------
- Load model (Hugging Face) theo RuntimeConfig.
- Dùng PromptPool để build prompt prediction:
    + template snap_predict_strict_v1.
    + format với {activities}, {candidates}, {prefix}.
- (Tuỳ chọn) chèn thêm few-shot IT từ InstructionMemory vào đầu prompt.
- Cung cấp 2 mode chọn nhãn:
    • "logprob": tính log-xác suất cho từng candidate, chọn max.
    • "generate": để LLM tự sinh, rồi parse nhãn.
- Cung cấp hàm generate_text(prompt) để dùng cho explain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    torch = None
    F = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from .prompt_pool import PromptPool
from .types import PromptTemplate
from .instruction_memory import InstructionMemory, ITExample


# ----------------------------------------------------------------------
# Dataclass cấu hình
# ----------------------------------------------------------------------
@dataclass
class RuntimeConfig:
    """Cấu hình runtime cho LLM."""

    model_name: str
    max_seq_len: int = 2048
    load_in_4bit: bool = False
    device: str = "auto"  # "auto" | "cpu" | "cuda"


@dataclass
class GenConfig:
    """Cấu hình sinh output (decode)."""

    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False


# ----------------------------------------------------------------------
# SemanticsLLM
# ----------------------------------------------------------------------
class SemanticsLLM:
    """Lớp wrap để gọi LLM theo chuẩn của S-NAPX."""

    def __init__(
        self,
        prompt_pool: PromptPool,
        runtime_cfg: RuntimeConfig,
        gen_cfg: Optional[GenConfig] = None,
        mode: str = "logprob",
        # IT few-shot
        instruction_memory: Optional[InstructionMemory] = None,
        use_it_fewshot: bool = False,
        it_num_shots: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        prompt_pool:
            Ngân hàng prompt (đã chứa template cho snap_predict & snap_explain).
        runtime_cfg:
            Cấu hình runtime model (tên model, device, 4bit...).
        gen_cfg:
            Cấu hình sinh output.
        mode:
            Cách chọn nhãn:
            - "logprob": tính logprob từng candidate.
            - "generate": model sinh text rồi parse nhãn.
        instruction_memory:
            Bộ nhớ IT (S-NAP_instructions) dùng để lấy few-shot ví dụ.
        use_it_fewshot:
            Nếu True và có instruction_memory → chèn few-shot IT vào đầu prompt.
        it_num_shots:
            Số lượng ví dụ IT tối đa lấy cho mỗi prefix.
        """
        self.prompt_pool = prompt_pool
        self.runtime_cfg = runtime_cfg
        self.gen_cfg = gen_cfg or GenConfig()
        if mode not in {"logprob", "generate"}:
            raise ValueError("mode phải là 'logprob' hoặc 'generate'")
        self.mode = mode

        self.instruction_memory = instruction_memory
        self.use_it_fewshot = use_it_fewshot
        self.it_num_shots = max(0, int(it_num_shots))

        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "Cần cài đặt transformers + torch để dùng SemanticsLLM "
                "(pip install transformers torch)."
            )

        # Resolve device
        self.device = self._resolve_device(runtime_cfg.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(runtime_cfg.model_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load model (có hỗ trợ 4bit đơn giản)
        if runtime_cfg.load_in_4bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                runtime_cfg.model_name,
                device_map={"": str(self.device)},
                load_in_4bit=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(runtime_cfg.model_name)
            self.model.to(self.device)

        self.model.eval()

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------
    def _resolve_device(self, device_str: str) -> "torch.device":
        """Chuyển string sang torch.device, có hỗ trợ 'auto'."""
        if not torch.cuda.is_available():
            return torch.device("cpu")

        if device_str == "auto":
            return torch.device("cuda")

        try:
            return torch.device(device_str)
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Core API: predict_from_candidates
    # ------------------------------------------------------------------
    def predict_from_candidates(
        self,
        activities: Sequence[str],
        prefix: Sequence[str],
        candidates: List[str],
        model_id: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Chọn 1 nhãn từ danh sách candidates.

        Parameters
        ----------
        activities:
            Tập toàn bộ activity có thể có trong process.
        prefix:
            Chuỗi hoạt động đã xảy ra.
        candidates:
            Danh sách top-k candidate từ sequence/graph layer.
        model_id:
            (tuỳ chọn) id process/model, dùng để ưu tiên ví dụ IT cùng domain.

        Returns
        -------
        chosen_label:
            Nhãn được chọn sau khi apply Semantics LLM.
        meta:
            Dict chứa các thông tin phụ:
            - 'prompt_id': id PromptTemplate được dùng.
            - 'raw_prompt': chuỗi prompt fully rendered.
            - 'mode': 'logprob' hoặc 'generate'.
            - 'candidate_logprobs': dict[candidate, logprob] (mode=logprob).
            - 'raw_output': output gốc khi generate (mode=generate).
            - 'it_fewshot_used': bool.
            - 'it_fewshot_count': int.
        """
        if not candidates:
            return "", {
                "error": "empty_candidates",
                "mode": self.mode,
                "it_fewshot_used": False,
                "it_fewshot_count": 0,
            }

        template: PromptTemplate = self.prompt_pool.get_prediction_template()

        # Lấy ví dụ IT (nếu bật use_it_fewshot)
        it_examples: List[ITExample] = []
        if self.use_it_fewshot and self.instruction_memory is not None and self.it_num_shots > 0:
            it_examples = self.instruction_memory.query_by_prefix(
                prefix, k=self.it_num_shots, model_id=model_id
            )

        prompt = self._render_prediction_prompt(
            template=template,
            activities=activities,
            prefix=prefix,
            candidates=candidates,
            it_examples=it_examples,
        )

        if self.mode == "logprob":
            cand_logprobs = self._score_candidates_logprob(prompt, candidates)
            chosen = max(cand_logprobs.items(), key=lambda x: x[1])[0]
            meta: Dict[str, Any] = {
                "prompt_id": template.id,
                "raw_prompt": prompt,
                "mode": "logprob",
                "candidate_logprobs": cand_logprobs,
                "it_fewshot_used": bool(it_examples),
                "it_fewshot_count": len(it_examples),
            }
            return chosen, meta

        # mode == "generate"
        raw_output = self.generate_text(prompt)
        chosen = self._parse_label_from_output(raw_output, candidates)
        meta = {
            "prompt_id": template.id,
            "raw_prompt": prompt,
            "mode": "generate",
            "raw_output": raw_output,
            "it_fewshot_used": bool(it_examples),
            "it_fewshot_count": len(it_examples),
        }
        return chosen, meta

    # ------------------------------------------------------------------
    # Public helper: generate text cho explain v.v.
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Sinh text từ model, dùng cho cả prediction (mode=generate) và explain.
        """
        if max_new_tokens is None:
            max_new_tokens = self.gen_cfg.max_new_tokens

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.runtime_cfg.max_seq_len,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=self.gen_cfg.temperature,
                top_p=self.gen_cfg.top_p,
                do_sample=self.gen_cfg.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text

    # ------------------------------------------------------------------
    # Render prompt prediction (+ IT few-shot nếu có)
    # ------------------------------------------------------------------
    def _render_prediction_prompt(
        self,
        template: PromptTemplate,
        activities: Sequence[str],
        prefix: Sequence[str],
        candidates: List[str],
        it_examples: Optional[Sequence[ITExample]] = None,
    ) -> str:
        """
        Render prompt prediction từ PromptTemplate + (optional) IT few-shot.

        Cấu trúc:
        - Nếu có it_examples:
            + "Here are some past examples..." + mỗi example: instruction + correct answer.
            + "Now solve the new case:\n"
        - Sau đó là phần core template:
            template.template.format(activities, prefix, candidates)
        """
        # stringify ...
        if activities:
            activities_str = "{ " + ", ".join(map(str, activities)) + " }"
        else:
            activities_str = "{ }"

        prefix_str = " -> ".join(map(str, prefix)) if prefix else "<EMPTY_PREFIX>"
        candidates_str = "{ " + ", ".join(map(str, candidates)) + " }"

        core_prompt = template.template.format(
            activities=activities_str,
            prefix=prefix_str,
            candidates=candidates_str,
        )

        # Nếu không có IT examples thì trả về core luôn
        if not it_examples:
            return core_prompt

        fewshot_block = self._format_it_examples_block(it_examples)
        full_prompt = (
            "You are an advanced AI system specialized in solving process mining tasks.\n"
            "Below are some previously solved examples of predicting the next activity.\n\n"
            f"{fewshot_block}\n"
            "Now, given a NEW case, answer the question below.\n\n"
            f"{core_prompt}"
        )
        return full_prompt

    def _format_it_examples_block(self, examples: Sequence[ITExample]) -> str:
        """
        Format danh sách ITExample thành block few-shot.

        Mỗi example dạng:
        Example i:
        <instruction>
        Correct answer: <output>
        """
        lines: List[str] = []
        for idx, ex in enumerate(examples, start=1):
            instr = ex.instruction.strip()
            out = ex.output.strip()
            if not instr or not out:
                continue
            lines.append(f"Example {idx}:")
            lines.append(instr)
            lines.append(f"Correct answer: {out}")
            lines.append("")  # dòng trống
        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Mode logprob: chấm điểm từng candidate
    # ------------------------------------------------------------------
    def _score_candidates_logprob(
        self,
        prompt: str,
        candidates: Sequence[str],
    ) -> Dict[str, float]:
        """
        Tính log-xác suất P(candidate | prompt) xấp xỉ cho từng candidate.
        """
        if F is None:
            raise RuntimeError("torch.nn.functional (F) không khả dụng.")

        scores: Dict[str, float] = {}

        for cand in candidates:
            prompt_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
            cand_ids = self.tokenizer.encode(
                " " + cand,
                add_special_tokens=False,
            )

            token_ids = prompt_ids + cand_ids

            max_len = self.runtime_cfg.max_seq_len
            if len(token_ids) > max_len:
                token_ids = token_ids[-max_len:]

            full_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=full_ids)
                logits = outputs.logits  # [1, L, V]

            log_probs = F.log_softmax(logits, dim=-1)  # [1, L, V]
            log_probs_tokens = log_probs[:, :-1, :]  # [1, L-1, V]
            target_ids = full_ids[:, 1:]  # [1, L-1]
            token_logprobs = log_probs_tokens.gather(
                2, target_ids.unsqueeze(-1)
            ).squeeze(-1)[0]  # [L-1]

            cand_len = min(len(cand_ids), token_logprobs.size(0))
            if cand_len == 0:
                scores[cand] = -math.inf
                continue
            cand_logprob = float(token_logprobs[-cand_len:].sum().item())
            scores[cand] = cand_logprob

        return scores

    # ------------------------------------------------------------------
    # Parse label từ output generate
    # ------------------------------------------------------------------
    def _parse_label_from_output(
        self,
        raw_output: str,
        candidates: Sequence[str],
    ) -> str:
        """
        Parse nhãn từ output model (mode=generate) theo heuristic đơn giản.
        """
        if not raw_output.strip():
            return candidates[0]

        first_line = raw_output.strip().splitlines()[0].strip()

        for prefix in ["Answer:", "answer:", "Next activity:", "next activity:"]:
            if first_line.lower().startswith(prefix.lower()):
                first_line = first_line[len(prefix) :].strip()
                break

        first_clean = first_line.strip().strip(" '\"")
        while first_clean and first_clean[-1] in {".", ",", ";", ":"}:
            first_clean = first_clean[:-1].strip()

        if not first_clean:
            return candidates[0]

        for c in candidates:
            if first_clean.lower() == c.lower():
                return c

        low_map = {c.lower(): c for c in candidates}
        if first_clean.lower() in low_map:
            return low_map[first_clean.lower()]

        for c in candidates:
            if c in first_line:
                return c

        return candidates[0]


__all__ = ["SemanticsLLM", "RuntimeConfig", "GenConfig"]
