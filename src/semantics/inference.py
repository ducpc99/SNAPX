# src/semantics/inference.py
# --------------------------
"""
SemanticsLLM
============

Lớp bọc LLM cho Semantics Layer của S-NAP.

Nhiệm vụ
--------
- Load model (Hugging Face) theo RuntimeConfig.
- Dùng PromptPool để build prompt prediction:
    + template snap_predict_strict_v1.
    + format với {activities}, {candidates}, {prefix}.
- Cung cấp 2 mode chọn nhãn:
    • "logprob": tính log-xác suất cho từng candidate, chọn max.
    • "generate": để model sinh text, sau đó parse nhãn từ output.
- Cung cấp hàm generate_text(prompt) để dùng cho explain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover - môi trường chưa cài transformers/torch
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from .prompt_pool import PromptPool
from .types import PromptTemplate


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


class SemanticsLLM:
    """Lớp wrap để gọi LLM theo chuẩn của S-NAP."""

    def __init__(
        self,
        prompt_pool: PromptPool,
        runtime_cfg: RuntimeConfig,
        gen_cfg: Optional[GenConfig] = None,
        mode: str = "logprob",
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
        """
        self.prompt_pool = prompt_pool
        self.runtime_cfg = runtime_cfg
        self.gen_cfg = gen_cfg or GenConfig()
        if mode not in {"logprob", "generate"}:
            raise ValueError("mode phải là 'logprob' hoặc 'generate'")
        self.mode = mode

        if torch is None or AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError(
                "SemanticsLLM cần 'torch' và 'transformers'. "
                "Hãy cài đặt: pip install torch transformers"
            )

        # Thiết lập device
        if runtime_cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = runtime_cfg.device

        # ----- Load tokenizer -----
        self.tokenizer = AutoTokenizer.from_pretrained(
            runtime_cfg.model_name,
            use_fast=True,
        )
        # Nếu model không có pad_token, dùng eos_token
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ----- Load model với chiến lược nhiều tầng -----
        self.model = self._load_model_with_fallback()

        self.model.eval()

    # ------------------------------------------------------------------
    # Chiến lược load model: 4bit + GPU -> fallback CPU nếu lỗi
    # ------------------------------------------------------------------
    def _load_model_with_fallback(self):
        """
        Thử lần lượt:
        1) Nếu load_in_4bit=True và có GPU: load 4bit + device_map='auto'.
        2) Nếu fail (thiếu VRAM / cấu hình bitsandbytes):
            -> in cảnh báo, fallback về CPU full-precision.
        """
        model_name = self.runtime_cfg.model_name

        # Ưu tiên: 4bit + GPU (nếu được yêu cầu và có CUDA)
        if self.runtime_cfg.load_in_4bit and self.device == "cuda":
            model_kwargs: Dict[str, Any] = {}
            try:
                from transformers import BitsAndBytesConfig  # type: ignore

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs["quantization_config"] = quant_config
                model_kwargs["device_map"] = "auto"

                print(
                    f"[SemanticsLLM] Đang thử load model 4-bit trên GPU: {model_name} "
                    f"(device_map=auto)..."
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs,
                )
                # Với device_map="auto" thì không cần .to(...)
                print("[SemanticsLLM] Load 4-bit GPU thành công.")
                return model

            except Exception as e:
                # Nếu lỗi (thiếu VRAM, cấu hình offload, bitsandbytes, ...) -> fallback CPU
                print(
                    f"[SemanticsLLM] Không load được model 4-bit trên GPU, "
                    f"fallback CPU fp32. Lý do: {e}"
                )

        # Fallback: CPU full-precision (hoặc GPU nếu device='cuda' nhưng không 4bit)
        print(f"[SemanticsLLM] Đang load model full-precision trên device={self.device}: {model_name}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(self.device)
            print("[SemanticsLLM] Load full-precision thành công.")
            return model
        except Exception as e:
            # Nếu cả hai cách đều fail -> raise để bên ngoài xử lý
            raise RuntimeError(f"Không thể load model {model_name} trên device={self.device}: {e}")

    # ------------------------------------------------------------------
    # API chính: chọn nhãn từ top-k candidate
    # ------------------------------------------------------------------
    def predict_from_candidates(
        self,
        activities: Sequence[str],
        prefix: Sequence[str],
        candidates: List[str],
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
        """
        if not candidates:
            return "", {
                "error": "empty_candidates",
                "mode": self.mode,
            }

        template: PromptTemplate = self.prompt_pool.get_prediction_template()
        prompt = self._render_prediction_prompt(
            template=template,
            activities=activities,
            prefix=prefix,
            candidates=candidates,
        )

        if self.mode == "logprob":
            cand_logprobs = self._score_candidates_logprob(prompt, candidates)
            # Chọn candidate có logprob lớn nhất
            chosen = max(cand_logprobs.items(), key=lambda x: x[1])[0]
            meta: Dict[str, Any] = {
                "prompt_id": template.id,
                "raw_prompt": prompt,
                "mode": "logprob",
                "candidate_logprobs": cand_logprobs,
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
        Sinh text từ prompt (dùng cho explain hoặc debug).

        Parameters
        ----------
        prompt:
            Chuỗi prompt hoàn chỉnh.
        max_new_tokens:
            Nếu None → dùng self.gen_cfg.max_new_tokens.
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
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Chỉ lấy phần mới sinh sau prompt
        generated_ids = output_ids[0][input_ids.shape[1] :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()

    # ------------------------------------------------------------------
    # Render prompt prediction
    # ------------------------------------------------------------------
    def _render_prediction_prompt(
        self,
        template: PromptTemplate,
        activities: Sequence[str],
        prefix: Sequence[str],
        candidates: List[str],
    ) -> str:
        """
        Render prompt prediction từ PromptTemplate.

        Quy ước string:
        - {activities}: liệt kê trong ngoặc nhọn, ngăn cách bởi dấu phẩy.
        - {prefix}: liệt kê theo thứ tự, ngăn cách bởi " -> ".
        - {candidates}: liệt kê trong ngoặc nhọn, ngăn cách bởi dấu phẩy.
        """
        activities_str = "{ " + ", ".join(map(str, activities)) + " }" if activities else "{}"
        prefix_str = " -> ".join(map(str, prefix)) if prefix else "<EMPTY_PREFIX>"
        candidates_str = "{ " + ", ".join(map(str, candidates)) + " }"

        return template.template.format(
            activities=activities_str,
            prefix=prefix_str,
            candidates=candidates_str,
        )

    # ------------------------------------------------------------------
    # Mode logprob: chấm điểm từng candidate
    # ------------------------------------------------------------------
    def _score_candidates_logprob(
        self,
        prompt: str,
        candidates: List[str],
    ) -> Dict[str, float]:
        """
        Ước lượng log-xác suất cho từng candidate:

        log P(candidate | prompt) ≈ - loss(candidate_tokens) * len(candidate_tokens)

        Cách làm (xấp xỉ):
        - full_text = prompt + " " + candidate
        - tokenize full_text → input_ids
        - tokenize candidate → cand_ids
        - labels = input_ids.clone(); set các token thuộc phần prompt = -100
        - model(input_ids, labels=labels) → loss (mean per token)
        - logprob ≈ -loss * len(candidate_tokens)
        """
        cand_logprobs: Dict[str, float] = {}
        if torch is None:
            return {c: 0.0 for c in candidates}

        for cand in candidates:
            cand_text = str(cand)
            full = prompt + "\n" + cand_text

            enc = self.tokenizer(
                full,
                return_tensors="pt",
                truncation=True,
                max_length=self.runtime_cfg.max_seq_len,
            )
            input_ids = enc["input_ids"].to(self.device)

            # Tokenize candidate riêng để ước lượng độ dài phần candidate
            cand_ids = self.tokenizer(
                cand_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            cand_len = cand_ids.shape[0]

            # Giả định candidate nằm ở cuối chuỗi (thực tế hầu hết đúng)
            labels = input_ids.clone()
            # mask toàn bộ token phần prompt
            if labels.shape[1] > cand_len:
                labels[:, :-cand_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=labels)
                loss = outputs.loss  # cross-entropy mean trên các token được tính

            # Tổng negative log-likelihood xấp xỉ
            neg_loglik = loss.item() * max(cand_len, 1)
            cand_logprobs[cand] = -neg_loglik

        return cand_logprobs

    # ------------------------------------------------------------------
    # Mode generate: parse nhãn từ output
    # ------------------------------------------------------------------
    def _parse_label_from_output(self, raw_output: str, candidates: List[str]) -> str:
        """
        Cố gắng trích nhãn từ output sinh ra:

        Chiến lược đơn giản:
        - Lấy dòng đầu tiên không rỗng.
        - Strip khoảng trắng, dấu hai chấm, dấu chấm,...
        - Nếu đúng bằng 1 trong các candidate → dùng luôn.
        - Nếu không khớp hoàn toàn:
            + thử so sánh lower-case / strip.
            + nếu vẫn không tìm thấy → fallback: candidate[0].
        """
        if not raw_output:
            return candidates[0]

        lines = [ln.strip() for ln in raw_output.splitlines() if ln.strip()]
        first = lines[0] if lines else raw_output.strip()

        # Loại bỏ prefix kiểu "Answer:" hoặc "The next activity is"
        for prefix in ["Answer:", "answer:", "Next activity:", "The next activity is"]:
            if first.lower().startswith(prefix.lower()):
                first = first[len(prefix) :].strip()

        # Bỏ các dấu câu cuối
        first_clean = first.strip().strip(".:;,!").strip()

        # Thử match trực tiếp
        if first_clean in candidates:
            return first_clean

        # Thử match lower-case
        low_map = {c.lower(): c for c in candidates}
        if first_clean.lower() in low_map:
            return low_map[first_clean.lower()]

        # Nếu output chứa candidate như một từ con
        for c in candidates:
            if c in first:
                return c

        # Fallback
        return candidates[0]


__all__ = ["SemanticsLLM", "RuntimeConfig", "GenConfig"]
