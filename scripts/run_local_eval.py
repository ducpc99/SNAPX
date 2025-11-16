# scripts/run_local_eval.py
# -------------------------
# Chạy toàn bộ pipeline S-NAPX Hybrid ở chế độ local, hỗ trợ gộp nhiều file YAML cấu hình
# và dùng kiến trúc mới (HybridSNAP + Explain + Cost-Performance + Guard).

from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Cho phép import src.* dù script nằm ngoài thư mục src
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Utils config
from src.utils.config import load_config

# Loader chuẩn (S-NAP, A-SAD, T-SAD + split train/val/test)
from src.data.loader import load_task_csv

# Core models
from src.sequence.ensemble import SequenceEnsemble
from src.graph_reasoning.dfg import DFG
from src.graph_reasoning.reasoner import GraphReasoner, GraphReasonerConfig
from src.graph_reasoning.semantics_prior import SemanticsPrior, SemanticsPriorConfig
from src.semantics.prompt_pool import PromptPool, PromptPoolConfig
from src.semantics.inference import SemanticsLLM, RuntimeConfig, GenConfig
from src.hybrid.pipeline import HybridSNAP, HybridConfig

# Guard
from src.guard.config import GuardConfig
from src.guard.core import Guard

# Eval
from src.eval.runner import run_evaluation, EvalSample


# =========================
# GPU monitor (tuỳ chọn)
# =========================
def _gpu_snapshot() -> str:
    """
    Cố gắng lấy thông tin GPU/VRAM theo thứ tự ưu tiên:
    1) pynvml (ổn định nhất)
    2) nvidia-smi (subprocess)
    3) torch.cuda.mem_get_info (cuối cùng)
    """
    # 1) pynvml
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(h).decode("utf-8")
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        txt = f"{name} | util={util:>3}% | VRAM={mem.used//(1024**2)}/{mem.total//(1024**2)} MiB"
        pynvml.nvmlShutdown()
        return txt
    except Exception:
        pass

    # 2) nvidia-smi
    try:
        import subprocess, shlex

        cmd = "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.DEVNULL, text=True, timeout=2)
        name, util, used, total = [x.strip() for x in out.split(",")]
        util = util if util != "[Unknown Error]" else "?"
        return f"{name} | util={util}% | VRAM={used}/{total} MiB"
    except Exception:
        pass

    # 3) torch (free/total)
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            name = torch.cuda.get_device_name(0)
            used = (total - free) // (1024**2)
            total_mb = total // (1024**2)
            return f"{name} | VRAM≈{used}/{total_mb} MiB"
    except Exception:
        pass

    return "GPU monitor unavailable"


def _gpu_monitor_loop(stop_flag, interval_sec: float = 2.0) -> None:
    while not stop_flag.is_set():
        print(f"[GPU] {_gpu_snapshot()}")
        stop_flag.wait(interval_sec)


# =========================
# Print config summary (schema mới)
# =========================
def _print_cfg_summary(cfg: Dict[str, Any]) -> None:
    dataset = cfg.get("dataset", {}) or {}
    data = cfg.get("data", {}) or {}
    seq = cfg.get("sequence", {}) or {}
    graph = cfg.get("graph", {}) or {}
    sem = cfg.get("semantics", {}) or {}
    semp = cfg.get("semantics_prior", {}) or {}
    guard = cfg.get("guard", {}) or {}
    hybrid = cfg.get("hybrid", {}) or {}
    evl = cfg.get("eval", {}) or {}
    paths = cfg.get("paths", {}) or {}

    print("⚙️  Cấu hình tóm tắt")
    print("   ├─ run_name    :", cfg.get("run_name"))
    print("   ├─ task        :", cfg.get("task", "next_activity"))
    print("   ├─ dataset     :", dataset.get("path"))
    print("   ├─ split_file  :", dataset.get("split_file"))
    print("   ├─ case_id_col :", dataset.get("case_id_col", "model_id"))
    print("   ├─ prefix_col  :", dataset.get("prefix_col", "prefix"))
    print("   ├─ label_col   :", dataset.get("label_col", "next"))
    print("   ├─ prefix_sep  :", dataset.get("prefix_sep", ";"))
    print(
        "   ├─ data        :",
        f"drop_end={data.get('drop_end')}, invert_labels={data.get('invert_labels')}, "
        f"min_activities={data.get('min_activities')}, load_limit={data.get('load_limit')}",
    )
    print("   ├─ output_dir  :", paths.get("output_dir", "outputs"))
    print(
        "   ├─ sequence    :",
        f"alpha={seq.get('alpha',0.5)}, "
        f"markov_order={seq.get('markov_order',2)}, min_count={seq.get('min_count',1)}",
    )
    print(
        "   ├─ graph       :",
        f"enabled={graph.get('enabled',True)}, "
        f"dfg(min_count={graph.get('dfg',{}).get('min_count',1)}, "
        f"min_ratio={graph.get('dfg',{}).get('min_ratio',0.0)}), "
        f"reasoner={graph.get('reasoner',{})}",
    )
    print(
        "   ├─ semantics   :",
        f"use_llm={sem.get('use_llm',False)}, model={sem.get('model_name')}, "
        f"4bit={sem.get('load_in_4bit',False)}, max_len={sem.get('max_seq_len',2048)}, mode={sem.get('mode','logprob')}",
    )
    print("   ├─ sem_prior   :", semp)
    print("   ├─ guard       :", guard)
    print("   ├─ hybrid      :", hybrid)
    print(
        "   └─ eval        :",
        f"num_samples={evl.get('num_samples')}, ndcg_k={evl.get('ndcg_k',5)}, "
        f"measure_cost={evl.get('measure_cost',False)}",
    )
    print()


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Local Evaluation for S-NAPX Hybrid")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        nargs="+",
        required=True,
        help="Đường dẫn tới 1 hoặc nhiều file YAML cấu hình (vd: config/base.yaml config/local_eval.yaml)",
    )
    parser.add_argument(
        "--monitor-gpu",
        action="store_true",
        help="In trạng thái GPU/VRAM mỗi vài giây trong khi chạy",
    )
    parser.add_argument(
        "--gpu-interval",
        type=float,
        default=2.0,
        help="Chu kỳ (giây) in trạng thái GPU khi bật --monitor-gpu",
    )
    parser.add_argument(
        "--save-merged-cfg",
        action="store_true",
        help="Lưu bản config đã merge vào thư mục output để tái lập thí nghiệm",
    )
    args = parser.parse_args()

    cfg_paths = [str(Path(p)) for p in args.config]

    # 1) Load & merge YAML (dùng utils/config.py)
    cfg = load_config(cfg_paths)

    # 2) Chuẩn bị output dir
    paths_cfg = cfg.get("paths", {}) or {}
    run_name = cfg.get("run_name", "run_local")
    base_output = Path(paths_cfg.get("output_dir", "outputs"))
    out_dir = base_output / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🚀  Bắt đầu chạy S-NAPX Hybrid (Local Mode)...\n")
    _print_cfg_summary(cfg)

    # (tuỳ chọn) Lưu bản config đã merge để reproducible
    if args.save_merged_cfg:
        merged_cfg_path = out_dir / "run_config_merged.yaml"
        try:
            with open(merged_cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True)
            print(f"📝  Đã lưu merged config → {merged_cfg_path}")
        except Exception as e:
            print(f"[Warn] Không thể lưu merged config: {e}")

    # GPU monitor (tuỳ chọn)
    stop_flag = threading.Event()
    t_monitor = None
    if args.monitor_gpu:
        t_monitor = threading.Thread(
            target=_gpu_monitor_loop,
            args=(stop_flag, args.gpu_interval),
            daemon=True,
        )
        t_monitor.start()

    t0 = time.time()
    try:
        # ==============================
        # 3) Load dữ liệu bằng load_task_csv (S-NAP + split train/test)
        # ==============================
        dataset_cfg = cfg.get("dataset", {}) or {}
        data_cfg = cfg.get("data", {}) or {}
        eval_cfg = cfg.get("eval", {}) or {}
        task = cfg.get("task", "next_activity")

        dataset_path = dataset_cfg.get("path")
        split_file = dataset_cfg.get("split_file")
        if not dataset_path:
            raise ValueError("dataset.path chưa được cấu hình trong YAML.")

        load_limit = data_cfg.get("load_limit", None)
        drop_end = data_cfg.get("drop_end", True)
        invert_labels = data_cfg.get("invert_labels", False)
        min_acts = data_cfg.get("min_activities", 2)

        ret = load_task_csv(
            task=task,
            dataset_path=dataset_path,
            limit=load_limit,
            drop_end=drop_end,
            invert_labels=invert_labels,
            min_activities=min_acts,
            split_file=split_file,
            return_splits_if_available=True,
        )

        # Nếu có split_file hợp lệ: ret là dict {'train','val','test'}
        if isinstance(ret, dict):
            df_train = ret["train"]
            df_test = ret["test"]
            df_val = ret.get("val")
        else:
            df_train = df_test = ret
            df_val = None

        print(f"📦  Loaded train={len(df_train)}, test={len(df_test)} từ {dataset_path}")

        # Eval trên test (có thể giới hạn bằng eval.num_samples)
        num_samples = eval_cfg.get("num_samples")
        if isinstance(num_samples, int) and 0 < num_samples < len(df_test):
            df_eval = df_test.head(num_samples)
            print(f"🔍  Giới hạn xuống {len(df_eval)} mẫu test để eval nhanh.")
        else:
            df_eval = df_test

        # Tạo EvalSample từ df_eval
        samples: List[EvalSample] = []
        for _, row in df_eval.iterrows():
            pref = list(row["prefix"])
            y_true = str(row["next"])
            mid = str(row.get("model_id", ""))
            rid = str(row.get("revision_id", "")) if "revision_id" in row else ""
            case_id = f"{mid}_{rid}" if rid != "" else mid
            samples.append(EvalSample(case_id=case_id, prefix=pref, y_true=y_true))

        if not samples:
            raise ValueError("Không có sample nào sau khi tạo EvalSample.")

        print(f"🔍  Eval samples: {len(samples)}")

        # Traces cho Sequence & DFG lấy từ TRAIN (đúng methodology)
        traces_for_seq: List[List[str]] = []
        for _, row in df_train.iterrows():
            pref = list(row["prefix"])
            nxt = str(row["next"])
            if pref:
                traces_for_seq.append(pref + [nxt])
            else:
                traces_for_seq.append([nxt])

        activities_vocab = sorted({act for tr in traces_for_seq for act in tr})
        print(f"🧩  Num activities in vocab: {len(activities_vocab)}")

        # =======================================
        # 4) SequenceEnsemble + DFG + Reasoner
        # =======================================
        seq_cfg = cfg.get("sequence", {}) or {}
        seq_model = SequenceEnsemble(
            alpha=seq_cfg.get("alpha", 0.5),
            markov_order=seq_cfg.get("markov_order", 2),
            min_count=seq_cfg.get("min_count", 1),
        )
        seq_model.fit(traces_for_seq)
        print("✅  SequenceEnsemble fitted.")

        graph_cfg = cfg.get("graph", {}) or {}
        dfg_cfg = graph_cfg.get("dfg", {}) or {}
        dfg = DFG.from_traces(
            traces_for_seq,
            min_count=dfg_cfg.get("min_count", 1),
            min_ratio=dfg_cfg.get("min_ratio", 0.0),
        )
        print("✅  DFG built.")

        reasoner_cfg = graph_cfg.get("reasoner", {}) or {}
        graph_reasoner = GraphReasoner(
            dfg=dfg,
            cfg=GraphReasonerConfig(
                enabled=graph_cfg.get("enabled", True),
                hard_mask=reasoner_cfg.get("hard_mask", False),
                boost_factor=reasoner_cfg.get("boost_factor", 0.5),
                min_prob_keep=reasoner_cfg.get("min_prob_keep", 0.0),
            ),
        )

        # =======================
        # 5) Semantics-Prior
        # =======================
        import json

        semp_cfg = cfg.get("semantics_prior", {}) or {}
        sem_prior_conf = SemanticsPriorConfig(
            enabled=semp_cfg.get("enabled", False),
            mode=semp_cfg.get("mode", "activity"),
            default_weight=semp_cfg.get("default_weight", 1.0),
            lambda_mix=semp_cfg.get("lambda_mix", 0.2),
            floor=semp_cfg.get("floor", 0.7),
            ceil=semp_cfg.get("ceil", 1.3),
        )
        prior_map = {}
        prior_path = semp_cfg.get("prior_map_path")
        if prior_path:
            try:
                with open(prior_path, "r", encoding="utf-8") as f:
                    prior_map = json.load(f)
                print(f"✅  Loaded semantics prior map từ {prior_path}")
            except Exception as e:
                print(f"[Warn] Không đọc được prior_map từ {prior_path}: {e}")
        sem_prior = SemanticsPrior(cfg=sem_prior_conf, prior_map=prior_map)

        # =======================
        # 6) PromptPool + LLM
        # =======================
        prompt_pool = PromptPool(PromptPoolConfig())
        sem_cfg = cfg.get("semantics", {}) or {}
        use_llm = sem_cfg.get("use_llm", False)
        sem_llm = None

        if use_llm:
            print("🧠  Khởi tạo SemanticsLLM (có thể tốn thời gian / VRAM)...")
            runtime_cfg = RuntimeConfig(
                model_name=sem_cfg.get("model_name", "microsoft/Phi-3-mini-4k-instruct"),
                max_seq_len=sem_cfg.get("max_seq_len", 2048),
                load_in_4bit=sem_cfg.get("load_in_4bit", False),
                device=sem_cfg.get("device", "auto"),
            )
            gen_cfg = GenConfig(
                max_new_tokens=sem_cfg.get("gen", {}).get("max_new_tokens", 16),
                temperature=sem_cfg.get("gen", {}).get("temperature", 0.0),
                top_p=sem_cfg.get("gen", {}).get("top_p", 1.0),
                do_sample=sem_cfg.get("gen", {}).get("do_sample", False),
            )
            mode = sem_cfg.get("mode", "logprob")
            try:
                sem_llm = SemanticsLLM(
                    prompt_pool=prompt_pool,
                    runtime_cfg=runtime_cfg,
                    gen_cfg=gen_cfg,
                    mode=mode,
                )
                print("✅  SemanticsLLM ready trên device:", sem_llm.device)
            except Exception as e:
                print(f"[Warn] Không khởi tạo được SemanticsLLM, fallback sequence-only: {e}")
                sem_llm = None
                use_llm = False
        else:
            print("ℹ️  semantics.use_llm = False → chạy sequence + graph (không dùng LLM).")

        # =======================
        # 6b) Guard (footprint / anomaly) – FAIL-SOFT
        # =======================
        guard_cfg_raw = cfg.get("guard", {}) or {}
        guard = None
        guard_enabled_flag = guard_cfg_raw.get("enabled", False)

        if guard_enabled_flag:
            gcfg = GuardConfig(
                enabled=True,
                mode=guard_cfg_raw.get("mode", "soft"),
                use_activity_guard=guard_cfg_raw.get("use_activity_guard", True),
                use_trace_guard=guard_cfg_raw.get("use_trace_guard", False),
                penalty_factor=guard_cfg_raw.get("penalty_factor", 0.5),
                min_prefix_len=guard_cfg_raw.get("min_prefix_len", 1),
                activity_dataset=guard_cfg_raw.get("activity_dataset"),
                trace_dataset=guard_cfg_raw.get("trace_dataset"),
            )
            try:
                guard = Guard.from_datasets(gcfg)
                print("✅  Guard loaded từ datasets (activity/trace).")
            except Exception as e:
                # Không để Guard làm crash toàn bộ pipeline
                print(f"[Warn] Không build được Guard từ datasets, tạm thời tắt Guard: {e}")
                guard = None
                guard_enabled_flag = False
        else:
            print("ℹ️  guard.enabled = False → không dùng Guard.")

        # =======================
        # 7) HybridSNAP
        # =======================
        hybrid_cfg_raw = cfg.get("hybrid", {}) or {}
        hyb_cfg = HybridConfig(
            top_k=hybrid_cfg_raw.get("top_k", 5),
            enable_graph_reasoner=hybrid_cfg_raw.get("enable_graph_reasoner", True),
            enable_semantics_prior=hybrid_cfg_raw.get(
                "enable_semantics_prior",
                semp_cfg.get("enabled", False),
            ),
            # nếu YAML không ghi rõ, mặc định bám theo guard_enabled_flag
            enable_guard=hybrid_cfg_raw.get("enable_guard", guard_enabled_flag),
            enable_explanation=hybrid_cfg_raw.get("enable_explanation", False) and bool(sem_llm),
        )

        hybrid = HybridSNAP(
            cfg=hyb_cfg,
            seq_model=seq_model,
            graph_reasoner=graph_reasoner if hyb_cfg.enable_graph_reasoner else None,
            sem_prior=sem_prior if hyb_cfg.enable_semantics_prior else None,
            semantics_llm=sem_llm,
            guard=guard,
            prompt_pool=prompt_pool,
            activities_vocab=activities_vocab,
        )
        print("✅  HybridSNAP ready.")

        # =======================
        # 8) Evaluation
        # =======================
        ndcg_k = eval_cfg.get("ndcg_k", 5)
        measure_cost = eval_cfg.get("measure_cost", False)

        eval_result = run_evaluation(
            samples=samples,
            model=hybrid,
            activities_vocab=activities_vocab,
            measure_cost=measure_cost,
            ndcg_k=ndcg_k,
        )

    finally:
        if args.monitor_gpu:
            stop_flag.set()
            if t_monitor is not None:
                t_monitor.join(timeout=1)

    elapsed = time.time() - t0

    # =======================
    # 9) In & lưu kết quả
    # =======================
    print("\n📊  Kết quả tổng hợp:")
    print("----------------------")
    print(f"Accuracy     : {eval_result.accuracy:.4f}")
    print(f"Macro-F1     : {eval_result.macro_f1:.4f}")
    print(f"MRR          : {eval_result.mrr:.4f}")
    print(f"NDCG@{ndcg_k:<3}: {eval_result.ndcg_at_k:.4f}")

    print("\n🔎  Explain Metrics:")
    for k, v in eval_result.explain_metrics.items():
        print(f"{k:<25}: {v:.4f}")

    if eval_result.latency is not None:
        lat = eval_result.latency
        print("\n⏱️  Latency (ms):")
        print(f"avg_ms        : {lat.avg_ms:.2f}")
        print(f"p50_ms        : {lat.p50_ms:.2f}")
        print(f"p90_ms        : {lat.p90_ms:.2f}")
        print(f"p95_ms        : {lat.p95_ms:.2f}")

    if eval_result.cpr is not None:
        print(f"\n⚖️  CPR (Macro-F1 / avg_sec): {eval_result.cpr:.4f}")

    print("----------------------")
    print(f"⏱️  Thời gian chạy: {elapsed/60:.2f} phút ({elapsed:.1f} giây)")
    print(f"📂  Output dir   : {out_dir}\n")

    # Lưu results.yaml để sau này dùng cho báo cáo
    results_dict: Dict[str, Any] = {
        "accuracy": eval_result.accuracy,
        "macro_f1": eval_result.macro_f1,
        "mrr": eval_result.mrr,
        "ndcg_at_k": eval_result.ndcg_at_k,
        "explain_metrics": eval_result.explain_metrics,
        "latency_ms": None,
        "cpr": eval_result.cpr,
    }
    if eval_result.latency is not None:
        results_dict["latency_ms"] = {
            "avg_ms": eval_result.latency.avg_ms,
            "p50_ms": eval_result.latency.p50_ms,
            "p90_ms": eval_result.latency.p90_ms,
            "p95_ms": eval_result.latency.p95_ms,
        }

    results_path = out_dir / "results.yaml"
    try:
        with open(results_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(results_dict, f, allow_unicode=True)
        print(f"📝  Đã lưu kết quả chi tiết → {results_path}")
    except Exception as e:
        print(f"[Warn] Không thể lưu results.yaml: {e}")


if __name__ == "__main__":
    main()
