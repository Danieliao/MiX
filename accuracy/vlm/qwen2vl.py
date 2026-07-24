"""
Qwen2-VL VLM compression pipeline.

Supports per-component (vision/projector/LLM) quantization.
Qwen2-VL architecture:
  - model.visual — ViT blocks + PatchMerger (acts as projector)
  - model.model — Qwen2 decoder layers
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

from transformers import AutoProcessor, set_seed
from src.stage.base import Execute
from src.t2c.convert_vlm import Qwen2VL4Compress
from src.trainer.vlm.ptq import VLMPTQ
from src.data.vlm.calib import create_vlm_calib_loader
from vlm.evaluate import (
    POPEEvaluator, OCRBenchEvaluator,
    MMMUEvaluator, ChartQAEvaluator, SEEDBench2PlusEvaluator,
    TextVQAEvaluator, VizWizEvaluator,
)

parser = argparse.ArgumentParser(description='Qwen2-VL VLM compression')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()


class CompressQwen2VL(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        set_seed(self.config["seed"])

        model_type = self.config["model"]["model_type"]

        # Load model
        model = self.create_model()
        self.processor = AutoProcessor.from_pretrained(model_type, trust_remote_code=True)

        # Per-component quantization configs
        qconfig = self.config["quantization"]
        vision_config = qconfig.get("vision", {})
        projector_config = qconfig.get("projector", {})
        llm_config = qconfig.get("llm", {})

        # Convert model to quantized modules
        converter = Qwen2VL4Compress(
            model=model,
            vision_config=vision_config,
            projector_config=projector_config,
            llm_config=llm_config,
        )
        self.model = converter.convert()
        self.model = self.model.half().to(self.device)

        self.logger.info(f"Qwen2-VL model converted. Components ready for PTQ.")

        # Create calibration data loader
        num_samples = llm_config.get("num_samples", 128)
        self.calib_loader = create_vlm_calib_loader(
            processor=self.processor,
            num_samples=num_samples,
            batch_size=1,
            model_family="qwen2_vl",
        )

        # Setup PTQ
        smooth_config = self.config.get("smooth", {})
        self.task = VLMPTQ(
            model=self.model,
            calib_loader=self.calib_loader,
            vision_config=vision_config,
            projector_config=projector_config,
            llm_config=llm_config,
            logger=self.logger,
            run_dir=self.run_dir,
            smooth_alpha=smooth_config.get("alpha", 0.85),
            smooth_flag=smooth_config.get("flag", True),
        )

    def register_run_dir(self):
        super().register_run_dir()
        self.t2c_dir = os.path.join(self.run_dir, "t2c")
        self.tensors_dir = os.path.join(self.t2c_dir, "tensors")
        if not os.path.isdir(self.tensors_dir):
            os.makedirs(self.tensors_dir, exist_ok=True)

    def ptq(self):
        """Run PTQ on all Qwen2-VL components."""
        # Qwen2-VL structure:
        #   model.visual — ViT blocks + PatchMerger
        #   model.model — Qwen2 decoder
        vision_module = self.model.visual
        projector_module = None  # Qwen2-VL's PatchMerger is inside visual — no separate projector
        llm_module = self.model.model

        fake_quantized_model = self.task.run(
            vision_module=vision_module,
            projector_module=projector_module,
            llm_module=llm_module,
        )
        return fake_quantized_model

    @torch.inference_mode()
    def evaluate(self):
        """Run evaluation benchmark(s) specified in config."""
        self.model.eval()

        eval_config = self.config.get("eval", {})
        max_samples = eval_config.get("max_samples", None)
        dryrun = eval_config.get("dryrun", False)
        benchmarks = eval_config.get("benchmarks", ["pope"])

        if isinstance(benchmarks, str):
            benchmarks = [benchmarks]

        if dryrun:
            max_samples = 10

        all_results = {}
        bench_timings = {}
        for bench in benchmarks:
            bench_start = time.time()
            evaluator_kwargs = dict(
                model=self.model,
                processor=self.processor,
                device=self.device,
                logger=self.logger,
                model_family="qwen2_vl",
            )
            if bench == "pope":
                self.logger.info(f"Running POPE evaluation (max_samples={max_samples})...")
                evaluator = POPEEvaluator(**evaluator_kwargs)
                all_results["pope"] = evaluator.run(max_samples=max_samples)
            elif bench == "ocrbench":
                self.logger.info(f"Running OCRBench evaluation (max_samples={max_samples})...")
                evaluator = OCRBenchEvaluator(**evaluator_kwargs)
                all_results["ocrbench"] = evaluator.run(max_samples=max_samples)
            elif bench == "mmmu":
                self.logger.info(f"Running MMMU evaluation (max_samples={max_samples})...")
                evaluator = MMMUEvaluator(**evaluator_kwargs)
                all_results["mmmu"] = evaluator.run(max_samples=max_samples)
            elif bench == "chartqa":
                self.logger.info(f"Running ChartQA evaluation (max_samples={max_samples})...")
                evaluator = ChartQAEvaluator(**evaluator_kwargs)
                all_results["chartqa"] = evaluator.run(max_samples=max_samples)
            elif bench == "seedbench2plus":
                self.logger.info(f"Running SEED-Bench-2-Plus evaluation (max_samples={max_samples})...")
                evaluator = SEEDBench2PlusEvaluator(**evaluator_kwargs)
                all_results["seedbench2plus"] = evaluator.run(max_samples=max_samples)
            elif bench == "textvqa":
                self.logger.info(f"Running TextVQA evaluation (max_samples={max_samples})...")
                evaluator = TextVQAEvaluator(**evaluator_kwargs)
                all_results["textvqa"] = evaluator.run(max_samples=max_samples)
            elif bench == "vizwiz":
                self.logger.info(f"Running VizWiz evaluation (max_samples={max_samples})...")
                evaluator = VizWizEvaluator(**evaluator_kwargs)
                all_results["vizwiz"] = evaluator.run(max_samples=max_samples)
            else:
                self.logger.warning(f"Unknown benchmark: {bench}")
                continue
            bench_timings[bench] = time.time() - bench_start
            self.logger.info(f"{bench} completed in {bench_timings[bench]:.1f}s")

        # Save results as JSON
        import json
        # Derive result directory from config_dir path (more reliable than parsing run_dir)
        # Config path: config/qwen2vl-2b/{format}/{model}-{format}-{bench}.yaml
        #          or: config/qwen2vl/{format}/{model}-{format}-{bench}.yaml
        config_path = self.config_dir if hasattr(self, 'config_dir') else ""
        config_parts = config_path.replace("\\", "/").split("/")
        if "k-smooth-static" in config_parts:
            # K-smoothing ablation: config/k-smooth-static/{model}/{format}/{file}.yaml
            # -> accuracy_result/k-smooth-static/{model}/{format}/{bench}.json
            model_dir = config_parts[config_parts.index("k-smooth-static") + 1]
            result_subdir = os.path.join("k-smooth-static", model_dir)
            format_name = config_parts[-2]
        elif len(config_parts) >= 3 and config_parts[-3].startswith("qwen2vl"):
            result_subdir = config_parts[-3]  # "qwen2vl" or "qwen2vl-2b"
            format_name = config_parts[-2]     # e.g. "mxint4g16-mix4.5b-l-flip"
        else:
            # Fallback: parse from run_dir
            run_dir_name = os.path.basename(os.path.normpath(self.run_dir))
            parts = run_dir_name.split("-")
            model_size = None
            size_idx = None
            for sz in ("2b", "7b"):
                if sz in parts:
                    model_size = sz
                    size_idx = parts.index(sz)
                    break
            if size_idx is not None:
                format_name = "-".join(parts[size_idx + 1 : -1])
            else:
                format_name = "default"
            if not format_name:
                format_name = "fp16"
            result_subdir = "qwen2vl-2b" if model_size == "2b" else "qwen2vl"
        result_dir = os.path.join("accuracy_result", result_subdir, format_name)
        os.makedirs(result_dir, exist_ok=True)
        for bench_name, bench_result in all_results.items():
            save_data = {k: v for k, v in bench_result.items() if k != "results"}
            save_data["elapsed_seconds"] = bench_timings.get(bench_name, 0)
            result_path = os.path.join(result_dir, f"{bench_name}.json")
            with open(result_path, "w") as f:
                json.dump(save_data, f, indent=2)
            self.logger.info(f"Results saved to {result_path}")

        return all_results

    def run(self):
        fake_quant_model = self.ptq()

        # Free calibration data to reclaim GPU memory before freeze
        import gc
        if hasattr(self, 'task'):
            if hasattr(self.task, 'calib_loader'):
                del self.task.calib_loader
            del self.task
        gc.collect()
        torch.cuda.empty_cache()

        # Freeze quantized weights
        from src.module.base import freeze_model_weights
        num_frozen = freeze_model_weights(self.model)
        self.logger.info(f"Frozen {num_frozen} quantized linear/conv layers")

        self.evaluate()


def starter():
    executor = CompressQwen2VL(args.config_dir)
    executor.run()


if __name__ == "__main__":
    starter()
