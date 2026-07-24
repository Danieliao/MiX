"""
Qwen2.5-VL VLM compression pipeline (3B / 7B / 32B).

Supports per-component (vision/LLM) quantization with different quantizers for
each component.

Qwen2.5-VL structure (nested, like Qwen3-VL):
    model.model.visual — ViT blocks (window attention) + merger
    model.model.language_model — Qwen2.5 text decoder
No separate projector (merger is part of visual).
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

from transformers import AutoProcessor, set_seed
from src.stage.base import Execute
from src.t2c.convert_vlm import Qwen2_5_VL4Compress
from src.trainer.vlm.ptq import VLMPTQ
from src.data.vlm.calib import create_vlm_calib_loader
from vlm.evaluate import (
    POPEEvaluator, OCRBenchEvaluator,
    MMMUEvaluator, ChartQAEvaluator, SEEDBench2PlusEvaluator,
    TextVQAEvaluator, VizWizEvaluator,
)

parser = argparse.ArgumentParser(description='Qwen2.5-VL VLM compression')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

MODEL_FAMILY = "qwen2_5_vl"


class CompressQwen2_5VL(Execute):
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
        # Qwen2.5-VL has no separate projector; merger is part of visual.
        # projector_config kept for API compatibility but unused.
        projector_config = qconfig.get("projector", {})
        llm_config = qconfig.get("llm", {})

        # Convert model to quantized modules
        converter = Qwen2_5_VL4Compress(
            model=model,
            vision_config=vision_config,
            projector_config=projector_config,
            llm_config=llm_config,
        )
        self.model = converter.convert()
        self.model = self.model.to(self.device)

        self.logger.info(f"Model converted. VLM components ready for PTQ.")

        # Create calibration data loader
        num_samples = llm_config.get("num_samples", 128)
        self.calib_loader = create_vlm_calib_loader(
            processor=self.processor,
            num_samples=num_samples,
            batch_size=1,
            model_family=MODEL_FAMILY,
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
        """Run PTQ on all VLM components (nested Qwen2.5-VL structure)."""
        vision_module = self.model.model.visual
        projector_module = None  # merger is part of visual, quantized there
        llm_module = self.model.model.language_model

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

        evaluator_map = {
            "pope": POPEEvaluator,
            "ocrbench": OCRBenchEvaluator,
            "mmmu": MMMUEvaluator,
            "chartqa": ChartQAEvaluator,
            "seedbench2plus": SEEDBench2PlusEvaluator,
            "textvqa": TextVQAEvaluator,
            "vizwiz": VizWizEvaluator,
        }

        all_results = {}
        bench_timings = {}
        for bench in benchmarks:
            if bench not in evaluator_map:
                self.logger.warning(f"Unknown benchmark: {bench}")
                continue
            bench_start = time.time()
            self.logger.info(f"Running {bench} evaluation (max_samples={max_samples})...")
            evaluator = evaluator_map[bench](
                model=self.model,
                processor=self.processor,
                device=self.device,
                logger=self.logger,
                model_family=MODEL_FAMILY,
            )
            all_results[bench] = evaluator.run(max_samples=max_samples)
            bench_timings[bench] = time.time() - bench_start
            self.logger.info(f"{bench} completed in {bench_timings[bench]:.1f}s")

        # Save results as JSON — derive subdirectory from run_dir.
        # Pattern: qwen2.5-vl-{size}b-{format}-{benchmark}
        # e.g. "save/qwen2.5-vl-3b-fp16-mmmu/" → "qwen2.5vl-3b/fp16"
        import json
        run_dir_name = os.path.basename(os.path.normpath(self.run_dir))
        parts = run_dir_name.split("-")
        # size token: a part like "3b"/"7b"/"32b" (digits + trailing 'b').
        # "mix4.5b" is excluded because "mix4.5" is not all-digits.
        size_token = next((p for p in parts if p.endswith("b") and p[:-1].isdigit()), None)
        if size_token is not None and parts.index(size_token) + 1 < len(parts) - 1:
            format_name = "-".join(parts[parts.index(size_token) + 1 : -1])
        else:
            format_name = "default"
        family = f"qwen2.5vl-{size_token}" if size_token else "qwen2.5vl"
        result_dir = os.path.join("accuracy_result", family, format_name)
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

        # Freeze quantized weights: bake fake-quantized values in-place
        from src.module.base import freeze_model_weights
        num_frozen = freeze_model_weights(self.model)
        self.logger.info(f"Frozen {num_frozen} quantized linear/conv layers")

        self.evaluate()


def starter():
    executor = CompressQwen2_5VL(args.config_dir)
    executor.run()


if __name__ == "__main__":
    starter()
