"""
LLaVA-OneVision-7B VLM compression pipeline.

Supports per-component (vision/projector/LLM) quantization.

LLaVA-OneVision structure:
    model.model.vision_tower — SiglipVisionModel
    model.model.multi_modal_projector — linear_1 + GELU + linear_2
    model.model.language_model — Qwen2 decoder
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

from transformers import AutoProcessor, set_seed
from src.stage.base import Execute
from src.t2c.convert_vlm import LlavaOnevision4Compress
from src.trainer.vlm.ptq import VLMPTQ
from src.data.vlm.calib import create_vlm_calib_loader
from vlm.evaluate import (
    POPEEvaluator, OCRBenchEvaluator,
    MMMUEvaluator, ChartQAEvaluator, SEEDBench2PlusEvaluator,
    TextVQAEvaluator, VizWizEvaluator,
)

parser = argparse.ArgumentParser(description='LLaVA-OneVision VLM compression')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()


class CompressLlavaOnevision(Execute):
    # accuracy_result/{RESULT_FAMILY}/{format}/{bench}.json — overridden by
    # architecture-compatible variants (e.g. VARCO-VISION-14B).
    RESULT_FAMILY = "llava-onevision"

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

        # Check if all quantizers are identity (FP16 baseline — skip conversion)
        all_identity = all(
            qconfig.get(c, {}).get("wqtype", "identity") == "identity" and
            qconfig.get(c, {}).get("xqtype", "identity") == "identity"
            for c in ("vision", "projector", "llm")
        )

        if all_identity:
            self.model = model
            self.logger.info(f"FP16 baseline — skipping model conversion.")
        else:
            converter = LlavaOnevision4Compress(
                model=model,
                vision_config=vision_config,
                projector_config=projector_config,
                llm_config=llm_config,
            )
            self.model = converter.convert()

            # Strip weight masks before GPU transfer to save ~14 GB
            # (_QBaseLinear.trainFunc does not use the mask buffer)
            from src.module.base import _QBaseLinear
            mask_count = 0
            for m in self.model.modules():
                if isinstance(m, _QBaseLinear) and 'mask' in m._buffers:
                    del m._buffers['mask']
                    mask_count += 1

            self.model = self.model.half().to(self.device)
            self.logger.info(f"LLaVA-OneVision model converted (stripped {mask_count} masks). Components ready for PTQ.")

        # Create calibration data loader
        num_samples = llm_config.get("num_samples", 128)
        self.calib_loader = create_vlm_calib_loader(
            processor=self.processor,
            num_samples=num_samples,
            batch_size=1,
            model_family="llava_onevision",
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
        """Run PTQ on all VLM components."""
        # LLaVA-OneVision structure (wrapped in LlavaOnevisionForConditionalGeneration):
        #   model.vision_tower — SiglipVisionModel
        #   model.multi_modal_projector — MLP projector
        #   model.language_model — Qwen2 decoder
        vision_module = self.model.vision_tower
        projector_module = self.model.multi_modal_projector
        llm_module = self.model.language_model

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
                model_family="llava_onevision",
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
        run_dir_name = os.path.basename(os.path.normpath(self.run_dir))
        parts = run_dir_name.split("-")
        # size token: a part like "7b"/"14b" (digits + trailing 'b'); "mix4.5b"
        # is excluded because "mix4.5" is not all-digits.
        size_token = next((p for p in parts if p.endswith("b") and p[:-1].isdigit()), None)
        if size_token is not None and parts.index(size_token) + 1 < len(parts) - 1:
            format_name = "-".join(parts[parts.index(size_token) + 1 : -1])
        else:
            format_name = "default"
        if not format_name:
            format_name = "fp16"
        result_dir = os.path.join("accuracy_result", self.RESULT_FAMILY, format_name)
        os.makedirs(result_dir, exist_ok=True)
        for bench_name, bench_result in all_results.items():
            save_data = {k: v for k, v in bench_result.items() if k != "results"}
            save_data["elapsed_seconds"] = bench_timings.get(bench_name, 0)
            result_path = os.path.join(result_dir, f"{bench_name}.json")
            with open(result_path, "w") as f:
                json.dump(save_data, f, indent=2)
            self.logger.info(f"Results saved to {result_path}")

        return all_results

    def _is_identity_only(self):
        """Check if all quantizers are identity (FP16 baseline)."""
        qconfig = self.config["quantization"]
        for component in ("vision", "projector", "llm"):
            cfg = qconfig.get(component, {})
            if cfg.get("wqtype", "identity") != "identity":
                return False
            if cfg.get("xqtype", "identity") != "identity":
                return False
        return True

    def run(self):
        import gc
        if self._is_identity_only():
            self.logger.info("All quantizers are identity — skipping PTQ and freeze for FP16 baseline.")
        else:
            fake_quant_model = self.ptq()

            # Freeze quantized weights
            from src.module.base import freeze_model_weights
            num_frozen = freeze_model_weights(self.model)
            self.logger.info(f"Frozen {num_frozen} quantized linear/conv layers")

        # Defragment CUDA memory: move model to CPU, clear GPU, move back
        del self.calib_loader
        del self.task
        self.calib_loader = None
        self.task = None
        self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        self.model.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()

        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        self.logger.info(f"Post-defrag GPU memory: {gpu_mem:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")

        self.evaluate()


def starter():
    executor = CompressLlavaOnevision(args.config_dir)
    executor.run()


if __name__ == "__main__":
    starter()
