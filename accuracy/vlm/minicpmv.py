"""
MiniCPM-V-2.6 VLM compression pipeline.

Supports per-component (vision/projector/LLM) quantization.

MiniCPM-V-2.6 structure:
    model.vpm — SiglipVisionModel (from timm)
    model.resampler — Cross-attention resampler (projector)
    model.llm — Qwen2ForCausalLM
"""

import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse

from transformers import AutoProcessor, AutoTokenizer, set_seed
from src.stage.base import Execute
from src.t2c.convert_vlm import MiniCPMV4Compress
from src.trainer.vlm.ptq import VLMPTQ
from src.data.vlm.calib import create_vlm_calib_loader
from vlm.evaluate import (
    POPEEvaluator, OCRBenchEvaluator,
    MMMUEvaluator, ChartQAEvaluator, SEEDBench2PlusEvaluator,
    TextVQAEvaluator, VizWizEvaluator,
)

parser = argparse.ArgumentParser(description='MiniCPM-V-2.6 VLM compression')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()


class CompressMiniCPMV(Execute):
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
            self.model = model.half().to(self.device)
            self.model.vpm = self.model.vpm.to(self.device)
            self.model.resampler = self.model.resampler.to(self.device)
            self.logger.info(f"FP16 baseline — skipping model conversion.")
        else:
            converter = MiniCPMV4Compress(
                model=model,
                vision_config=vision_config,
                projector_config=projector_config,
                llm_config=llm_config,
            )
            self.model = converter.convert()

            # Strip weight masks before GPU transfer to save memory
            from src.module.base import _QBaseLinear
            mask_count = 0
            for m in self.model.modules():
                if isinstance(m, _QBaseLinear) and 'mask' in m._buffers:
                    del m._buffers['mask']
                    mask_count += 1

            self.model = self.model.half().to(self.device)
            self.logger.info(f"MiniCPM-V model converted (stripped {mask_count} masks). Components ready for PTQ.")

        # MiniCPM-V API compatibility wrappers:
        # 1. forward() expects forward(data={...}) but calibration calls model(**batch)
        # 2. generate() requires tokenizer arg and returns only new tokens
        _tokenizer = self.processor.tokenizer

        _orig_forward = self.model.forward
        def _forward_compat(**kwargs):
            # Generate position_ids if not provided (needed by MiniCPM-V forward)
            if "position_ids" not in kwargs and "input_ids" in kwargs:
                seq_len = kwargs["input_ids"].shape[1]
                kwargs["position_ids"] = torch.arange(seq_len, device=kwargs["input_ids"].device).unsqueeze(0).expand(kwargs["input_ids"].shape[0], -1)
            return _orig_forward(data=kwargs)
        self.model.forward = _forward_compat

        _orig_generate = self.model.generate
        def _generate_with_tokenizer(**kwargs):
            kwargs["tokenizer"] = _tokenizer
            kwargs.pop("image_sizes", None)
            input_ids = kwargs.get("input_ids")
            result = _orig_generate(**kwargs)
            # MiniCPM-V returns only generated tokens; prepend input_ids so evaluators
            # can strip them with output_ids[:, input_ids.shape[1]:]
            if input_ids is not None and result.dim() == 2:
                result = torch.cat([input_ids, result], dim=1)
            return result
        self.model.generate = _generate_with_tokenizer

        # Create calibration data loader
        num_samples = llm_config.get("num_samples", 128)
        self.calib_loader = create_vlm_calib_loader(
            processor=self.processor,
            num_samples=num_samples,
            batch_size=1,
            model_family="minicpm_v",
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
        """Run PTQ on all MiniCPM-V components."""
        vision_module = self.model.vpm
        projector_module = self.model.resampler
        llm_module = self.model.llm

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
                model_family="minicpm_v",
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

        # Save results as JSON — derive path from config_dir
        import json
        config_path = self.config_dir if hasattr(self, 'config_dir') else ""
        config_parts = config_path.replace("\\", "/").split("/")
        if "k-smooth-static" in config_parts:
            # K-smoothing ablation: config/k-smooth-static/{model}/{format}/{file}.yaml
            # -> accuracy_result/k-smooth-static/{model}/{format}/{bench}.json
            model_dir = config_parts[config_parts.index("k-smooth-static") + 1]
            result_subdir = os.path.join("k-smooth-static", model_dir)
            format_name = config_parts[-2]
        elif len(config_parts) >= 3 and config_parts[-3].startswith("minicpm"):
            result_subdir = config_parts[-3]
            format_name = config_parts[-2]
        else:
            # Fallback: parse from run_dir
            run_dir_name = os.path.basename(os.path.normpath(self.run_dir))
            parts = run_dir_name.split("-")
            # Remove "minicpm-v-" prefix, take format before last part (benchmark)
            format_name = "-".join(parts[2:-1]) if len(parts) > 3 else "default"
            if not format_name:
                format_name = "fp16"
            result_subdir = "minicpm-v"
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
    executor = CompressMiniCPMV(args.config_dir)
    executor.run()


if __name__ == "__main__":
    starter()
