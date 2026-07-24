"""
LLM perplexity evaluation (WikiText-2 strided, C4 GPTQ-style segments)
on the fake-quantized model, matching the methodology of
llm/multiple_choice.py and the VLM entry scripts.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import torch

from transformers import AutoTokenizer, set_seed
from src.stage.base import Execute
from src.module.base import freeze_model_weights, _QBaseLinear
from src.trainer.llm.ptq import SmoothQuant
from src.trainer.llm.evaluator import WikiText, C4Perplexity
from src.t2c.convert import CONVERTNN

parser = argparse.ArgumentParser(description='LLM perplexity evaluation (wikitext / c4)')
parser.add_argument('--config_dir', type=str, default=None, help="Path to the configuration file (.yaml)")
args = parser.parse_args()

EVALUATORS = {
    "wikitext": WikiText,
    "c4": C4Perplexity,
}

class PerplexityEval(Execute):
    def __init__(self, config_dir):
        super().__init__(config_dir)
        set_seed(self.config["seed"])

        model_type = self.config["model"]["model_type"]
        model = self.create_model()
        self.tokenizer = self.prepare_tokenizer()

        # conversion transiently duplicates the weights; run it on CPU and
        # move the converted model to the GPU once
        model = model.cpu()
        torch.cuda.empty_cache()

        qcfg = self.config["quantization"]
        converter = CONVERTNN[model_type](
            model,
            wbit=qcfg["wbit"],
            abit=qcfg["abit"],
            quantize_bmm_input=qcfg.get("quantize_bmm_input", False),
            bmm_qtype=qcfg.get("bmm_qtype", "smooth_quant"),
            bmm_bits=qcfg.get("bmm_bits", 8),
            bmm_ebit=qcfg.get("bmm_ebit", None),
            bmm_block_size=qcfg.get("bmm_block_size", None),
            bmm_q_bits=qcfg.get("bmm_q_bits", None),
            bmm_kv_bits=qcfg.get("bmm_kv_bits", None),
            bmm_kv_ebit=qcfg.get("bmm_kv_ebit", None),
            bmm_q_ebit=qcfg.get("bmm_q_ebit", None),
            bmm_q_block_size=qcfg.get("bmm_q_block_size", None),
            bmm_kv_block_size=qcfg.get("bmm_kv_block_size", None),
            bmm_kv_shared_exp_bits=qcfg.get("bmm_kv_shared_exp_bits", None),
            bmm_kv_shared_exp_relative=qcfg.get("bmm_kv_shared_exp_relative", False),
        )

        self.model = converter.convert()
        self.model = self.model.half()

        for m in self.model.modules():
            if isinstance(m, _QBaseLinear):
                m.mask = None

        self.model = self.model.to(self.device)

        self.task = SmoothQuant(config_dir, self.model, self.tokenizer, self.logger)

    def prepare_tokenizer(self):
        model_type = self.config["model"]["model_type"]
        tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.pad_token_id = 0

        return tokenizer

    def ptq(self):
        fake_quantized_model = self.task.run()
        return fake_quantized_model

    def result_path(self):
        cfg_path = os.path.abspath(self.config_dir)
        fmt = os.path.basename(os.path.dirname(cfg_path))
        model_name = os.path.basename(os.path.dirname(os.path.dirname(cfg_path)))

        result_dir = self.config["save"].get("result_dir", None)
        if result_dir is None:
            result_dir = os.path.join("accuracy_result", "llm", model_name, fmt)

        task_name = self.config["dataset"]["name"]
        os.makedirs(result_dir, exist_ok=True)

        return os.path.join(result_dir, f"{task_name}.json"), fmt

    def evaluate(self, fake_quant_model):
        num_frozen = freeze_model_weights(fake_quant_model)
        self.logger.info(f"Froze quantized weights of {num_frozen} modules")

        task_name = self.config["dataset"]["name"]
        evaluator = EVALUATORS[task_name](self.config_dir, fake_quant_model, self.tokenizer)
        self.logger.info(f"\n Evaluating the fake-quantized model...")

        start = time.time()
        ppl = evaluator.run()

        results = {
            "task": task_name,
            "perplexity": ppl,
            "elapsed_seconds": round(time.time() - start, 1),
            "model": self.config["model"]["model_type"],
        }

        result_file, fmt = self.result_path()
        results["format"] = fmt
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {result_file}")

        return results

    def run(self):
        fake_quant_model = self.ptq()
        self.evaluate(fake_quant_model)

def starter():
    executor = PerplexityEval(args.config_dir)
    executor.run()

if __name__ == "__main__":
    starter()
