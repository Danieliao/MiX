# Part 1 — Model Accuracy Evaluation (Tables 2, 3, 4, 7, 8)

Post-training quantization + evaluation for the MiX / MX / NVFP4 formats.
Every result is produced by running a single per-benchmark YAML config through
the matching entry script. Run all commands from **this** directory
(`accuracy/`) with the `mix` conda environment active.

```bash
conda activate mix
export CUDA_VISIBLE_DEVICES=0
```

## Entry scripts and models

| Script | Model | HuggingFace ID |
|--------|-------|----------------|
| `vlm/qwen2vl.py`         | Qwen2-VL-7B          | `Qwen/Qwen2-VL-7B-Instruct` |
| `vlm/llava_onevision.py` | LLaVA-OneVision-7B   | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| `vlm/minicpmv.py`        | MiniCPM-V-2.6        | `openbmb/MiniCPM-V-2_6` (trust_remote_code) |
| `vlm/qwen2_5vl.py`       | Qwen2.5-VL 3B–72B    | `Qwen/Qwen2.5-VL-{3B,7B,32B,72B}-Instruct` |
| `llm/multiple_choice.py` | LLM reasoning (ARC-c/HellaSwag/WinoGrande) | Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B/14B |
| `llm/perplexity.py`      | LLM perplexity (WikiText/C4)               | same four LLMs |

Results are written to `accuracy_result/{model}/{format}/{benchmark}.json`
(LLM: `accuracy_result/llm/{model}/{format}/{task}.json`), where `{format}` is
the config directory name. **This repo ships no precomputed results.**

## Quantization formats (config directory names)

| Dir | Paper name | EBW | Notes |
|-----|-----------|-----|-------|
| `fp16` | FP16 | 16 | baseline |
| `mxint4` / `mxint4g16` | MXINT4 / MXINT4_g16 | 4.25 / 4.5 | |
| `mxfp4` / `mxfp4g16` | MXFP4 / MXFP4_g16 | 4.25 / 4.5 | **ceil** shared-exp (`mxfp4_ceil_quant`) |
| `nvfp4` | NVFP4 | 4.5 | |
| `amxfp4`, `mxfp4plus` | AMXFP4, MXFP4+ | 4.5 | outlier-aware FP4 baselines |
| `mix-4.25b-int4`, `mix-4.5b-int4` | MiX-4.25b/4.5b-INT4 | 4.25 / 4.5 | **Inverted-MX** |
| `mix-4.25b-fp4`, `mix-4.5b-fp4` | MiX-4.25b/4.5b-FP4 | 4.25 / 4.5 | |
| `mix-4.5b-int5` | MiX-4.5b-INT5 | 4.88 | |

## Table 2 — End-to-end VLM accuracy (3 models × 12 formats × 6 benchmarks)

```bash
# one benchmark of one format:
python vlm/qwen2vl.py --config_dir config/qwen2vl/mix-4.5b-int4/qwen2-vl-7b-mix-4.5b-int4-mmmu.yaml
# sweep a model with a shell loop over config/{qwen2vl,minicpm-v,llava-onevision}/*/*.yaml
```
Benchmarks: `mmmu`, `ocrbench`, `vizwiz`, `textvqa`, `chartqa`, `seedbench2plus`.

Build the table after the runs finish:
```bash
python accuracy_result/generate_table2_json.py   # -> accuracy_result/table2.json
```

## Table 3 — Static vs dynamic K-smoothing (`config/k-smooth-static/`)

Static-K configs for Qwen2-VL-7B and MiniCPM-V-2.6 (4 formats × MMMU/OCRBench/SEED2+).
The "dynamic" column reuses the Table 2 results.
```bash
python vlm/qwen2vl.py --config_dir config/k-smooth-static/qwen2vl/mix-4.5b-int4/<file>.yaml
```

## Table 4 — MXFP4 floor-vs-ceil anomaly (`config/mxfp4-anomaly/`)

The **floor** MXFP4/MiX-FP4 variants (Qwen2-VL-7B, LLaVA-OV-7B; ChartQA + VizWiz).
The ceil counterparts and MXINT4 rows come from Table 2.
```bash
python vlm/qwen2vl.py --config_dir config/mxfp4-anomaly/qwen2vl/mxfp4-floor/<file>.yaml
```

## Table 7 — Qwen2.5-VL model-size scaling (`config/qwen2.5vl-{3b,7b,32b,72b}/`)

4 sizes × {`fp16`, `mxfp4`, `nvfp4`, `mix-4.5b-int4`} × {MMMU, OCRBench, SEED2+}.
```bash
python vlm/qwen2_5vl.py --config_dir config/qwen2.5vl-3b/mix-4.5b-int4/<file>.yaml
```

## Table 8 — Text-only LLM ablation (`config/llm/`)

4 models × {`fp16`, `nvfp4`, `mix-4.5b-int4`} × 5 tasks.
Reasoning tasks use `multiple_choice.py`; perplexity uses `perplexity.py`:
```bash
python llm/multiple_choice.py --config_dir config/llm/llama3.1-8b/mix-4.5b-int4/llama3.1-8b-mix-4.5b-int4-arc-challenge.yaml
python llm/perplexity.py     --config_dir config/llm/llama3.1-8b/mix-4.5b-int4/llama3.1-8b-mix-4.5b-int4-wikitext.yaml
```

## Methodology note

Accuracy is measured on the **fake-quantized** model (weights frozen; no T2C
integer fusion) — this matches the paper. SmoothQuant is disabled for all MiX
configs (`smooth.flag: false`); the multiplier-free MiX pipeline needs no
activation smoothing.
