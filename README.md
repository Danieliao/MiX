# MiX: Micro-Inverted-Scaling — Artifact Evaluation

This repository reproduces the results of the MICRO 2026 paper
*Micro-Inverted-Scaling (MiX)*. It is organized into three self-contained parts:

| Dir | Reproduces | What it is |
|-----|-----------|------------|
| [`accuracy/`](accuracy/) | **Tables 2, 3, 4, 7, 8** | Model-accuracy evaluation: the MiX/MX/NVFP4 quantization code, per-benchmark YAML configs, and eval scripts for VLMs (LLaVA-OneVision-7B, Qwen2-VL-7B, MiniCPM-V-2.6, Qwen2.5-VL 3B–72B) and text-only LLMs (Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B/14B). |
| [`hardware_rtl/`](hardware_rtl/) | **Tables 5, 6** | SystemVerilog RTL, self-checking testbenches, and Design Compiler / SAIF-power scripts for 15 systolic-array formats + the MiX/MX quantizer and K-smoother. Reproduces `results_28nm_iso512.csv`. Needs Synopsys DC + VCS and a 28 nm cell library. |
| [`hardware_model/`](hardware_model/) | **Figures 9, 10** | Standalone analytical energy/area/speedup simulator (no GPU, no `torch`). |

## Environment

```bash
conda create -n mix python=3.10 -y
conda activate mix
pip install -r requirements.txt
```

The accuracy pipeline needs a GPU with ≥48 GB VRAM for the 7B/8B models
(≥80 GB for Qwen2.5-VL-32B/72B and Qwen2.5-14B). The hardware model runs on CPU.

Model weights are pulled from HuggingFace on first run. Some are gated
(Llama-3.1-8B) or need `trust_remote_code=True` + HF auth (MiniCPM-V-2.6);
set `HF_TOKEN` in your environment. Each part's README lists the exact model IDs.

## Reproduce order (important)

Figure 10 (`hardware_model/`) consumes `accuracy_result/table2.json`, which is
**produced by Part 1** — this repo ships no precomputed results. To reproduce
everything end to end:

1. Run the `accuracy/` evaluations (Table 2 at minimum) and build
   `accuracy/accuracy_result/table2.json` with `generate_table2_json.py`.
2. Copy that `table2.json` into `hardware_model/accuracy_result/`.
3. Run the `hardware_model/` simulator for Figures 9 and 10.

Figure 9 is independent of Part 1 and can be run at any time.

See each subdirectory's `README.md` for exact commands.
