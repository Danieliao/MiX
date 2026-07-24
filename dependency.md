# Artifact Dependencies

This artifact has three self-contained parts, each reproducing a different set of
paper results:

| Part | Directory | Reproduces |
|------|-----------|-----------|
| **Part 1 — Accuracy** | [`accuracy/`](accuracy/) | Tables 2, 3, 4, 7, 8 |
| **Part 2 — Hardware RTL** | [`hardware_rtl/`](hardware_rtl/) | Tables 5, 6 |
| **Part 3 — Hardware Model** | [`hardware_model/`](hardware_model/) | Figures 9, 10 |

The four sections below (Key Results, Hardware, Software, Data) answer the AE
form, each broken down per part.

---

## 1. Key Results to be Reproduced

| Result | Part | Description |
|--------|------|-------------|
| **Table 2** | 1 | End-to-end VLM task accuracy — LLaVA-OneVision-7B, Qwen2-VL-7B, MiniCPM-V-2.6 × 12 sub-8-bit formats (+ FP16) × 6 benchmarks (MMMU, OCRBench, VizWiz, TextVQA, ChartQA, SEED-Bench-2+) |
| **Table 3** | 1 | Static vs. dynamic K-smoothing (Qwen2-VL-7B, MiniCPM-V-2.6; 4 formats × MMMU/OCRBench/SEED2+) |
| **Table 4** | 1 | MXFP4 floor-vs-ceil shared-exponent anomaly (Qwen2-VL-7B, LLaVA-OV-7B; ChartQA, VizWiz) |
| **Table 7** | 1 | Model-size scaling on Qwen2.5-VL 3B→72B (FP16, MXFP4, NVFP4, MiX-INT4_g16; MMMU/OCRBench/SEED2+) |
| **Table 8** | 1 | Text-only LLM ablation — Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B/14B (ARC-Challenge, HellaSwag, WinoGrande, WikiText ppl, C4 ppl) |
| **Table 5** | 2 | Area & power breakdown of the MiX-INT4_g16 28 nm accelerator (systolic array, quantizer, K-smoother, SRAM) at 500 MHz |
| **Table 6** | 2 | RTL synthesis (28 nm, 500 MHz, SAIF power) of the iso-throughput 512-MAC arrays across 15 formats → `results_28nm_iso512.csv` |
| **Figure 9** | 3 | Iso-area speedup and normalized energy breakdown |
| **Figure 10** | 3 | PE-efficiency Pareto frontier across formats |

**Cross-part dependency:** Figure 10 (Part 3) consumes `accuracy_result/table2.json`
(produced by Part 1) and `results_28nm_iso512_saif.csv` (identical to Part 2's
synthesis output). Run Part 1's Table 2 and build `table2.json` before Figure 10.

---

## 2. Hardware Dependencies

### Part 1 — Accuracy (GPU + large system RAM)

Post-training quantization inflates each layer to FP32 on the CPU before casting
back to FP16, so **system RAM must exceed the FP32 model footprint** during the
quantization stage, and the fake-quantized model must fit on a **single GPU**.
GPU VRAM requirements grow faster than the raw model size because of this
quantization overhead:

| Model size | Example models | Minimum GPU | System RAM |
|------------|----------------|-------------|------------|
| ≤ 7–8 B | LLaVA-OV-7B, Qwen2-VL-7B, MiniCPM-V-2.6, Qwen2.5-VL-3B/7B, Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B | **NVIDIA RTX A6000 (48 GB)** | ≥ 128 GB |
| 14 B | Qwen2.5-14B | **RTX A6000 (48 GB)** — measured peak ≈ 33.5 GB, fits comfortably | ≥ 128 GB |
| 32 B | Qwen2.5-VL-32B | **NVIDIA A100 (80 GB)** | ≥ 256 GB |
| 72 B | Qwen2.5-VL-72B | **NVIDIA B200 (180 GB)** — needed to hold the whole quantized model on one GPU | **≥ 500 GB** |

- **Single GPU only** — no multi-GPU/model-parallel path is used.
- CPU cores are not a bottleneck; **4–8 cores** suffice. The binding constraints
  are **GPU VRAM** and, for the 72 B model, **≥ 500 GB system RAM** for the
  initial (FP32, CPU-side) quantization calculations.
- Compatible 48 GB GPUs for the ≤14 B tier: A6000, A40, RTX 6000 Ada. The A100
  (80 GB) / H100 / H200 / B200 also work and are required at 32 B / 72 B.

### Part 2 — Hardware RTL (CPU EDA host)

- Any **x86-64 Linux workstation/server** that runs the Synopsys tools (§3).
- **No GPU, FPGA, or emulator** — this is logic synthesis + gate-level
  simulation of small (512-MAC) arrays. Modest resources are enough:
  ~**8 cores, 16–32 GB RAM** per design.

### Part 3 — Hardware Model (CPU only)

- **CPU-only**, no GPU. The analytical simulator runs on a laptop
  (a few cores, < 4 GB RAM).

---

## 3. Software Dependencies

### Part 1 — Accuracy (free / open-source)

- **Conda + Python 3.10**, then `pip install -r requirements.txt`. Key pinned
  packages: `torch==2.3.0` (CUDA 12.1 wheels), `transformers==4.57.6`,
  `timm==1.0.14`, `datasets==3.2.0`, `accelerate==1.3.0`, `numpy==1.26.4`,
  `matplotlib==3.10.8`, plus `sentencepiece==0.2.1` / `protobuf` (Mistral & some
  VLM tokenizers).
- A **CUDA 12.x** driver/toolkit matching the Torch build.
- **HuggingFace Hub** access (model/dataset download; `trust_remote_code=True`
  for MiniCPM-V-2.6). **No proprietary software.**

### Part 2 — Hardware RTL (proprietary EDA required)

| Tool | Purpose | Version used |
|------|---------|--------------|
| **Synopsys Design Compiler** (`dc_shell`) | Logic synthesis → area, timing, the `.ddc`; and `read_saif` + `report_power` for the SAIF-annotated `Power_mW` | U-2022.12-SP5 |
| **Synopsys VCS** | SystemVerilog simulation — functional testbenches **and** the gate-level sim that captures SAIF switching activity | U-2023.03-SP1-1 |

Both are commercial EDA tools requiring a valid Synopsys license. Any reasonably
recent release works; the flow uses only standard commands (`analyze` /
`elaborate` / `compile_ultra` / `report_area` / `report_power` / `read_saif`, and
VCS's `$toggle_*` SAIF API). **No Cadence, Xilinx, or Mentor tools are needed;
PrimeTime is not required** — power is done entirely in Design Compiler.

Free/standard also needed: **Python 3** (standard library only — `csv`, `os`,
`re`; no pip packages) for `power_saif/gen_tb.py` and `update_csv.py`; **GNU Make**
(optional `make syn` targets); and a **Unix shell** (`bash` for
`power_saif/run_all.sh`; `tcsh` used to load the DC environment module — a
site-config placeholder). No UVM, DPI, or external IP.

### Part 3 — Hardware Model (free / open-source)

- **Python 3** with only **`numpy`** and **`matplotlib`** (both already in
  `requirements.txt`). No `torch`, no `src/`, no proprietary software.

---

## 4. Data Dependencies

### Part 1 — Accuracy (models + datasets, auto-downloaded, not shipped)

Nothing is bundled; everything is pulled from HuggingFace on first run.

**Models (HuggingFace IDs):**
- VLM: `llava-hf/llava-onevision-qwen2-7b-ov-hf`, `Qwen/Qwen2-VL-7B-Instruct`,
  `openbmb/MiniCPM-V-2_6`, `Qwen/Qwen2.5-VL-{3B,7B,32B,72B}-Instruct`
- LLM: `meta-llama/Llama-3.1-8B` **(gated — accept the license and set `HF_TOKEN`)**,
  `mistralai/Mistral-7B-v0.3`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`

**Evaluation datasets (HuggingFace IDs):**
- VLM: `HuggingFaceM4/ChartQA`, `MMMU/MMMU`, `echo840/OCRBench`,
  `AILab-CVC/SEED-Bench-2-plus`, `facebook/textvqa`, `lmms-lab/VizWiz-VQA`
- LLM: `allenai/ai2_arc` (ARC-Challenge), `Rowan/hellaswag`,
  `allenai/winogrande`, `wikitext` (wikitext-2-raw-v1), `allenai/c4`

**Calibration data:** `lmms-lab/POPE` (COCO images + yes/no prompts) for VLM
activation profiling; `mit-han-lab/pile-val-backup` for the LLM path. Downloaded
automatically. A HuggingFace token is required for the gated Llama model and
recommended to avoid rate limits.

### Part 2 — Hardware RTL (28 nm standard-cell library, not shippable)

**Required but not included (not redistributable):** a **28 nm standard-cell
library**. The paper used **TSMC `tcbn28hpcplusbwp30p140`**; the flow needs

- the **`.db` timing/power library** (worst-case corner `ssg0p81v125c`, SS /
  0.81 V / 125 °C) — for synthesis and `report_power`, and
- the matching **Verilog cell simulation model (`.v`)** — for gate-level SAIF
  capture.

Point the scripts at your own library via the `SITE CONFIGURATION` block in each
`syn/<Design>/syn.tcl` (and in `power_saif/run_all.sh` / `gen_tb.py`). Any 28 nm
library runs; absolute area/power shift, but the *relative* ranking across
formats — the paper's claim — is preserved. **No benchmarks, datasets, or trained
models are needed:** every testbench generates its own random stimulus and checks
against an in-testbench reference; the target result `results_28nm_iso512.csv` is
included.

### Part 3 — Hardware Model (self-contained)

**No external data or models.** The simulator uses hardcoded per-model layer
profiles (`model_profile.py`) and two included inputs — the Focus baseline
`simulator/focus_eval/focus_results.json` and the synthesis table
`results_28nm_iso512_saif.csv` (from Part 2). Its only cross-part input is
`accuracy_result/table2.json`, which you generate by running Part 1 (see the
Figure 10 instructions in `hardware_model/README.md`).
