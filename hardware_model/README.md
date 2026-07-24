# Part 3 — Hardware Energy/Area Model (Figures 9, 10)

A standalone analytical simulator for the MiX accelerator. **No GPU, no `torch`,
no `src/`** — only `numpy` + `matplotlib`. Run everything from **this** directory
(`hardware_model/`) so the `simulator` package resolves as a module.

The 28 nm / 500 MHz per-PE synthesis numbers (from Part 2) are baked in via
`simulator/mix_simulator/config.py`, `hardware.csv`, and
`results_28nm_iso512_saif.csv`.

## Figure 9 — iso-area speedup + normalized energy

```bash
python -m simulator.mix_simulator.main --model llava     --output_dir simulator/mix_simulator/output/llava
python -m simulator.mix_simulator.main --model minicpm_v --output_dir simulator/mix_simulator/output/minicpm_v
python -m simulator.mix_simulator.plot_fig9
# -> simulator/mix_simulator/output/combined_speedup_energy.pdf
```
`plot_fig9` also reads the Focus baseline from `simulator/focus_eval/focus_results.json`.

## Figure 10 — PE-efficiency Pareto frontier

Figure 10 needs `accuracy_result/table2.json`, which is produced by **Part 1**
(`accuracy/`). This repo ships no results, so build it first, then stage it here:

```bash
# after running Part 1 and its generate_table2_json.py:
cp ../accuracy/accuracy_result/table2.json accuracy_result/table2.json
python -m simulator.mix_simulator.pareto.gen_fig10
# -> simulator/mix_simulator/pareto/output/pe_efficiency_pareto.pdf
```

## Contents

```
simulator/mix_simulator/
├── main.py              # analytical sim -> results.json
├── energy.py            # cycle/energy model
├── config.py            # accelerator configs + hardcoded 28nm area/power (from Part 2)
├── model_profile.py     # per-model layer shapes (no external model files)
├── plot_fig9.py           # Figure 9
├── hardware.csv                    # per-PE synthesis specs
├── results_28nm_iso512_saif.csv    # iso-512 SAIF area/power (Fig 10 input)
└── pareto/gen_fig10.py  # Figure 10
simulator/focus_eval/focus_results.json   # Focus baseline (Fig 9 input)
accuracy_result/          # empty; table2.json lands here from Part 1
```
