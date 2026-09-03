"""
Generate accuracy_result/table2.json for filling Table 2 (tab:accuracy_results)
in paper/main.tex.

Rebuttal revision (2026-06): all MXFP4-family rows now read from the *ceil*
shared-exponent directories (the MXFP4-ceil anomaly fix the reviewers asked for),
and two new 4.5b FP4 baselines are added: MX+ (MXFP4+) and AMXFP4.

  floor -> ceil dir remaps (MXFP4 anomaly fix):
    mxfp4                       -> mxfp4ceil
    mxfp4g16                    -> mxfp4ceilg16
    mxfp4g16-mix4.5b-l-flip     -> mxfp4ceilg16-mix4.5b-l-flip
    mxfp4-mix4.25b-l-flip       -> mxfp4ceil-mix4.25b-l-flip
  new rows:
    mxfp4plus  (MX+, FLOOR variant -- consistently beats the ceil variant; see
               docs/mxfp4_baselines.md "use whichever evaluates better")
    amxfp4     (AMXFP4, non-clipping by construction -- no floor/ceil variant)

Handles:
- OCRBench: uses total_score/total (not accuracy field), normalized to percentage
- OOM cases: some benchmarks have fewer samples than expected, adjusts accordingly
- Computes average across all 6 benchmarks

Output: accuracy_result/table2.json with per-model, per-format, per-benchmark values
(each entry also carries `bits` and `tier` for the LaTeX renderer in paper/table/).
"""

import json
import os

# Expected total samples per benchmark
EXPECTED_TOTAL = {
    'chartqa': 2500,
    'mmmu': 900,
    'ocrbench': 1000,
    'seedbench2plus': 2277,
    'textvqa': 5000,
    'vizwiz': 4319,
}

# Table column order
BENCHMARKS = ['mmmu', 'ocrbench', 'vizwiz', 'textvqa', 'chartqa', 'seedbench2plus']

# Format spec: (directory name, table display name, effective bits, tier)
#   tier groups the rows in the paper table; 'baseline' = FP16 reference row.
#   NOTE: directory tokens are the clean names used in this artifact repo;
#   MXFP4-family rows use the ceil shared-exponent quantizer (mxfp4_ceil_quant)
#   inside the YAMLs, so the "MXFP4"/"MXFP4_g16" dirs are the ceil variant.
FORMATS = [
    ('fp16',                'FP16',           16.0,  'baseline'),
    # ---- 4.88b ----
    ('mix-4.5b-int5',       'MiX-4.5b-INT5',  4.88,  '4.88b'),
    # ---- 4.50b (block size 16, plus the 4.5b FP4 baselines at block 32) ----
    ('mxint4g16',           'MXINT4_g16',     4.50,  '4.50b'),
    ('mix-4.5b-int4',       'MiX-4.5b-INT4',  4.50,  '4.50b'),
    ('mxfp4g16',            'MXFP4_g16',      4.50,  '4.50b'),
    ('mix-4.5b-fp4',        'MiX-4.5b-FP4',   4.50,  '4.50b'),
    ('mxfp4plus',           'MXFP4+',         4.50,  '4.50b'),
    ('amxfp4',              'AMXFP4',         4.50,  '4.50b'),
    ('nvfp4',               'NVFP4',          4.50,  '4.50b'),
    # ---- 4.25b (block size 32) ----
    ('mxint4',              'MXINT4',         4.25,  '4.25b'),
    ('mix-4.25b-int4',      'MiX-4.25b-INT4', 4.25,  '4.25b'),
    ('mxfp4',               'MXFP4',          4.25,  '4.25b'),
    ('mix-4.25b-fp4',       'MiX-4.25b-FP4',  4.25,  '4.25b'),
]

MODELS = ['qwen2vl', 'minicpm-v', 'llava-onevision']


# Results whose evaluator dropped samples after a CUDA OOM. Those samples never
# enter `total`, so the score below is renormalized over a subset and is not
# comparable to the paper. Collected here and reported at the end of the run.
PARTIAL = []


def load_accuracy(model, fmt_dir, bench):
    """Load and compute accuracy percentage for a single result."""
    path = f'accuracy_result/{model}/{fmt_dir}/{bench}.json'
    if not os.path.exists(path):
        return None

    with open(path) as f:
        d = json.load(f)

    skipped = d.get('skipped_oom', 0)
    if skipped:
        PARTIAL.append((path, skipped, d.get('total', 0)))

    if bench == 'ocrbench':
        # OCRBench: total_score out of total (typically 1000)
        total_score = d.get('total_score', 0)
        total = d.get('total', EXPECTED_TOTAL['ocrbench'])
        # Normalize to percentage, scaled to 100 (like other benchmarks)
        # If OOM caused fewer samples, scale up: score/tested * 100
        return total_score / total * 100
    else:
        # Standard benchmarks: accuracy field is already a ratio [0, 1]
        acc = d.get('accuracy', 0)
        return acc * 100


def main():
    result = {}

    for model in MODELS:
        result[model] = {}
        for fmt_dir, fmt_name, bits, tier in FORMATS:
            entry = {'name': fmt_name, 'bits': bits, 'tier': tier, 'benchmarks': {}}
            values = []

            for bench in BENCHMARKS:
                acc = load_accuracy(model, fmt_dir, bench)
                if acc is not None:
                    entry['benchmarks'][bench] = round(acc, 1)
                    values.append(acc)
                else:
                    entry['benchmarks'][bench] = None

            # Average: only if all 6 benchmarks available
            if len(values) == 6:
                entry['avg'] = round(sum(values) / len(values), 1)
            else:
                entry['avg'] = None

            result[model][fmt_dir] = entry

    if PARTIAL:
        print('\n*** WARNING: results with OOM-skipped samples ***')
        for path, skipped, total in PARTIAL:
            print(f'  {path}: {skipped} skipped, only {total} of '
                  f'{skipped + total} samples scored')
        print('  These cells are renormalized over a subset and are NOT '
              'comparable to the paper.\n  Rerun them on a GPU with more '
              'memory (see dependency.md).')

    # Save
    out_path = 'accuracy_result/table2.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {out_path}')

    # Print table for verification
    for model in MODELS:
        print(f'\n=== {model} ===')
        header = f"{'Method':<25} {'MMMU':>7} {'OCR':>7} {'VizWiz':>7} {'TxtVQA':>7} {'ChrtQA':>7} {'SEED2+':>7} {'Avg':>7}"
        print(header)
        print('-' * len(header))
        for fmt_dir, fmt_name, bits, tier in FORMATS:
            e = result[model][fmt_dir]
            row = f"{fmt_name:<25}"
            for bench in BENCHMARKS:
                v = e['benchmarks'][bench]
                row += f" {v:>7.1f}" if v is not None else f" {'--':>7}"
            avg = e['avg']
            row += f" {avg:>7.1f}" if avg is not None else f" {'--':>7}"
            print(row)


if __name__ == '__main__':
    main()
