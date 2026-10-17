# Results

One row per FinQA item for every experimental run, plus a summary JSON per run.
These are committed deliberately: every figure quoted in the report and the README
can be traced back to the individual answers that produced it.

## Naming

| Suffix | Meaning |
|---|---|
| `_dev` | Run against the FinQA dev split |
| *(no suffix)* | Run against the test split |
| `_prototype` | Early run, retained for provenance — superseded, do not cite |
| `_per_example` | One row per item per condition, for multi-condition experiments |
| `_summary.json` | Aggregate metrics and confidence intervals for that run |

Where a `_prototype` file and a final file both exist, the final file is authoritative.

## Scoring note: the recorded runs predate a fix

The `dev` files for Experiments 1, 2, 4 and 5 were scored while the evidence-support
metric still counted the model's own `Confidence: 0.9` value as an unsupported number.
`framework/nrf.py` now excludes it (`exclude_confidence_from_support=True`). The files
here are what was run and have not been regenerated. `confidence_line_audit_dev.json`
holds the re-scored comparison: exact for Experiments 1 and 2, near-exact for 5, and
bracketed for 4 (which did not save its responses). To reproduce a recorded number
from code, construct `NumericalReliabilityFramework(exclude_confidence_from_support=False)`.

`cpu_fp32/` holds a re-run of all five experiments with the corrected scorer in float32 on CPU
(see `cpu_fp32/README.md`). Its numbers are a replication on different numerics and should be
quoted next to, not in place of, the recorded runs.

`scale/` holds the larger runs (more items, a test split, other models, sampled seeds, and a corrupted-evidence manipulation), one
directory per run, named `<model>_<split>[_s<seed>]`; `scale/summary.json` and `scale/SUMMARY.md` are produced by
`python -m experiments.scale_analysis` and `python -m experiments.scale_report`. Two of the seed runs (`qwen1p5b_dev_s1`, `_s2`) hit the
six-hour job limit during Experiment 5, so they hold Experiments 1, 3 and 4 only.

## Which file backs which claim

| Claim | File |
|---|---|
| Baseline accuracy 0.340 | `experiment1_results_dev_summary.json` |
| Calibration figures (n = 50) | `experiment3_results_dev_summary.json` |
| Noise correlation −0.10 | `experiment4_results_dev.csv` (ONRS per noise level); the computed r is in `weight_sensitivity_dev_summary.json` |
| Oracle vs retrieval, ranking inversion | `experiment5_results_dev_summary.json` |
| Measured re-run with the corrected scorer | `cpu_fp32/` (`weight_sensitivity_dev_summary.json` there for the sweep) |
| Larger runs, bootstrap intervals, held-out validation, seeds | `scale/SUMMARY.md` (`scale/summary.json`) |
| Effect of the confidence-line scoring defect | `confidence_line_audit_dev.json` (`python -m experiments.confidence_line_audit`) |
| No weighting passes both falsification tests | `weight_sensitivity_dev_summary.json` (one row per weighting in `weight_sensitivity_dev.csv`) |

## What the rows contain

Most CSVs are one row per FinQA item: the item `id`, the resolved gold value, the
predicted value, whether it was correct, and the component scores for that item
(evidence support, hallucination flag, self-reported confidence, and verifier outcome
where the experiment ran it). Experiment 1 and 2 dev files also keep the question,
the evidence text and the full model response, so any single answer can be audited.

Two kinds of file are aggregates rather than per-item:

- `experiment4_results*.csv` (without `_per_example`) holds one row per noise level.
  The per-item rows for the same run are in the matching `_per_example` file.
- `weight_sensitivity_dev.csv` holds one row per weight vector, produced by
  `python -m experiments.weight_sensitivity` from the Experiment 4 and 5 dev results.

Per-item aggregates can be recomputed from the rows rather than taken on trust.
