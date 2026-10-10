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

## Which file backs which claim

| Claim | File |
|---|---|
| Baseline accuracy 0.340 | `experiment1_results_dev_summary.json` |
| Calibration figures (n = 50) | `experiment3_results_dev_summary.json` |
| Noise correlation −0.10 | `experiment4_results_dev.csv` (ONRS per noise level); the computed r is in `weight_sensitivity_dev_summary.json` |
| Oracle vs retrieval, ranking inversion | `experiment5_results_dev_summary.json` |
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
