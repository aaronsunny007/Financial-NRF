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
| Noise correlation −0.10 | `experiment4_results_dev_summary.json` |
| Oracle vs retrieval, ranking inversion | `experiment5_results_dev_summary.json` |

## Per-item columns

Each CSV records the question, the retrieved evidence identifiers, the generated
answer, the resolved gold value, and the four component scores for that item — so
any aggregate can be recomputed from the rows rather than taken on trust.
