# CPU re-run with the corrected scorer

Experiments 1 to 5 re-run on free GitHub-hosted CPU runners, with no GPU and no paid API,
using the evidence-support scorer that no longer counts the model's own `Confidence: x`
value as an unsupported number (`results/README.md`, scoring note).

## What differs from the recorded runs

| | Recorded runs (`results/`) | This re-run (`results/cpu_fp32/`) |
|---|---|---|
| Generator | Qwen2.5-3B-Instruct, 4-bit NF4 | Qwen2.5-3B-Instruct, **float32** |
| Hardware | local RTX 3050 | 4-vCPU Intel Xeon Platinum 8573C runners |
| Evidence-support scorer | counted the confidence value | **excludes it** |
| Items, seed, prompts, retrieval, scoring | same | same |

Greedy decoding is deterministic for a given set of weights, but 4-bit and 32-bit weights are
different weights, so individual answers differ: of the 50 baseline items, 15 give the same
predicted value and 37 the same correct/incorrect outcome. This is a replication of the
pipeline on a different numeric precision, not a regeneration of the recorded answers.
Accuracy fell from 0.34 to 0.20, which is itself a measure of how unstable a 3B model's
answers are at n = 50 (see the noise floor in the experimental chapter, section 2.3).

## How it was produced

`.github/workflows/cpu-rerun.yml`. Generation is the slow part (mean 118 s per answer, 579
distinct generations), so 20 shard jobs each ran the five experiment scripts over their share
of the items and recorded every answer in a cache keyed by a hash of the exact chat messages.
A final job ran the **unchanged** experiment scripts with `--engine replay`, which loads no model,
reads the answers from `generations.jsonl`, and raises on any prompt that is not in it.

## Files

| File | Contents |
|---|---|
| `generations.jsonl` | Every generation, keyed by prompt hash, with seconds taken |
| `experiment{1..5}_results_dev*.csv/json` | The five experiments, same format as `results/` |
| `weight_sensitivity_dev*.csv/json` | The weight sweep (`python -m experiments.weight_sensitivity --results-dir results/cpu_fp32`) |
| `log_experiment{1..5}.txt` | Console output of each replayed experiment |

To redo only the scoring and replay from the committed answers, dispatch the workflow with
`replay_only = true`. To regenerate everything, dispatch it with the defaults.
