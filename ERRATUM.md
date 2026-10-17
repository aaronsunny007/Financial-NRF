# Erratum and re-analysis

**Applies to:** the submitted dissertation, `NRF_Report_IEEE_v3.docx`, `NRF_Supporting_Evidence_v3.docx` and
`NRF_Experimental_Chapter.md` as they stood before this repository was audited. The current files in the repository
carry the corrections below; the submitted copies do not.

**Status.** One scoring defect, one experimental-design flaw, and several claims that did not survive a larger re-run.
The framework's design critique (gold-free evidence measures cannot detect corrupted evidence) stands and is now
demonstrated with a valid manipulation. The headline that the composite "fails to track reliability" and that
"no weighting rescues it" does not stand as stated.

## 1. The scoring defect

`framework/nrf.py` computed evidence support from every number in the model's response, including the number in
its own `Confidence: 0.9` line. The retrieved evidence can never support that number, so every response was
scored as carrying one extra unsupported number. That lowered every evidence-support score and pushed some responses
under the 0.5 hallucination threshold on that number alone. The scorer now removes the confidence statement before
counting (`exclude_confidence_from_support=True`, with regression tests; `False` reproduces the original numbers).

## 2. The experimental-design flaw

Experiment 4 degraded the evidence by adding up to four sentences about unrelated figures. A manipulation check
showed those sentences reached the model, but nothing showed they *harmed* it. At 150 items per split the paired
per-item change from 0% to 40% noise is indistinguishable from zero for accuracy (dev +0.007, 95% CI −0.05 to +0.07;
test −0.027, CI −0.07 to +0.03), evidence support, the hallucination flag, entailment support and support measured
against the clean evidence. The same holds for Qwen2.5-1.5B and SmolLM2-1.7B; across
three sampled seeds the accuracy change is +0.02, +0.05 and −0.11, and of eight noise-injection runs one interval (−0.22 to −0.01)
excludes zero, so any effect is small and inconsistent. A metric cannot be faulted for not tracking a degradation that did not happen, so
Experiment 4 as designed could not test the composite.

## 3. Claims in the submitted text, and what became of them

| Claim as submitted | Status | Basis |
|---|---|---|
| Hallucination rate rose from 0.10 to 0.20 under evidence injection (Exp. 2) | **Withdrawn** | Every strict-mode flag in Experiments 1 and 2 was the scoring defect. Re-scored from the saved responses: 0.00 in every condition. |
| Strict baseline hallucination rate 0.08 (Exp. 1) | **Corrected to 0.00** | Same cause. Evidence-support scores also rise by about 0.05 to 0.06. |
| The hallucination sub-metric tracks noise (r = +0.76, Exp. 4) | **Withdrawn** | Rested on four to six borderline flags per level. Corrected re-runs: −0.12 (n = 50), −0.42 (n = 150 dev), −0.32 (n = 150 test); with up to three flagged responses per level there is no signal. |
| The composite does not track noise (r = −0.10) | **Not established** | The r over five level averages is not stable: Qwen2.5-3B gives +0.09 on dev and −0.64 on test, each with a 95% interval of about ±0.8 Qwen2.5-1.5B sampled with three seeds gives +0.46, +0.52 and −0.37; SmolLM2-1.7B gives −0.81 [−0.97, +0.52]. And the noise did not degrade the model (section 2). |
| Oracle evidence is more accurate yet scores lower on the composite (0.5811 vs 0.5891) | **Not reproducible** | With the corrected scorer the two composites are a tie. In the larger runs oracle evidence does not significantly raise accuracy either (3B, dev: +0.007, CI −0.05 to +0.07; test: +0.034, CI −0.02 to +0.09). |
| No weighting of the four components passes both falsification tests (0.0% of 1,771) | **Not established** | 0.0% on the 3B dev run, but 41.2% on the 3B test run. For Qwen2.5-1.5B, 56 weightings passed both on dev and none of them on test; for SmolLM2-1.7B on dev, 73.1% pass both. The outcome is dominated by sampling noise in a five-point correlation. |
| Execution accuracy 0.340 | **Not stable** | Same model, items, prompts and seed in float32 gave 0.200 (n = 50), 0.255 on dev and 0.207 on test (n = 150). A single-run figure at n = 50 should not be quoted without an interval. |
| Evidence support measured against the evidence supplied cannot detect corrupted evidence (the structural argument in the chapter) | **Stands, now demonstrated** | With a manipulation that does degrade the model (numbers in the evidence replaced), accuracy falls from 0.22 to about 0.03; evidence support against the *shown* evidence, entailment support and the hallucination flag do not move; support against the *clean* evidence falls by 0.36 to 0.38 (95% CI 0.33 to 0.42), on both splits. |
| Three of the four components are insensitive to evidence quality; only accuracy responds | **Stands, now demonstrated** | The composite falls with corruption (r = −0.97 dev, −0.93 test), entirely through accuracy and the calibration term, which is a function of accuracy. |

## 4. Re-analysis in the repository

* `results/scale/SUMMARY.md` has every table, with bootstrap intervals over items. Code: `experiments/scale_analysis.py`.
* Runs: Qwen2.5-3B on 150 dev and 150 test items; Qwen2.5-1.5B on 100 and 100; SmolLM2-1.7B (a second model family) on 100 dev;
  Qwen2.5-1.5B with three sampled-decoding seeds on 100 dev items (Experiment 4 only for two of them, which hit the job time
  limit during Experiment 5); corrupted-evidence runs on 100 dev and 100 test items.
* `results/cpu_fp32/` is the earlier n = 50 float32 replication.
* Everything runs free on GitHub-hosted CPU runners (`.github/workflows/cpu-rerun.yml`); no GPU or paid API.

## 5. What was not done

* A 4-bit GPU run with the corrected scorer (the original runs were 4-bit on an RTX 3050).
* Models above 3B, datasets other than FinQA, or real (rather than synthetic) retrieval corruption.
