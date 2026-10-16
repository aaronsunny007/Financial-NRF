# Erratum and re-analysis

**Applies to:** the submitted dissertation, `NRF_Report_IEEE_v3.docx`, `NRF_Supporting_Evidence_v3.docx` and
`NRF_Experimental_Chapter.md` as they stood before this repository was audited. The current versions of
the repository files already carry the corrections below; the submitted PDF/Word copies do not.

**Status:** one scoring defect and several claims that did not survive re-running. The headline negative result
(the composite score does not behave as a reliability signal) stands, and is now supported by more data.

## 1. The defect

`framework/nrf.py` computed evidence support from every number in the model's response, including the
number in its own `Confidence: 0.9` line. The retrieved evidence can never support that number, so every
response was scored as carrying one extra unsupported number. This lowered every evidence-support score
and pushed some responses under the 0.5 hallucination threshold on that number alone.

The scorer now removes the confidence statement before counting
(`NumericalReliabilityFramework(exclude_confidence_from_support=True)`, with regression tests).
`exclude_confidence_from_support=False` reproduces the original numbers.

## 2. Claims in the submitted text, and what became of them

| Claim as submitted | Status | Why |
|---|---|---|
| Hallucination rate rose from 0.10 to 0.20 under evidence injection (Experiment 2) | **Withdrawn** | All strict-mode flags in Experiments 1 and 2 were this defect. Re-scored from the saved responses: 0.00 in every condition. |
| Strict hallucination rate 0.08 at baseline (Experiment 1) | **Corrected to 0.00** | Same cause. Evidence-support scores also rise by about 0.05 to 0.06. |
| The hallucination sub-metric tracks noise (r = +0.76, Experiment 4) | **Withdrawn** | Rested on four to six borderline flags per level. Corrected re-run: r = −0.12; see section 3 for the larger runs. |
| The composite does not track noise (r = −0.10, five levels) | **Stands, with the figure retracted as a point estimate** | Direction of the claim holds; the number is not stable (see section 3, where its confidence interval spans zero at n = 50). |
| Oracle evidence is more accurate yet scores lower on the composite (0.5811 against 0.5891) | **Not reproducible as stated** | With the corrected scorer the two composites are 0.6211 and 0.6224 (a tie). In the float32 re-run oracle evidence did not raise accuracy at all (0.200 for both). |
| No weighting of the four components passes both falsification tests | **Stands** | 0.0% of 1,771 weightings in the recorded runs, the corrected estimates, and the re-run. |
| Execution accuracy 0.340 | **Not stable** | The same model, items, prompts and seed in float32 gave 0.200. Single-run accuracy at n = 50 should not be quoted without an interval. |

## 3. Re-analysis

See `results/scale/SUMMARY.md` for the full tables. It contains, for several models, both the dev and the
test splits of FinQA, and several decoding seeds: bootstrap intervals over items for every headline
statistic, paired item-level noise effects, entailment-based evidence support, and weightings chosen
on dev and evaluated on test.

