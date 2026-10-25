# Study guide: `scale_analysis.py` and `entailment_scoring.py`

Most of this repository was written with an AI assistant (the README says so). That is fine, but you are the author, so you
should be able to explain what the code does and why, in your own words. This guide walks through the two files step by
step. Read a section, open the file beside it, and then close the guide and say each step out loud. The questions at the end
of each part are the ones a reviewer or interviewer is likely to ask.

Run everything on a real result directory so the outputs mean something:

```
python -m experiments.entailment_scoring --dir results/scale/qwen3bcorrupt_dev --split dev   # needs sentence-transformers
python -m experiments.scale_analysis                                                          # reads every folder under results/scale/
```

---

## Part 1. `experiments/entailment_scoring.py`

**The problem it solves.** The main support score (NESS) asks "does this number appear in the evidence?". That says nothing
about whether the *sentence* the number sits in is backed by the evidence. This file asks the second question with a
natural-language-inference (NLI) model.

**NLI in one line.** An NLI model reads a *premise* and a *hypothesis* and says how likely it is that the premise
*entails* the hypothesis (the other two options are "contradiction" and "neutral").

**Step by step**

1. `extract_claims(response)` picks the sentences of the model's reasoning that make a claim.
   - It cuts the response at "Final Answer:" and drops a leading "Reasoning:", so only the reasoning is scored.
   - It splits into sentences and keeps those that contain a digit, have at least 4 words, are not "Final answer / Confidence /
     Expression" lines, and are **not arithmetic** (`_ARITH` matches things like `8.1 / 56.0 * 100`). Arithmetic is skipped on
     purpose: an NLI model cannot judge a calculation; the arithmetic verifier does that.
   - At most `MAX_CLAIMS = 8` claims per response.
2. `evidence_windows(evidence)` cuts the evidence into chunks of at most `WINDOW_WORDS = 280` words, line by line. The NLI
   model has a limited input length, so the whole evidence cannot be the premise at once.
3. `load_predictor()` loads `cross-encoder/nli-deberta-v3-small`, finds which output is "entailment", and returns a function that
   turns each (premise, hypothesis) pair into an entailment **probability** (softmax of the model's three scores).
4. `score_rows(rows, predict)` does the scoring for each response:
   - Every claim is paired with every evidence window.
   - A claim's support is the **maximum** entailment probability over the windows (one window that backs it is enough).
   - A small cache avoids scoring the same pair twice.
   - Output per response: `nli_claims` (how many claims), `nli_mean_entail` (average support), and `nli_frac_entailed` (the
     share of claims with probability at least `ENTAIL_THRESHOLD = 0.5`). If there are no claims the values are NaN, which
     means "nothing to score", not "unsupported".
5. `_run` and `main` read the per-example files from Experiments 4 and 5, score them, and write
   `experiment4_nli_<split>.csv` and `experiment5_nli_<split>.csv` next to them.

**Check yourself**
- Why take the *maximum* over windows instead of the average?
- Why are arithmetic sentences left out, and what checks them instead?
- What does a NaN in `nli_frac_entailed` mean, and why must it not be treated as 0?
- Why does this score also fail to see corrupted evidence? (The premise *is* the corrupted evidence.)
- What would change if you set the threshold to 0.9?

---

## Part 2. `experiments/scale_analysis.py`

**The problem it solves.** One run on 50 items gives numbers that wobble. This script takes the saved per-item results of
many runs and puts honest uncertainty on every claim, so you can tell a real effect from luck.

**Vocabulary you need**
- **Composite (ONRS).** `0.30 * accuracy + 0.30 * evidence support + 0.20 * (1 - hallucination rate) + 0.20 * (1 - ECE)`.
  These four are the "components"; the weights are `DEFAULT_W`.
- **ECE** (expected calibration error): how far stated confidence is from actual accuracy, averaged over confidence bins.
- **Noise / corruption level.** The experiment-4 condition (0%, 25%, ... of the evidence made worse).
- **Bootstrap.** To get a confidence interval, resample the *items* with replacement many times, recompute the statistic each
  time, and take the 2.5th and 97.5th percentiles. Resampling whole items keeps each item's rows together.
- **AUROC.** The chance that a randomly picked correct answer gets a higher score than a randomly picked wrong one
  (0.5 = no better than a coin).

**Step by step (in file order)**

1. **`load_config(d, split)`** reads one run folder: the Experiment 4 per-example CSV (one row per item and level) and the
   Experiment 5 CSV (one row per item and condition: `closed_book`, `retrieval`, `oracle`). It turns them into grids of shape
   *items x levels* (or *items x 3 conditions*) for correctness, evidence support, hallucination flag, stated confidence, and
   "support against the clean evidence". For Experiment 5 it keeps only items that appear in all three conditions. It also
   loads the NLI file if present. If the Experiment 4 file is missing it returns nothing (the run is skipped).
2. **`components(t, idx)`** takes a chosen set of items and returns the four components per level: accuracy over the numeric
   questions, mean evidence support, `1 - mean hallucination`, and `1 - ECE`.
3. **`corr`** is a Pearson correlation that returns NaN when one side is constant (a constant has no correlation).
4. **`weight_grid()`** lists every way to split weight 1.0 across the four components in steps of 0.05: 1,771 weightings.
5. **`sweep(c4, c5, noise, W)`** asks, for each of those weightings, two questions. (a) *Tracks noise?* Is the correlation
   between the composite and the noise level at most -0.5? (b) *Ranks the conditions correctly?* Is oracle > retrieval >
   closed-book? A weighting "passes both" if the answer to both is yes.
6. **`auroc` and `discrimination`** measure, on clean evidence, how well each gold-free measure (overlap support, "no
   hallucination flag", stated confidence, NLI) separates correct from wrong answers, with bootstrap intervals.
7. **`analyse`** is the main routine for one run.
   - *Point estimates*: the composite and components at each level, correlation with noise, oracle-minus-retrieval gaps.
   - *Paired item-level effects*: for each item, the **last level minus the first level**, then the mean and a bootstrap
     interval over items. This uses every item instead of five level averages, which is why it is more trustworthy than the
     five-point correlation.
   - *Bootstrap of the headline statistics*: resample items, recompute, take percentiles. The weight sweep is bootstrapped
     fewer times (200) because it is slow.
8. **`held_out(dev, test)`** is a guard against fooling yourself. It finds the weightings that pass both tests on **dev**, then
   checks whether they still pass on **test**. A weighting that works only where it was chosen is overfitting.
9. **`parse_name` and `main`** loop over the folders in `results/scale/` (names like `qwen3b_dev`, `qwen1p5b_dev_s2`),
   analyse each, run the held-out check per model, compute the spread across the sampled-decoding seeds (needs three or more
   seeds), and write `results/scale/summary.json`. `experiments/scale_report.py` turns that JSON into `SUMMARY.md`.

**Check yourself**
- Why is a correlation over five noise levels fragile, and what replaces it?
- What exactly is resampled in the bootstrap, and why items rather than rows?
- What does "passes both tests" mean, and why do 0.0% on dev and 41.2% on test (Qwen2.5-3B) show the sweep is unstable?
- What does `held_out` protect against?
- Why is "support against the clean evidence" not a deployable measure, even though it detects corruption?
- In one sentence: what does this whole script let you claim that a single run would not?

---

## Part 3. Make the next change yourself

Doing a small change end to end is the best preparation. Pick one, change the code, run it, and look at the output.

1. In `entailment_scoring.py`, set `ENTAIL_THRESHOLD` to 0.9 and rerun on one folder. How does `nli_frac_entailed` change? Why?
2. In `entailment_scoring.py`, change `MAX_CLAIMS` to 3. Which responses now have fewer claims?
3. In `scale_analysis.py`, change the `r_thr=-0.5` default in `sweep` to -0.8. How does the share of weightings that "track
   noise" change on a run, and what does that tell you about the threshold you chose?
4. Add one more column to the `discrimination` dictionary (for example AUROC of `vs_clean` against correctness) and read the result.

Then commit it under your own name. The commit history is part of the record that you understand the project.
