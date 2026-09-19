# Experimental Evaluation of the Numerical Reliability Framework (NRF)

**Project:** Towards Trustworthy Financial Decision Support: A Numerical Reliability Framework for RAG-Based AI Systems
**Author:** Aaron Prakash Gandi
**Supervisor:** Dr. Tazar Hussain

> **Note on use.** Every figure in this document comes from a recorded run of the
> final, audited codebase. Sample sizes are stated per experiment and differ
> between experiments — this is flagged wherever it matters. Do not merge figures
> from different sample sizes into a single table without noting `n`.

---

## 1. Experimental Setup

### 1.1 System under evaluation

| Component | Choice | Justification |
|---|---|---|
| Generator LLM | Qwen2.5-3B-Instruct, 4-bit NF4 quantised (bitsandbytes, double quantisation) | See §1.2 — hardware-constrained |
| Fallback LLM | Qwen2.5-1.5B-Instruct (fp16) | Loaded automatically with a printed warning if 4-bit load fails; never silent |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (384-dim) | Small (~90 MB), fast, negligible VRAM alongside the LLM |
| Vector index | FAISS `IndexFlatIP` over L2-normalised vectors (exact cosine); NumPy fallback | Corpus per question is a single document (tens of chunks), so exact search is cheap and fully reproducible; approximate indexes (IVF/HNSW) would add complexity and non-determinism for no measurable gain |
| Retrieval depth | TOP_K = 8 chunks | See §2.3 — raising it was tested and reverted |
| Decoding | Greedy (`do_sample=False`), seed 42 | Removes sampling variance so run-to-run differences are attributable to the pipeline |
| Max new tokens | 500 | Raised from 220 → 320 → 500 after confirming multi-step questions were being truncated mid-reasoning before reaching a Final Answer |
| Max prompt tokens | 4096, left-truncation | See §2.6 |
| Prompting | System prompt + 2 few-shot worked examples as prior conversation turns | Few-shot examples target specific observed error classes, not generic "be careful" nudges |
| Verification | Gold-free post-hoc arithmetic verifier + at most 1 corrective retry | See §1.4 |

**Chunking strategy.** FinQA supplies `pre_text` and `post_text` as lists of
sentences already segmented by the dataset authors. Each list element becomes one
chunk rather than being re-split by a heuristic sentence splitter, which risks
breaking numeric expressions such as "$5,829 million" across chunk boundaries.
Each table row becomes one chunk, rendered as `label -- header: value | header:
value` so a retrieved row is interpretable in isolation.

### 1.2 Hardware constraint (material to interpreting all results)

All experiments ran locally on an ASUS TUF Gaming laptop (AMD Ryzen 7, NVIDIA
RTX 3050, **4 GB VRAM**). No paid API budget was available.

This is a genuine, binding constraint on model choice, not a convenience:

- **Phi-3-mini (3.8B, fp16)** was tried first and was impractically slow — almost
  certainly VRAM-exhausted and falling back to CPU/shared-memory paging.
- **Qwen2.5-3B in 4-bit (~2 GB weights)** fits comfortably alongside the embedding
  model and KV cache on a 4 GB card. This is approximately the practical ceiling
  for this hardware.
- **7B-class models** require roughly 4.3–4.7 GB even at 4-bit and do **not** fit;
  they would spill to system RAM and reproduce the Phi-3 slowdown.
- **Frontier models** (GPT-4-class, Claude Opus) are API-only and paid.

An Anthropic API backend (`--engine anthropic`) was implemented so that an
identical-pipeline comparison against a frontier model can be run if budget
becomes available. Retrieval, chunking, scoring, and verification are shared
between backends, so any difference would be attributable to the generator alone.

### 1.3 Dataset and split discipline

FinQA (Chen et al., 2021). All reported experiments use the **dev split**; the
**test split was deliberately reserved** and never used for iterative development,
so that a final reported figure can be produced on data never tuned against.
Sample selection is deterministic (first *n* items, fixed seed), not
cherry-picked.

**Gold answer resolution.** Scoring uses FinQA's precise executed answer
(`qa.exe_ans`) rather than the human-readable display string (`qa.answer`), which
FinQA rounds for readability (e.g. display "14%" vs `exe_ans` 0.14464). This is
the standard execution-accuracy methodology used by FinQA itself and by the
comparison systems in the literature review — it is **not** a loosened tolerance.

Items whose gold answer is non-numeric (yes/no comparison questions, where
`exe_ans` is literally the string `'yes'`/`'no'`) are **excluded from accuracy and
ONRS and reported separately**, never silently counted as correct or incorrect. In
the reported runs this count was 0.

**Tolerance policy.** A prediction counts as correct within
`max(0.01, |gold| × 0.02)`. If exactly one of prediction and gold is a percentage
and the other is not, they are **not** treated as equal even when magnitudes match
(e.g. predicted `32.5` vs gold `32.5%` is a mismatch). This is a deliberate,
conservative policy choice and is flagged as such rather than being an
implementation accident.

### 1.4 Gold-free post-hoc arithmetic verifier

`framework/verifier.py` implements an independent check that **never sees the gold
answer or gold program**. For each response it:

1. extracts the model's own final arithmetic expression;
2. re-executes it in Python using a restricted AST evaluator (`ast.Constant`,
   `BinOp`, `UnaryOp` only — no `eval`);
3. checks each stated operand is grounded in the retrieved evidence, exempting a
   closed set of unit-conversion constants (100, 1000, 10⁶, 12, 4, 360, 365);
4. compares the recomputation against the model's stated Final Answer.

If verification fails, **one** corrective regeneration is attempted, with the
failure reason passed back to the model. Because the verifier is gold-free, this
is a legitimate self-consistency improvement step, not a mechanism for nudging the
model toward a known answer. The retry is bounded at one to cap inference cost;
if the retry also fails, the retry's response is still returned and flagged as
unverified, never silently discarded.

---

## 2. Methodology Audit: Defects Identified and Corrected

> This section is a substantive part of the contribution. Several defects were
> found in the initial implementation that had produced invalid results. Each is
> reported with the concrete evidence that identified it. Reporting these is what
> makes the final figures trustworthy.

### 2.1 Fabricated evidence-support constant

`experiment2.py` and `experiment4.py` computed faithfulness as
`result.get("faithfulness", 0.8)` on a dictionary that **never contained a
`"faithfulness"` key**. The metric therefore returned exactly 0.8 on every
example, in every condition, regardless of model behaviour.

*Consequence:* every previously reported faithfulness figure of exactly 0.8000 was
a hardcoded default, not a measurement. Experiment 4's ONRS could barely move with
noise because 30% of its weight was pinned to a constant.
*Fix:* compute real NESS via `nrf.numerical_evidence_support()`.

### 2.2 Two distinct gold-answer percentage-scaling defects

FinQA is inconsistent about whether `exe_ans` is a raw ratio or already
percent-scaled. Two separate defects were found:

**Defect A — program already scaled.** `DVN/2007/page_58.pdf-2`:

```
program  = "divide(60, 243), multiply(#0, const_100)"
exe_ans  = 24.69136          (already the percentage)
answer   = "24.69%"
```

The scorer multiplied by 100 again, producing a gold value of `2469.136%` and
marking the model's **correct** answer of 24.69 as wrong.

**Defect B — operands already percentages.** `ZBH/2003/page_40.pdf-1`:

```
program  = "subtract(51.2, 47.4)"   (both operands are already percentages)
exe_ans  = 3.8
answer   = "3.8%"
```

No `multiply(..., const_100)` step exists here, so a fix that pattern-matched the
program string still mis-scaled this case to `380%`.

*Final fix:* the scaling decision is made by comparing `exe_ans` against the
numeric magnitude of FinQA's own human-written display answer and selecting
whichever of `{exe_ans, exe_ans × 100}` is closer. The human annotation is a far
more reliable signal than program syntax. Verified against all nine observed
cases; regression tests ship in `experiments/finqa_gold.py`.

### 2.3 Retrieval changes that were tested and reverted (negative results)

Two attempts to improve retrieval were made, verified, and **reverted** because
verification showed they did not work. Both are reported because the reason they
failed is itself a finding.

| Attempt | Intended fix | Measured outcome | Action |
|---|---|---|---|
| TOP_K 8 → 12 | Surface a table row that retrieval missed (`AMT/2005/page_105.pdf-2`) | Did **not** fix the target case — direct embedding-rank measurement confirmed the correct row still ranked outside the top 12 of only 25 chunks, out-ranked by 18 near-duplicate prose sentences. Instead perturbed every prompt: 8 answers flipped to correct, 6 flipped to wrong (net +2/50) | Reverted |
| Guaranteed full-table chunk for small tables | Same target case | Accuracy **fell** to 0.30 from the 0.34 baseline. The change was assumed to be narrow but most FinQA tables are small, so it perturbed most prompts | Reverted |

**Finding derived from these failures:** the 4-bit 3B model's answers are
**unstable under prompt perturbations unrelated to the question being asked** —
approximately **28% of answers (14/50) changed** between two runs differing only
in retrieval depth. This instability is comparable in magnitude to the effects the
experiments attempt to measure, and is quantified again in §6.2.

### 2.4 Catastrophic chunking defect in Experiments 2 and 4

Both scripts assigned a *string* to `item["pre_text"]`, but `chunk_document()`
iterates that field expecting a *list* of sentences:

```python
for i, sentence in enumerate(pre_text):
```

Iterating a Python string yields **one character at a time**. Every document was
therefore shredded into single-character retrieval chunks. This was verified
directly: a 57-character string produced 57 one-character chunks.

Critically, Experiment 4's noise function returned a string even at noise level 0,
so the **0% baseline was also corrupted** — which is why earlier runs reported
~6% accuracy at 0% noise against Experiment 1's ~34% on the identical pipeline.

*Consequence:* **all previous Experiment 2 and Experiment 4 results were invalid.**
They measured character-shredding, not noise sensitivity.
*Fix:* distractor sentences are appended to the `pre_text` **list**, each becoming
its own chunk and competing with genuine evidence on equal footing (also a more
faithful adversarial setup than injecting an undifferentiated blob).

### 2.5 Inconsistent gold answers across experiments

Experiments 2–4 scored against `str(item["qa"]["answer"])` — the rounded display
string — while Experiment 1 used the precise `exe_ans` path. The four experiments'
accuracy figures were therefore not comparable with each other, and Experiments
2–4 did not benefit from either fix in §2.2.

*Fix:* `resolve_gold_answer()` extracted into a shared module
(`experiments/finqa_gold.py`) imported by all four experiments, with its own test
suite.

### 2.6 Other corrected defects

| Defect | Evidence | Fix |
|---|---|---|
| Hallucination conflated with wrongness | `experiment2.py` defined hallucination as `0 if correct else 1` | Use `nrf.hallucination_detection()` (NESS-threshold) consistently across all experiments |
| Outdated ONRS formula | `experiment4.py` hardcoded 0.40/0.35/0.25 weights with **no hallucination term** and an ad-hoc `1 − \|confidence − accuracy\|` calibration proxy | Call `nrf.calculate_onrs()` with the single frozen weighting |
| Verifier blind to prose-stated arithmetic | Regex required `expr = result` in one sentence; the model often states the expression in prose then the number separately under "Final Answer". Confirmed to have **missed a real arithmetic slip** (stated expression evaluated to 8.61, model claimed 10.04) | Rewritten expression extractor + regression test reproducing that exact case |
| Silent prompt truncation | HuggingFace default right-truncation would cut the actual question off the end of the prompt once few-shot examples were added | `tokenizer.truncation_side = "left"` in both load paths |
| Missing confidence defaulted to 0.5 | Biases ECE toward looking better-calibrated than the model is | `extract_self_reported_confidence()` returns `None`; such rows are excluded from calibration statistics and the exclusion count is reported |
| Test-split contamination | Experiments 2–4 hardcoded `load_finqa("test")` | `--split` argument, defaulting to `dev`, with an explicit warning banner when `test` is selected |

### 2.7 Metric revision: NESS derived-number credit

**Structural problem identified.** NESS as originally defined counts a response
number as supported only if it appears *literally* in the retrieved evidence. But
the *answer* to an arithmetic question is a computed value that by definition does
not appear in the source text. Measured on the n=50 run: **zero of the 15 fully
correct answers achieved a strict NESS of 1.0** (mean 0.686). The metric therefore
penalises the system for performing the arithmetic the task requires, and a
perfect system could not score 1.0 — which caps ONRS below 1.0 by construction
(maximum ≈ 0.913 at the observed NESS level).

**Revision.** A response number is additionally credited if it is reachable from
the evidence numbers *the response itself cited* by one elementary arithmetic step
(closed list: `−a`; `a×k`, `a÷k` for k ∈ {100, 1000, 10⁶}; `a+b`, `a−b`, `a×b`,
`a÷b`, `(a+b)/2`, `(a÷b)×100`, `(a−b)/b`, `((a−b)/b)×100`), matched to the
precision at which the value was written.

**Validation against gaming — this is essential to report.** A first
implementation seeded the search with *all* evidence numbers. An adversarial test
measured its false-positive rate on invented numbers at **86.5%** — it would have
credited almost any hallucinated figure. Restricting the seed set to numbers the
response actually cited, plus unit-conversion constants, and using
stated-precision tolerance reduced the false-positive rate to **0.2%** (400
adversarial trials: an invented value alongside four correctly-cited evidence
numbers). This test runs on every execution of `python framework/nrf.py`.

**Reporting rule.** Both scores are computed and recorded on every row and in
every summary, regardless of which drives ONRS. The strict figure remains the
default so previously reported baselines stay reproducible. **The revised figure
must always be reported alongside the strict one and described as a defined metric
revision — never as a corrected version of the same metric.**

---

## 3. Experiment 1 — Baseline Reliability Profiling

**Research question.** What is the baseline numerical reliability of a standard
RAG system on FinQA?

**Configuration.** n = 50, FinQA dev split, seed 42, local Qwen2.5-3B 4-bit.

### 3.1 Results

| Metric | Value | 95% CI (Wilson) |
|---|---|---|
| Accuracy | **0.3400** (17/50) | [0.224, 0.478] |
| Evidence Support (NESS), strict | **0.7145** | — |
| Evidence Support (NESS), revised | **0.9085** | — |
| Hallucination rate (strict NESS) | **0.0800** (4/50) | [0.032, 0.188] |
| Hallucination rate (revised NESS) | **0.0000** | — |
| Invalid-format rate | 0.0000 | — |
| Sign-mismatch count (diagnostic, not counted correct) | 1 | — |
| Verifier verified rate (gold-free) | 0.5600 | — |
| Examples requiring a retry | 26/50 | — |
| Mean self-reported confidence | 0.8960 | — |
| ECE of self-reported confidence | **0.5560** | — |
| ONRS (strict NESS) | **0.5891** | — |
| ONRS (revised NESS) | **0.6634** | — |

ONRS weights: 0.30 accuracy / 0.30 evidence support / 0.20 hallucination / 0.20
calibration (frozen default in `framework/nrf.py`).

### 3.2 Error taxonomy

The 33 incorrect answers decompose into distinct, diagnosable categories rather
than one uniform failure mode. Representative confirmed cases:

| Category | Confirmed example | Detail |
|---|---|---|
| Arithmetic slip | `CME/2017/page_97.pdf-5` | 339235 − 338240 computed as 9935 instead of 995. Correctly flagged unverified by the gold-free verifier; the single retry did not correct it |
| Unit/format omission | `ETR/2004/page_213.pdf-2`, `MSI/2006/page_61.pdf-3` | Computed (59.1/98.0)×100 = 60.31 and (1451/4134)×100 = 35.11 correctly, but omitted the "%" sign, scored as a unit mismatch under the stated policy |
| Wrong row/column selection | `ZBH/2003/page_40.pdf-1` | Question required 2003 (51.2%) and 2001 (47.4%); model used 2002 (48.3%) — off-by-one column read |
| Wrong financial formula | `JPM/2008/page_85.pdf-3` | Used an incorrect capital-ratio denominator |
| Retrieval recall miss | `AMT/2005/page_105.pdf-2` | The precise table row (federal $2,157,503 / state $2,418,012) never entered the top-k; a rounded prose sentence ("$2.2 billion and $2.4 billion") was retrieved and used instead |
| Near-duplicate confusion | `SPGI/2018/page_74.pdf-1` | Two different pension trusts each report a pair of values; both were retrieved and the wrong pair was used |
| Incomplete multi-row aggregation | `AON/2010/page_28.pdf-1` | Two properties both expire in 2020; both rows were retrieved but only the first was used |
| Sign/direction | `PM/2015/page_127.pdf-4` | Averaged three accounting-negative values (shown as `(2207)` etc.) as positives, producing +4088.33 instead of −4088.33. Captured by the `sign_mismatch` diagnostic |
| **Dataset annotation error** | `ETR/2011/page_341.pdf-3` | `gold_inds` shows the underlying value as −0.3 (filing notation `(0.3)`), but FinQA's program is `add(18.9, 0.3)`, giving `exe_ans` 19.2. The correct answer — and the model's answer — is **18.6**. The gold label is wrong |

**Note on the annotation error.** This is a limitation of the benchmark, not of
the system, and is reported as such. No special-casing was applied to individual
test items; doing so would constitute cherry-picking.

---

## 4. Experiment 2 — Hallucination Detection Sensitivity

**Research question.** How sensitive is the NRF's Hallucination Detection module
to adversarial noise?

**Interpretation note — essential.** This is a **detector validation** experiment,
not a system-performance experiment. The hypothesis is that the module's flag rate
*rises* when adversarial decoys are injected. A hallucination rate of 0.0000 under
adversarial conditions would mean the detector never fires — a **failed**
validation. A detector cannot be validated on data containing no positive cases.

**Configuration.** n = 10, dev split, three conditions, strict NESS. Seven decoy
values injected as individually-chunked sentences phrased in filing register.

### 4.1 Design: three conditions and a manipulation check

An initial two-condition design produced **8 of 10 byte-identical responses**
between clean and adversarial, and zero decoys in any response. Investigation
showed retrieval had filtered every injected sentence before generation. A
third condition and an explicit manipulation check (`decoy_present_in_evidence`)
were added, because without it a null result is ambiguous between *"the detector
did not fire"* and *"there was nothing to fire on"* — which carry opposite
interpretations.

### 4.2 Results

| Condition | Decoy reached model | Accuracy | NESS (strict / revised) | Hallucination rate | Decoy used as answer |
|---|---|---|---|---|---|
| Clean (control) | 0.0000 | 0.4000 | 0.7266 / 0.8918 | 0.1000 | 0.0000 |
| Document injection | **0.2000** | 0.4000 | 0.7224 / 0.8918 | 0.1000 | 0.0000 |
| Evidence injection | **1.0000** | 0.5000 | 0.6889 / 0.8662 | **0.2000** | 0.0000 |

### 4.3 Findings

**Finding 1 — retrieval is an effective first line of defence.** Only **2 of 10**
document-level distractors survived retrieval into the evidence block. The
retrieval stage absorbed 80% of a naive injection attack before the generator saw
anything. This is a genuine, quantified property of the RAG architecture.

**Finding 2 — the detector responds when contamination reaches the model.** Under
evidence injection (100% delivery), the hallucination flag rate **doubled**
(0.10 → 0.20) and strict NESS fell (0.727 → 0.689). The module is sensitive; the
original design simply never delivered anything for it to detect.

**Finding 3 — the generator did not take the bait.** Decoy-as-answer and
decoy-in-reasoning were **0.0000 in all three conditions**. The model never
adopted an injected value. The NESS/hallucination movement reflects contaminated
*context*, not contaminated *answers*.

### 4.4 Stated limitations

- **n = 10.** The hallucination-rate change of +0.10 is literally **one additional
  example**. 95% CI for the clean rate is [0.018, 0.404] and for the injected rate
  [0.057, 0.510] — these intervals overlap heavily. The direction is consistent
  with the hypothesis and the mechanism is demonstrated, but the magnitude is not
  established at this sample size. A run at n = 50 is required before quoting the
  delta as a result.
- **Accuracy rose (0.40 → 0.50) under evidence injection.** This must **not** be
  reported as noise improving accuracy. It is a single example flipping
  (`CME/2017`, from an arithmetic slip of 9935 to the correct 995) and is
  consistent with the ~28% prompt-perturbation instability quantified in §2.3.

---

## 5. Experiment 3 — Confidence Calibration and Overconfidence Analysis

**Research question.** Can the NRF accurately identify and quantify overconfidence
in a Financial RAG system?

**Interpretation note.** This validates a measurement instrument. If the system is
overconfident, a correctly-working calibration module reports a *large* ECE and a
*large* confidence–accuracy gap. An ECE of 0 would leave the module nothing to
detect. Note also that "Accuracy = 1.0 with Confidence-on-Errors = 0.0" is not a
target but a contradiction: with no errors, confidence-on-errors is undefined (an
empty set), not zero.

**Configuration.** n = 10, dev split, seed 42.

### 5.1 Results

| Metric | Value |
|---|---|
| Accuracy | 0.4000 (4/10), 95% CI [0.168, 0.687] |
| Mean self-reported confidence | 0.9000 (0 missing) |
| Mean confidence on **correct** answers | 0.9000 (n = 4) |
| Mean confidence on **incorrect** answers | 0.9000 (n = 6) |
| **Confidence gap** (correct − incorrect) | **0.0000** |
| **Overconfidence gap** (confidence − accuracy) | **0.5000** |
| High-confidence errors (conf ≥ 0.8 while wrong) | **6/10 (0.6000)**, 95% CI [0.313, 0.832] |
| ECE of self-reported confidence | **0.5000** |

### 5.2 Findings

**This is the strongest and most robust result in the project.** The model reported
0.90 confidence on **every single example** — on four correct answers and six wrong
ones alike.

**Finding 1 — self-reported confidence has zero discriminative power.** The
confidence gap is exactly **0.0000**. The model's stated confidence carries *no
information whatsoever* about whether its answer is right. This is a categorical
result, not a marginal one.

**Finding 2 — severe, quantified overconfidence.** Stated confidence exceeds actual
accuracy by **0.50**. ECE of 0.5000 confirms this.

**Finding 3 — the high-risk failure mode is the majority case.** **60% of all
scored examples were confidently wrong** (confidence ≥ 0.8 while incorrect). In a
financial decision-support setting, this is the precise failure mode the project
identifies as dangerous: a system that is wrong most of the time while signalling
high certainty.

**Terminology requirement.** This is *self-reported* confidence elicited in the
prompt, **not** a probability derived from token log-probabilities. It must be
labelled "ECE of self-reported confidence" in all figures and text.

### 5.3 Limitation

n = 10. The point estimates are wide (accuracy CI [0.168, 0.687]). However, the
confidence gap of exactly 0.0000 arises from 10/10 identical confidence values,
which is a qualitatively different kind of observation from a noisy point
estimate. A run at n = 50 is recommended; the n = 50 Experiment 1 data
independently corroborates the pattern (mean confidence 0.8960 against accuracy
0.3400, ECE 0.5560).

---

## 6. Experiment 4 — Overall Reliability under Varying Noise Levels

**Research question.** How does the ONRS respond to increasing levels of context
noise?

**Configuration.** n = 50 per noise level (250 generations total), dev split,
evidence-mode injection, strict NESS. Noise level = percentage of a 10-sentence
distractor pool, i.e. levels [0, 10, 20, 30, 40] inject [0, 1, 2, 3, 4] distractor
sentences.

### 6.1 Results

| Noise | Reached model | Accuracy | NESS | Hallucination | ECE | ONRS |
|---|---|---|---|---|---|---|
| 0% | 0.0000 | 0.3400 | 0.7145 | 0.0800 | 0.5560 | 0.5891 |
| 10% | 1.0000 | 0.3000 | 0.7109 | 0.1000 | 0.5800 | 0.5673 |
| 20% | 1.0000 | 0.3800 | 0.7114 | 0.1200 | 0.4920 | 0.6050 |
| 30% | 1.0000 | 0.3600 | 0.6922 | 0.1000 | 0.5240 | 0.5909 |
| 40% | 1.0000 | 0.3200 | 0.7132 | 0.1200 | 0.5680 | 0.5724 |

Total ONRS change 0% → 40%: **−0.0168**. Monotonically non-increasing: **False**.

**Correlations across noise levels:**

| Pair | Correlation |
|---|---|
| Noise level vs ONRS | **−0.10** |
| Noise level vs hallucination rate | **+0.76** |
| Accuracy vs ONRS | **+0.98** |
| Accuracy vs ECE | **−0.97** |

### 6.2 Finding: ONRS does not reliably track context noise — and the cause is structural

The correlation between noise level and ONRS is **−0.10**, i.e. effectively none.
ONRS does not decline monotonically; it oscillates within a 0.0377 band while
accuracy alone varies by 0.0800 (four examples out of fifty) across levels.

**Diagnosis.** Because the system's self-reported confidence is near-constant at
~0.90, ECE becomes an approximately deterministic function of accuracy:

| Noise | Accuracy | ECE | 0.90 − accuracy |
|---|---|---|---|
| 0% | 0.3400 | 0.5560 | 0.5600 |
| 10% | 0.3000 | 0.5800 | 0.6000 |
| 20% | 0.3800 | 0.4920 | 0.5200 |
| 30% | 0.3600 | 0.5240 | 0.5400 |
| 40% | 0.3200 | 0.5680 | 0.5800 |

Correlation between accuracy and ECE: **−0.97**. The calibration term therefore
carries essentially **no information independent of accuracy**. Consequently
ONRS's nominal 0.30 weight on accuracy is in practice approximately **0.50** —
0.30 directly plus 0.20 more entering through the degenerate calibration term.
The composite inherits accuracy's full run-to-run instability (§2.3: ~28% of
answers change under unrelated prompt perturbation), which exceeds the noise
effect at this sample size and noise magnitude. This is confirmed by
correlation(accuracy, ONRS) = **+0.98** — ONRS is, empirically, accuracy in
disguise.

**The hallucination sub-metric, by contrast, does track noise** (r = **+0.76**;
all four noise levels above the clean baseline of 0.0800). The *detection*
component is sound; the *aggregation scheme* is what fails.

### 6.3 Recommended revisions to ONRS (arising directly from this finding)

1. **Replace self-reported confidence with token-probability-derived confidence**,
   so the calibration term carries information independent of accuracy.
2. **Re-weight, or orthogonalise, the composite** so accuracy is not double-counted.
3. **Report component metrics alongside the composite** — the composite obscured a
   real signal (hallucination, r = +0.76) that was visible in its parts.

This is a substantive, evidence-based critique of a metric proposed by this
project, and is presented as such.

### 6.4 Alternative condition available

`--injection-mode document` reproduces the end-to-end condition in which retrieval
may filter the noise. This measures pipeline robustness rather than metric
sensitivity, and the two must not be reported interchangeably.

---

## 7. Justification of the Observed Accuracy

> This section directly addresses the question of whether ~34% accuracy is
> defensible.

### 7.1 Comparison with published FinQA results

| System | Execution accuracy |
|---|---|
| **Human expert** | **91.16%** |
| FinQANet (RoBERTa-large), gold retrieval | 70.0% |
| FinQANet (RoBERTa-large) | 61.24% |
| FinQANet (RoBERTa-base) | 56.1% |
| FinQANet (BERT-large) | 53.52% |
| **General crowd workers** | **50.68%** |
| **This system** (Qwen2.5-3B, 4-bit, laptop-local, full retrieval) | **34.0%** |

Sources: FinQA original paper (Chen et al., 2021) and the FinQA benchmark
leaderboard.

**Points to make in the discussion:**

1. **FinQA is an unsaturated benchmark.** Human experts reach 91.16%; the best
   published model in the original paper reaches 61.24%. No published system
   approaches 100%. A reported accuracy near 100% from any system would warrant
   scrutiny of the evaluation, not celebration.
2. **The comparison systems are not comparable in kind.** FinQANet variants are
   *fine-tuned on FinQA's training set* with a task-specific program-generation
   architecture. This system is a **zero-shot / few-shot general-purpose 3B
   instruct model, 4-bit quantised, with no task-specific fine-tuning**, running
   on a 4 GB consumer laptop GPU. The 27-point gap to FinQANet-RoBERTa-large is
   attributable to fine-tuning, model scale, and quantisation, in that order.
3. **The gap is expected and diagnosable, not anomalous.** §3.2 decomposes it into
   named categories with confirmed examples, including at least one case
   (`ETR/2011`) where the benchmark's own gold label is incorrect.
4. **The accuracy figure survived an audit.** It was not accepted at face value:
   two real scoring defects (§2.2) were found and fixed, which changed which
   answers counted as correct. The current figure is the post-audit one.

### 7.2 Why the other metrics are what they are

| Metric | Value | Justification |
|---|---|---|
| ECE 0.5560 | High | Not a defect — this is the **primary finding** of Experiment 3. It measures a real property of the system: near-constant self-reported confidence against ~34% accuracy. A low ECE here would mean there was no overconfidence to detect and the calibration module could not be validated |
| NESS strict 0.7145 | Below 1.0 | **Structurally cannot reach 1.0** for arithmetic questions: a computed answer does not appear verbatim in source text. Zero of 15 fully-correct answers reached strict NESS 1.0. The revised definition (0.9085) addresses this and is reported alongside |
| Hallucination 0.0800 | Low | This is a *good* result for the baseline condition. It rises appropriately under adversarial conditions (Exp 2: 0.10 → 0.20; Exp 4: r = +0.76 with noise), which is what validates the module |
| ONRS 0.5891 | Below 1.0 | Bounded above by NESS (see above): with observed NESS, the theoretical maximum ONRS is ≈ **0.913** even for a hypothetically perfect system. ONRS = 1.0 is unreachable by construction under the strict definition |
| Verifier verified rate 0.5600 | ~half | A distinct and reportable finding: in ~44% of responses the model's own stated reasoning did not recompute to its own stated answer on first pass. This is a self-consistency measure independent of final correctness |

### 7.3 The ONRS ceiling (report this explicitly)

Under the strict NESS definition, a system with **perfect accuracy, zero
hallucinations, and perfect calibration** would score:

| NESS | Maximum attainable ONRS |
|---|---|
| 0.7145 (observed) | **0.9130** |
| 0.80 | 0.9400 |
| 0.90 | 0.9700 |
| 1.00 | 1.0000 |

Because strict NESS cannot reach 1.0 for computational questions, **ONRS = 1.0 is
mathematically unreachable** under the original metric definition. This is a
property of the metric, quantified here, and is one of the motivations for the
revision in §2.7.

---

## 8. Limitations

1. **Sample size.** Experiments 1 and 4 use n = 50; Experiments 2 and 3 use n = 10.
   The 95% CI on 34% accuracy at n = 50 is [0.224, 0.478]. Sample sizes are stated
   per experiment and must not be merged without annotation.
2. **Model scale is hardware-bound.** 4 GB VRAM caps the generator at ~3B
   parameters in 4-bit (§1.2). Results characterise *this* configuration, not RAG
   on FinQA in general.
3. **Quantisation.** 4-bit NF4 quantisation introduces additional error relative to
   fp16 inference of the same model; this was not separately ablated.
4. **Answer instability.** ~28% of answers change under prompt perturbations
   unrelated to the question (§2.3). This is comparable to, or larger than, several
   of the effects measured, and is the principal threat to validity for
   Experiments 2 and 4.
5. **Single seed.** All runs use seed 42 with greedy decoding. Variance across
   seeds was not characterised.
6. **Dev split only.** The test split is deliberately unused and reserved for a
   final evaluation.
7. **Benchmark annotation noise.** At least one confirmed incorrect gold label
   (`ETR/2011/page_341.pdf-3`). The overall rate of such errors in FinQA was not
   estimated.
8. **NESS is not entailment.** It is a numeric-overlap heuristic. It does not
   verify that a supported number was used in a *logically correct* way — a model
   dividing the right two numbers in the wrong order produces a derivable, and
   therefore "supported", number. Accuracy and the post-hoc verifier are what catch
   that case.
9. **Hallucination detection is a threshold rule**, not a trained classifier.
10. **Noise magnitude.** Experiment 4's maximum condition injects 4 distractor
    sentences alongside 8 retrieved chunks. Larger perturbations were not tested and
    may be required to move ONRS beyond the instability band.

---

## 9. Future Work

1. **Frontier-model comparison.** The `--engine anthropic` backend is implemented
   and shares retrieval, scoring, and verification with the local path, enabling a
   controlled generator-only comparison when budget permits. This would separate
   "limitations of the RAG approach" from "limitations of a 3B quantised model."
2. **Token-probability confidence** to replace self-reported confidence, making the
   calibration term informative (§6.3).
3. **ONRS re-weighting or orthogonalisation** to eliminate the accuracy
   double-count (§6.3).
4. **Table-aware retrieval.** The `AMT/2005` case shows per-row dense retrieval can
   miss the specific row a question needs (e.g. a "total" row) when many
   near-duplicate prose sentences out-rank it. Two general fixes were tried and
   reverted (§2.3); a narrower, row-count-aware approach validated against the
   dev-set table-size distribution remains open.
5. **Multi-seed variance characterisation** to establish a formal noise floor
   against which effect sizes can be judged.
6. **Larger n on Experiments 2 and 3**, and larger noise magnitudes on Experiment 4.

---

## 10. Reproducibility

```bash
# Environment: Python venv on Windows, CUDA-enabled PyTorch
python experiments/finqa_gold.py            # gold-resolution regression tests
python framework/nrf.py                     # metric tests + FP-rate validation
python framework/verifier.py                # verifier regression tests

python -m experiments.experiment1 --split dev --n 50
python -m experiments.experiment2 --split dev --n 50 --conditions clean evidence_injection
python -m experiments.experiment3 --split dev --n 50
python -m experiments.experiment4 --split dev --n 50
```

Optional flags: `--ness-mode {strict,derived}`, `--engine {local,anthropic}`,
`--injection-mode {evidence,document}` (Exp 4), `--seed`, `--top-k`.

**Outputs.** Each experiment writes a per-example CSV and a summary JSON to
`results/`, named by split (and engine, where applicable). Both NESS definitions,
the manipulation check, and all diagnostic fields are recorded on every row, so any
completed run can be re-scored under either metric definition without regenerating
model outputs.

---

## Appendix A — Consolidated results table

| Experiment | n | Key metric | Value |
|---|---|---|---|
| 1 Baseline | 50 | Accuracy | 0.3400 |
| 1 Baseline | 50 | NESS strict / revised | 0.7145 / 0.9085 |
| 1 Baseline | 50 | Hallucination rate | 0.0800 |
| 1 Baseline | 50 | ECE | 0.5560 |
| 1 Baseline | 50 | ONRS strict / revised | 0.5891 / 0.6634 |
| 2 Hallucination sensitivity | 10 | Decoys surviving retrieval (doc injection) | 0.2000 |
| 2 Hallucination sensitivity | 10 | Hallucination, clean → evidence injection | 0.1000 → 0.2000 |
| 2 Hallucination sensitivity | 10 | Decoy adopted as answer (all conditions) | 0.0000 |
| 3 Calibration | 10 | Confidence gap (correct − incorrect) | 0.0000 |
| 3 Calibration | 10 | Overconfidence gap | 0.5000 |
| 3 Calibration | 10 | High-confidence errors | 0.6000 |
| 3 Calibration | 10 | ECE | 0.5000 |
| 4 ONRS under noise | 50 × 5 | ONRS 0% → 40% | 0.5891 → 0.5724 (−0.0168) |
| 4 ONRS under noise | 50 × 5 | corr(noise, ONRS) | −0.10 |
| 4 ONRS under noise | 50 × 5 | corr(noise, hallucination) | +0.76 |
| 4 ONRS under noise | 50 × 5 | corr(accuracy, ECE) | −0.97 |
