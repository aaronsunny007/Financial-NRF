# Financial-NRF

**A Numerical Reliability Framework for Retrieval-Augmented AI Systems**

MSc dissertation project — Ulster University
Aaron Prakash Gandi · Supervisor: Dr Tazar Hussain

**[Live demo](https://aaronsunny007.github.io/Financial-NRF/)**: score your own evidence and responses in the browser, browse the recorded results, and re-weight the composite yourself.

---

## What this is

Language models are increasingly placed in front of financial documents and asked
numerical questions. The output is a number, not a paragraph, and somebody may act on
it. Existing benchmarks check only whether that final number matches a reference
answer. They do not ask whether the reasoning that produced it was grounded in the
retrieved evidence, whether the arithmetic the model described actually holds, or
whether its stated confidence means anything.

This project builds a measurement layer that asks those questions, applies it to a
retrieval-augmented question answering system over FinQA, and then attempts to falsify
its own headline metric.

**The central finding is negative, and that is the point of the work.** The composite
reliability score proposed here fails two falsification tests: it does not track
reliability when the evidence is degraded, and it ranks a demonstrably better system
below a worse one. The aggregate fails; of the components, the verifier, accuracy and
calibration measurements stand, and the claim that the hallucination flag tracks noise did
not survive a re-run (see below).

---

## The framework

Every generated answer is interrogated four ways:

| Check | Question it asks | Needs the gold answer? |
|---|---|---|
| **Execution accuracy** | Does the produced value match the executed gold program? | Yes |
| **Numerical evidence support** | Does every number in the answer trace back to the retrieved evidence, or a standard conversion of it? | No |
| **Post-hoc arithmetic verification** | Does the model's own stated calculation produce the number it claimed? | No |
| **Calibration (ECE)** | Does stated confidence match observed correctness? | No |

Three of the four need no reference answer, so they can run at inference time on a
question nobody knows the answer to. The arithmetic verifier is the most directly
deployable piece: it parses the expression the model wrote into a syntax tree and
evaluates it under a restricted evaluator permitting only numeric literals and the four
arithmetic operators — never `eval()` on model-generated text.

The four components combine into a composite score weighted `0.30 / 0.30 / 0.20 / 0.20`.
Those weights are assigned by judgement rather than learned, and that is the framework's
weakest design decision.

---

## Results

All figures are on 50 FinQA items per condition. Wilson confidence intervals are
reported throughout the accompanying documents.

| Measure | Value | Condition |
|---|---|---|
| Execution accuracy — ordinary retrieval | 0.340 | baseline |
| Execution accuracy — oracle evidence | 0.420 | gold evidence supplied |
| Composite score — ordinary retrieval | 0.5891 | ranked **first** by the composite |
| Composite score — oracle evidence | 0.5811 | ranked **second**, despite higher accuracy |
| Correlation, noise level vs composite | −0.10 | five graded levels, manipulation-checked; see the scoring defect below |
| Confidence gap, correct vs incorrect | 0.0239 | — |
| Overconfidence gap | 0.5560 | — |
| Errors made at high confidence | 0.6400 | — |
| Verifier outcomes | 56 / 20 / 12 / 12 | verified / ungrounded / no expression / self-contradictory (%) |

### Failure one — the score does not track reliability

Evidence was degraded across five graded noise levels, with a manipulation check
confirming the corrupted evidence actually reached the model. The correlation between
noise level and the composite score is −0.10: effectively no relationship.

The mechanism is structural rather than a coding defect. Evidence support is measured
*against the evidence supplied* rather than against truth, so a model that grounds
itself faithfully in corrupted passages still scores well. The verifier tests only
internal consistency, which a model can satisfy while working from nonsense. Calibration
is a property of the confidence report. Three of the four components are insensitive to
evidence quality by design, leaving only accuracy — 30% of the weight — to respond.

### Failure two — the score inverts the ranking

Supplying oracle evidence raised accuracy from 0.340 to 0.420, yet the composite ranked
that condition *lower*. A score whose purpose is to let you choose between systems chose
the worse one.

### Could different weights fix it? No.

The obvious objection is that the composite failed only because its hand-picked weights
were wrong. `experiments/weight_sensitivity.py` tests that directly. It re-scores the
saved Experiment 4 and 5 results under all 1,771 weightings on a 0.05 grid, with no
model re-run, and checks each weighting against both tests.

| Weightings that… | Share |
|---|---|
| track noise (r ≤ −0.5) | 19.2% |
| rank oracle > retrieval > closed-book | 43.5% |
| **do both** | **0.0%** |

Among the weightings that rank correctly, the strongest noise correlation is −0.16. The
two tests pull in opposite directions. Only hallucination rate (r = −0.76, as recorded; see the scoring defect below) and evidence
support (r = −0.37) respond to noise, but the oracle condition scores *worse* on both
(evidence support 0.628 vs 0.714, hallucination 0.20 vs 0.08), so its advantage shows
only in accuracy and calibration. Accuracy itself barely moves with noise (r = +0.10)
at n = 50. The failure belongs to the components, not the weights. A finer 0.01 grid
and looser thresholds give the same picture: with the threshold relaxed to r ≤ −0.3,
still no weighting passes.

### A scoring defect found after the runs

The evidence-support score counted the model's own `Confidence: 0.9` line as one more
number the evidence had to support. It never could, so every response was marked down
slightly and some were pushed under the hallucination threshold by that number alone.
The scorer is fixed (`exclude_confidence_from_support`, on by default, with tests), but
the recorded runs were scored the old way and have **not** been re-run.
`experiments/confidence_line_audit.py` measures the effect from the saved files.

| What | As recorded | Corrected |
|---|---|---|
| Exp 1 hallucination rate (strict NESS) | 0.08 | **0.00** |
| Exp 2 hallucination, clean → evidence injection | 0.10 → 0.20 | **0.00 → 0.00** |
| Oracle vs retrieval ONRS (default weights) | 0.5811 vs 0.5891 | 0.6211 vs 0.6224 |
| Weightings passing both tests | 0.0% | 0.0% in every scenario |
| Noise vs hallucination correlation | +0.76 | between +0.00 and +0.89 |
| Noise vs ONRS correlation (default weights) | −0.10 | between +0.18 and −0.36 |

What holds: the oracle-vs-retrieval ranking is still inverted, now as a near tie, and
no weighting passes both tests. What does not: the doubling of the hallucination rate in
Experiment 2 was this defect, and the two noise correlations cannot be pinned down
without re-running Experiment 4. Accuracy, calibration, the verifier and Experiment 3
do not use evidence support and are unaffected. The Experiment 4 figures are bracketed
rather than exact because that run did not save its responses; the bracketing method is
checked against the 150 Experiment 5 rows where the exact answer is known.

### Measured re-run with the corrected scorer

The estimates above were then checked by actually re-running all five experiments with the
corrected scorer, for free, on GitHub-hosted CPU runners (`.github/workflows/cpu-rerun.yml`,
outputs in `results/cpu_fp32/`). It is the same model and pipeline in float32 rather than
4-bit, so individual answers differ (15 of 50 baseline answers are identical), and accuracy
came out lower. Read it as a replication on different numerics, not a regeneration.

| Measure | Recorded (4-bit GPU, old scorer) | Re-run (float32 CPU, corrected scorer) |
|---|---|---|
| Execution accuracy, retrieval | 0.340 | 0.200 |
| Execution accuracy, oracle evidence | 0.420 | 0.200 |
| Hallucination rate, retrieval | 0.08 | 0.06 |
| Composite, retrieval vs oracle | 0.5891 vs 0.5811 | 0.5425 vs 0.5316 |
| Noise vs composite, r | −0.10 | **+0.62** |
| Noise vs hallucination rate, r | +0.76 | **−0.12** |
| Weightings that track noise (r ≤ −0.5) | 19.2% | 0.1% |
| Weightings that pass both tests | 0.0% | 0.0% |

What replicated: the composite does not track noise (it moved the other way, because
accuracy drifted up with noise by chance, 0.20 to 0.28), and no weighting passes both
tests. What did not: the hallucination flag does not follow noise (0 to 3 flagged
responses per level, no trend), so the earlier +0.76 should not be relied on. Oracle
evidence did not beat retrieval in this run (both 0.200), so the ranking inversion
cannot be tested here; the composite still scored oracle lower. The 14-point drop in
accuracy from changing only numeric precision is itself a measure of how unstable a 3B
model's answers are at n = 50. A 4-bit GPU re-run is the one remaining like-for-like check.

### What the oracle comparison establishes

In the recorded run, perfect evidence lifts accuracy only to 0.420 (it did not help at all
in the float32 re-run), placing the majority of the residual error in numerical reasoning
rather than retrieval. That decomposition is the useful
result: it says where future work should go.

---

## Repository layout

```
data_loader/      FinQA loading and preparation (dataset files not committed)
rag_pipeline/     Retrieval and generation: embeddings, FAISS index, prompt assembly
framework/
  nrf.py          The four measurements and the composite score
  verifier.py     Gold-free arithmetic verification via restricted AST evaluation
experiments/
  experiment1.py  Baseline evaluation
  experiment2.py  Adversarial injection (clean / document / evidence conditions)
  experiment3.py  Confidence calibration and overconfidence decomposition
  experiment4.py  Noise sensitivity across five graded levels
  experiment5.py  Oracle evidence vs ordinary retrieval — the ranking inversion
  finqa_gold.py   Shared gold-answer resolution, with its own test suite
  weight_sensitivity.py  Re-scores saved results under every ONRS weighting (no GPU)
  confidence_line_audit.py  Measures the confidence-line scoring defect from saved results (no GPU)
results/          Per-item CSV output and summary JSON for every run
deploy/           Gradio results explorer (Hugging Face Space), optional live pipeline
index.html        Static browser demo: live scoring, recorded results, weight sweep
```

Per-item CSVs are committed deliberately: every figure quoted above can be traced back
to the individual answers it came from.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

FinQA is not redistributed here. Download `train.json`, `dev.json` and `test.json` from
[the FinQA repository](https://github.com/czyssrs/FinQA) and place them in `data_loader/`.

Developed against Python 3.11. A CUDA GPU is required for 4-bit quantised inference.

## Running the experiments

```bash
python -m experiments.experiment1 --n 50
python -m experiments.experiment3 --n 50
python -m experiments.experiment4 --n 50 --injection-mode evidence
python -m experiments.experiment5 --n 50
```

Each writes a per-item CSV and a summary JSON into `results/`.

No GPU? The **cpu-rerun** workflow (Actions tab, Run workflow) re-runs all five on free
GitHub-hosted CPUs in full precision, in about two hours, and commits the results to a new
branch. It shards the generation step, then replays the answers through the unchanged
experiment scripts (`--engine replay`). Set `replay_only` to re-score the committed
`results/cpu_fp32/generations.jsonl` without regenerating.

The weight-sensitivity sweep and the test suites need only `numpy`, not a GPU:

```bash
pip install -r requirements-dev.txt
python -m framework.nrf                    # parsing and evidence-support tests
python -m framework.verifier               # arithmetic verifier tests
python -m experiments.finqa_gold           # gold-answer resolution tests
python -m experiments.weight_sensitivity   # re-score saved results under every weighting
python -m experiments.confidence_line_audit  # measure the confidence-line defect
```

CI runs all five on every push and checks that the sweep and the audit reproduce the
committed results.

---

## Stack, and why

| Component | Choice | Reason |
|---|---|---|
| Dataset | FinQA | Ships the executable calculation program per item, not just the answer — which is what makes gold-free verification and the oracle condition possible |
| Embeddings | `all-MiniLM-L6-v2` | Fast on consumer hardware; the oracle experiment confirms retrieval is not the bottleneck |
| Vector search | FAISS `IndexFlatIP` | Exact cosine search, so approximation error cannot confound the results |
| Generator | Qwen2.5-3B-Instruct | Open weights and fully local, so no hosted model can change between runs |
| Quantisation | 4-bit NF4 (bitsandbytes) | Required to fit the model on the available GPU (RTX 3050) |
| Verification | Python `ast` | Parse-and-walk with a restricted node whitelist — never `eval()` on generated text |
| Intervals | Wilson score | Well-behaved at n = 50 and near the bounds, unlike the normal approximation |

---

## Limitations

- **Model scale is a constraint, not a result.** A 3B model at 4-bit precision is small
  for multi-step numerical reasoning. Absolute accuracy would rise with scale; the
  findings about the metric would not change.
- **Fifty items per condition.** Directions are claimable; margins of a few points are
  not, and are not claimed.
- **The recorded runs predate a scoring fix.** The confidence value was counted as an
  unsupported number (see above). The corrected scorer has been run only in float32 on
  CPU (`results/cpu_fp32/`); a 4-bit GPU re-run would make the comparison like for like.
- **Evidence support is numerical overlap, not entailment.** Correct numbers used inside
  a wrong argument pass it.
- **The composite weighting is assigned, not learned.** The sweep above shows that
  no reweighting rescues it on this data, so the next step is better components, not
  better weights: a noise-sensitive evidence measure, and entailment rather than overlap.
  Five noise levels and n = 50 per condition are thin, so the sweep should be repeated
  on a larger held-out run before treating it as settled.

---

## Documents

| File | Contents |
|---|---|
| `NRF_Report_IEEE_v3.docx` | The report, IEEE two-column format |
| `NRF_Supporting_Evidence_v3.docx` | Supporting evidence and extended methodology |
| `NRF_Experimental_Chapter.md` | The experimental chapter in full |

---

## Licence

MIT — see [`LICENSE`](LICENSE).
