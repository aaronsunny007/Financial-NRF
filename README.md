# Financial-NRF

**A Numerical Reliability Framework for Retrieval-Augmented AI Systems**

MSc dissertation project, Ulster University — Aaron Prakash Gandi
Supervisor: Dr Tazar Hussain

---

## What this is

Language models are increasingly placed in front of financial documents and asked
numerical questions. The output is a number, not a paragraph, and somebody may act
on it. Existing benchmarks check only whether that final number matches a reference
answer — they do not ask whether the reasoning that produced it was grounded in the
retrieved evidence, whether the arithmetic the model described actually holds, or
whether its stated confidence means anything.

This project builds a measurement layer that asks those questions, applies it to a
retrieval-augmented question answering system over FinQA, and then attempts to
falsify its own headline metric.

**The central finding is negative, and it is the point of the work.** The composite
reliability score proposed here fails two falsification tests: it does not track
reliability when the evidence is degraded, and it ranks a demonstrably better system
below a worse one. The individual diagnostic components survive; the aggregate does not.

---

## The framework

Every generated answer is interrogated four ways:

| Check | Question it asks | Needs the gold answer? |
|---|---|---|
| **Execution accuracy** | Does the produced value match the executed gold program? | Yes |
| **Numerical evidence support** | Does every number in the answer trace back to the retrieved evidence, or a standard conversion of it? | No |
| **Post-hoc arithmetic verification** | Does the model's own stated calculation produce the number it claimed? | No |
| **Calibration (ECE)** | Does stated confidence match observed correctness? | No |

Three of the four require no reference answer, which means they can run at inference
time on a question nobody knows the answer to. The arithmetic verifier is the piece
most directly deployable: it parses the expression the model wrote into a syntax tree
and evaluates it under a restricted evaluator that permits only numeric literals and
the four arithmetic operators — never `eval()` on model-generated text.

The four components are combined into a composite score with weights of
`0.30 / 0.30 / 0.20 / 0.20`. Those weights are assigned by judgement, not learned,
and that is the framework's weakest design decision.

---

## Results

All figures below are on 50 FinQA items per condition, with Wilson confidence
intervals reported in the accompanying documents.

| Measure | Value | Condition |
|---|---|---|
| Execution accuracy — ordinary retrieval | 0.340 | baseline |
| Execution accuracy — oracle evidence | 0.420 | gold evidence supplied |
| Composite score — ordinary retrieval | 0.5891 | ranked **first** by the composite |
| Composite score — oracle evidence | 0.5811 | ranked **second**, despite higher accuracy |
| Correlation, noise level vs composite | −0.10 | five graded levels, manipulation-checked |
| Confidence gap, correct vs incorrect | 0.0239 | — |
| Overconfidence gap | 0.5560 | — |
| Proportion of errors made at high confidence | 0.6400 | — |
| Verifier outcomes | 56 / 20 / 12 / 12 | verified / ungrounded / no expression / self-contradictory (%) |

### The two failures

**It does not track reliability.** Evidence was degraded across five graded noise
levels, with a manipulation check confirming the corrupted evidence actually reached
the model. The correlation between noise level and the composite score is −0.10 —
effectively no relationship. The mechanism is structural: evidence support is measured
*against the evidence supplied* rather than against truth, the verifier tests only
internal consistency, and calibration is a property of the confidence report. Three of
the four components are insensitive to evidence quality by design, leaving only
accuracy — 30% of the weight — to respond.

**It inverts the ranking.** Supplying oracle evidence raised accuracy from 0.340 to
0.420, yet the composite ranked that condition *lower*. A score whose purpose is to
let you choose between systems chose the worse one.

### What the oracle comparison establishes

Perfect evidence lifts accuracy only to 0.420, which places the majority of the
residual error in numerical reasoning rather than in retrieval. That decomposition is
the useful result: it says where future work should go.

---

## Repository layout

```
data_loader/      FinQA loading and preparation (dataset files not committed)
rag_pipeline/     Retrieval + generation: embeddings, FAISS index, prompt assembly
framework/
  nrf.py          The four measurements and the composite score
  verifier.py     Gold-free arithmetic verification via restricted AST evaluation
experiments/
  experiment1.py  Baseline evaluation
  experiment2.py  Adversarial injection (clean / document / evidence conditions)
  experiment3.py  Confidence calibration and overconfidence decomposition
  experiment4.py  Noise sensitivity across five graded levels
  experiment5.py  Oracle evidence vs ordinary retrieval — the ranking inversion
  finqa_gold.py   Shared gold-answer resolution (with its own test suite)
results/          Per-item CSV output and summary JSON for every run
deploy/           Static browser demo of the scoring logic
```

Per-item CSVs are committed deliberately: every figure quoted above can be traced
back to the individual answers it came from.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

FinQA is not redistributed here. Download `train.json`, `dev.json` and `test.json`
from [the FinQA repository](https://github.com/czyssrs/FinQA) and place them in
`data_loader/`.

## Running the experiments

```bash
python -m experiments.experiment1 --n 50
python -m experiments.experiment3 --n 50
python -m experiments.experiment4 --n 50 --injection-mode evidence
python -m experiments.experiment5 --n 50
```

Each writes a per-item CSV and a summary JSON into `results/`.

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
| Intervals | Wilson score | Stays well-behaved at n = 50 and near the bounds, unlike the normal approximation |

---

## Limitations

- **Model scale is a constraint, not a result.** A 3B model at 4-bit precision is
  small for multi-step numerical reasoning. Absolute accuracy would rise with scale;
  the findings about the metric would not change.
- **Fifty items per condition.** Directions are claimable; margins of a few points
  are not, and are not claimed.
- **Evidence support is numerical overlap, not entailment.** Correct numbers used
  inside a wrong argument pass it.
- **The composite weighting is assigned, not learned.** A sensitivity sweep across
  the weight space, followed by learning the weights against a held-out objective,
  is the most direct future work.

---

## Documents

`NRF_Report_IEEE_v3.docx` — the report, IEEE two-column format
`NRF_Supporting_Evidence_v3.docx` — supporting evidence and extended methodology
`NRF_Experimental_Chapter.md` — the experimental chapter in full

---

## Licence

MIT — see `LICENSE`.
