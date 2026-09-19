---
title: Numerical Reliability Framework for Financial RAG
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Numerical Reliability Framework (NRF)

Evaluation framework for retrieval-augmented financial question answering,
decomposing reliability into execution accuracy, numerical evidence support
(NESS), rule-based hallucination detection, and calibration of self-reported
confidence, aggregated into a proposed Overall Numerical Reliability Score
(ONRS).

Evaluated on FinQA using a locally hosted 3B instruction-tuned generator under
4-bit quantisation on consumer hardware.

MSc project — Aaron Prakash Gandi, School of Computing, Ulster University.
Supervisor: Dr. Tazar Hussain.

## Modes

**Results explorer** (default). Loads recorded experiment runs from `results/`.
No model, no GPU, instant. This is the mode intended for hosted demonstration.

**Live pipeline** (opt-in via `NRF_ENABLE_LIVE=1`). Runs the real retrieval,
generation and verification stack. Requires the dependencies in
`requirements-live.txt`. On a free CPU tier this generates at roughly 1–3
tokens/second, so a single answer takes minutes; on a CUDA machine it runs at
normal speed.

## Local run

```bash
pip install -r requirements.txt
python app.py                       # explorer only
NRF_ENABLE_LIVE=1 python app.py     # with live pipeline (needs requirements-live.txt)
```
