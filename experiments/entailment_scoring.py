"""
Entailment-based evidence support, as an alternative to numeric overlap.

NESS counts a response number as supported when the same number appears in the evidence. That
says nothing about whether the sentence the number sits in is actually backed by the evidence.
This script scores each factual claim in a response with a natural-language-inference model:
the claim is the hypothesis, the evidence (in windows that fit the model) is the premise, and
a claim's support is the largest entailment probability over the windows.

Claims are the number-bearing sentences of the model's reasoning that are *lookups or
statements*. Sentences that are themselves arithmetic ("8.1 / 56.0 * 100 = 14.46") are left to
the arithmetic verifier, since NLI cannot judge a calculation.

Per response it records:
    nli_claims          number of claims found (0 means there was nothing to score)
    nli_mean_entail     mean entailment probability over claims
    nli_frac_entailed   share of claims with entailment probability >= 0.5

Reads the per-example files of Experiments 4 and 5 (which keep the evidence and the response)
and writes experiment4_nli_<split>.csv / experiment5_nli_<split>.csv next to them.

Usage:
    python -m experiments.entailment_scoring --dir results/<tag> --split dev
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
WINDOW_WORDS = 280
ENTAIL_THRESHOLD = 0.5
MAX_CLAIMS = 8

_ARITH = re.compile(r"[\d%]\s*[+\-*/×÷]\s*[\d($]")
_SKIP_PREFIX = re.compile(r"^\s*(final answer|confidence|expression)\b", re.I)


def extract_claims(response):
    """Number-bearing, non-arithmetic sentences from the reasoning part of a response."""
    if not response:
        return []
    text = re.split(r"final\s*answer\s*[:\-]", response, flags=re.I)[0]
    text = re.sub(r"^\s*reasoning\s*[:\-]\s*", "", text, flags=re.I)
    claims = []
    for sent in re.split(r"(?<=[.!?])\s+(?=[A-Z$(\d])|\n+", text):
        sent = sent.strip()
        if not sent or _SKIP_PREFIX.match(sent):
            continue
        if not re.search(r"\d", sent) or _ARITH.search(sent):
            continue
        if len(sent.split()) < 4:
            continue
        claims.append(sent)
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


def evidence_windows(evidence):
    """Group the evidence lines into premises of at most WINDOW_WORDS words."""
    lines = [ln.strip() for ln in (evidence or "").split("\n") if ln.strip()]
    windows, cur, words = [], [], 0
    for ln in lines:
        n = len(ln.split())
        if cur and words + n > WINDOW_WORDS:
            windows.append("\n".join(cur))
            cur, words = [], 0
        cur.append(ln)
        words += n
    if cur:
        windows.append("\n".join(cur))
    return windows or [""]


def load_predictor():
    """Return predict(pairs) -> list of entailment probabilities."""
    import numpy as np
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(NLI_MODEL, max_length=512)
    id2label = {int(k): v.lower() for k, v in ce.model.config.id2label.items()}
    ent = next(i for i, v in id2label.items() if v.startswith("entail"))

    def predict(pairs):
        if not pairs:
            return []
        logits = np.asarray(ce.predict(pairs, batch_size=32, show_progress_bar=False), dtype=float)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs[:, ent].tolist()

    return predict


def score_rows(rows, predict):
    cache = {}
    out = []
    for r in rows:
        claims = extract_claims(r.get("response", ""))
        wins = evidence_windows(r.get("evidence_text", ""))
        pairs = [(w, c) for c in claims for w in wins]
        todo = [p for p in dict.fromkeys(pairs) if p not in cache]
        for p, v in zip(todo, predict(todo)):
            cache[p] = v
        per_claim = [max(cache[(w, c)] for w in wins) for c in claims]
        out.append({
            "nli_claims": len(claims),
            "nli_mean_entail": (sum(per_claim) / len(per_claim)) if per_claim else float("nan"),
            "nli_frac_entailed": (sum(p >= ENTAIL_THRESHOLD for p in per_claim) / len(per_claim))
                                 if per_claim else float("nan"),
        })
    return out


def _run(path_in, path_out, keys, predict):
    with open(path_in, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = score_rows(rows, predict)
    with open(path_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys + ["nli_claims", "nli_mean_entail", "nli_frac_entailed"])
        w.writeheader()
        for r, s in zip(rows, scores):
            w.writerow({**{k: r.get(k, "") for k in keys}, **s})
    n_scored = sum(1 for s in scores if s["nli_claims"])
    print(f"{path_in.name}: {len(rows)} rows, {n_scored} with at least one claim -> {path_out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    predict = load_predictor()
    d = args.dir
    p4 = d / f"experiment4_results_{args.split}_per_example.csv"
    p5 = d / f"experiment5_results_{args.split}.csv"
    if p4.exists():
        _run(p4, d / f"experiment4_nli_{args.split}.csv", ["noise_level", "id", "correct"], predict)
    if p5.exists():
        _run(p5, d / f"experiment5_nli_{args.split}.csv", ["condition", "id", "correct"], predict)


if __name__ == "__main__":
    main()
