"""
experiments/experiment4.py

EXPERIMENT 4: Overall Reliability under Varying Noise Levels

Research question:
    "How does the Overall Numerical Reliability Score (ONRS) respond to
    increasing levels of context noise?"

Objective:
    Validate the ONRS as a holistic metric -- i.e. show it provides a
    single, consistent signal of system health.

WHAT "SUCCESS" LOOKS LIKE FOR THIS EXPERIMENT -- read this before
interpreting the numbers. The hypothesis is that ONRS DECREASES as noise
increases. A flat ONRS across noise levels would be evidence that ONRS
does NOT track system health, i.e. a negative result for the metric
itself. In particular, ONRS = 1.000 at every noise level would not be a
good result; it would demonstrate the metric is insensitive to exactly the
degradation it is supposed to signal, disproving its validity as a health
measure. Observing degradation is this experiment working.

FIXES IN THIS VERSION (all confirmed by direct measurement, not assumed):

1. CATASTROPHIC CHUNKING BUG (invalidated all previous runs, INCLUDING the
   0% baseline). The previous version did:
       modified_item["pre_text"] = add_noise(item, noise_level)  # a string
       modified_item["post_text"] = ""
   but FinQA's pre_text is a LIST of sentences, and
   rag_pipeline.rag.chunk_document() iterates it with
       for i, sentence in enumerate(pre_text)
   Iterating a STRING in Python yields one CHARACTER at a time, so every
   character became its own retrieval chunk (verified: a 57-character
   string produced 57 single-character chunks). Critically, add_noise()
   returned a string even at noise_level=0, so the "0% noise baseline" was
   ALSO shredded -- which is why previous runs reported ~6% accuracy at 0%
   noise against experiment1's ~34% on the same pipeline. Every previous
   number from this script measured character-shredding, not noise
   sensitivity. Fixed by appending noise sentences to the pre_text LIST and
   leaving post_text untouched, so noise_level=0 is now a genuine,
   unmodified baseline identical to experiment1's condition.

2. GOLD ANSWER. Now uses the shared experiments/finqa_gold.py
   resolve_gold_answer(), the same rule experiment1 uses, instead of the
   raw display string. See that module for the two percentage-scaling bugs
   this avoids. Non-numeric (yes/no) items are excluded from accuracy.

3. --split (default dev) instead of a hardcoded test split, and
   --ness-mode to select which NESS definition drives evidence support,
   hallucination detection and hence ONRS.

Retained from the previous fixed version: ONRS comes from
nrf.calculate_onrs() with the frozen default weights (0.30 accuracy / 0.30
evidence support / 0.20 hallucination / 0.20 calibration), NOT a locally
hardcoded formula; and if no example at a noise level yields a parseable
confidence, ECE is NaN and ONRS is reported as None for that level rather
than defaulting ECE to 0 and silently claiming perfect calibration.
"""

import sys
import csv
import json
import math
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from framework.nrf import NumericalReliabilityFramework
from rag_pipeline.rag import FinancialRAG, SEED, set_seed
from data_loader.load_finqa import load_finqa
from experiments.finqa_gold import resolve_gold_answer


DEV_SAMPLES = 10
FULL_SAMPLES = 50

NOISE_LEVELS = [0, 10, 20, 30, 40]

# Each noise value is its own sentence, phrased in filing register, so it
# becomes its own retrieval chunk and competes with genuine evidence on
# equal footing -- the same reasoning as experiment2's adversarial set.
NOISE_SENTENCES = [
    "an unrelated financial figure of 918.7 million was recorded .",
    "an unrelated financial figure of 431.2 million was recorded .",
    "an unrelated financial figure of 77.4 million was recorded .",
    "an unrelated financial figure of 12840 million was recorded .",
    "an unrelated financial figure of 6320 million was recorded .",
    "an unrelated financial figure of 284.6 million was recorded .",
    "an unrelated financial figure of 1523.9 million was recorded .",
    "an unrelated financial figure of 67.8 million was recorded .",
    "an unrelated financial figure of 984.2 million was recorded .",
    "an unrelated financial figure of 4217.5 million was recorded .",
]


def apply_noise(item: dict, noise_level: int) -> dict:
    """Return a COPY of `item` with `noise_level`% of the noise sentences
    appended to the pre_text LIST.

    noise_level=0 returns an unmodified copy -- a genuine baseline. The
    previous version returned a flattened string even at level 0, which
    silently corrupted the baseline condition (see module docstring).

    "Noise level" is defined as a percentage of the available noise
    sentence pool, so with 10 sentences the levels [0,10,20,30,40] inject
    [0,1,2,3,4] distractor sentences respectively. This is stated
    explicitly because "40% noise" is otherwise ambiguous -- it does not
    mean 40% of the document is noise.
    """
    modified = dict(item)
    count = int(len(NOISE_SENTENCES) * noise_level / 100)
    if count <= 0:
        return modified
    modified["pre_text"] = list(item.get("pre_text", []) or []) + list(NOISE_SENTENCES[:count])
    return modified


def _fmt(x):
    return "N/A" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Overall Reliability under Varying Noise Levels")
    parser.add_argument("--mode", choices=["dev", "full"], default="dev",
                        help=f"Sample SIZE per noise level: dev={DEV_SAMPLES}, full={FULL_SAMPLES} (default: dev)")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev",
                        help="FinQA SPLIT to load (default: dev). Reserve test for final reported runs.")
    parser.add_argument("--n", type=int, default=None, help="Override sample count per noise level")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--injection-mode", choices=["evidence", "document"], default="evidence",
                        help="WHERE the noise is injected. 'evidence' (default) forces distractors into "
                             "the evidence block after retrieval, so the model definitely sees them -- "
                             "this is what tests whether ONRS RESPONDS to degradation. 'document' adds "
                             "them to the source document, where retrieval may filter them out before "
                             "the model ever sees them (measured on the dev set: only ~20% got through, "
                             "which is why an earlier document-mode run showed an essentially flat ONRS "
                             "of -0.0013 across all noise levels -- that measured retrieval's filtering, "
                             "not the metric's sensitivity). Use 'document' to study end-to-end pipeline "
                             "robustness; use 'evidence' to validate ONRS as a health signal.")
    parser.add_argument("--ness-mode", choices=["strict", "derived"], default="strict",
                        help="Which NESS definition drives evidence support, hallucination detection "
                             "and hence ONRS. Both are always recorded. See framework/nrf.py.")
    parser.add_argument("--engine", choices=["local", "anthropic"], default="local",
                        help="Generation backend (see rag_pipeline/rag.py).")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_examples = args.n if args.n is not None else (DEV_SAMPLES if args.mode == "dev" else FULL_SAMPLES)
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "results" / f"experiment4_results_{args.split}.csv"
    )

    if args.split == "test":
        print("!" * 60)
        print("WARNING: running against the TEST split -- final reported runs only.")
        print("!" * 60)

    print("=" * 60)
    print("EXPERIMENT 4: OVERALL RELIABILITY UNDER VARYING NOISE")
    print(f"Split: {args.split} | N per noise level: {n_examples} | Seed: {args.seed} | "
          f"Engine: {args.engine} | NESS mode: {args.ness_mode}")
    print(f"Noise levels: {NOISE_LEVELS} | Total generations: {n_examples * len(NOISE_LEVELS)}")
    print(f"Noise definition: level% of a {len(NOISE_SENTENCES)}-sentence distractor pool "
          f"-> {[int(len(NOISE_SENTENCES) * lv / 100) for lv in NOISE_LEVELS]} sentences injected")
    print("=" * 60)

    set_seed(args.seed)
    data = load_finqa(args.split)[:n_examples]

    rag_kwargs = {"seed": args.seed, "llm_backend": args.engine}
    if args.top_k is not None:
        rag_kwargs["top_k"] = args.top_k
    rag = FinancialRAG(**rag_kwargs)
    nrf = NumericalReliabilityFramework(
        credit_derived_numbers=(args.ness_mode == "derived")
    )
    if hasattr(rag, "nrf"):
        rag.nrf.credit_derived_numbers = (args.ness_mode == "derived")

    per_example_rows = []
    level_summaries = []

    for noise_level in NOISE_LEVELS:
        injected = int(len(NOISE_SENTENCES) * noise_level / 100)
        print("\n" + "-" * 60)
        print(f"NOISE LEVEL: {noise_level}%  ({injected} distractor sentence(s) injected)")
        print("-" * 60)

        level_rows = []
        for i, item in enumerate(data, 1):
            if args.injection_mode == "document":
                model_item, extra = apply_noise(item, noise_level), None
            else:  # evidence injection -- bypass retrieval so the noise is seen
                model_item = item
                extra = ("\n".join(f"- [pre_text] {s}" for s in NOISE_SENTENCES[:injected])
                         if injected else None)

            result = rag.answer(model_item, extra_evidence=extra)
            response = result.get("response", "")
            evidence_text = result.get("evidence_text", "")
            gold_answer, answer_type = resolve_gold_answer(item)

            acc = nrf.numerical_accuracy(response, gold_answer)
            support = nrf.numerical_evidence_support(response, evidence_text)
            halluc = nrf.hallucination_detection(response, evidence_text)
            confidence = nrf.extract_self_reported_confidence(response)

            # Manipulation check: did the injected noise actually reach the
            # model? Without this, a flat ONRS is ambiguous between "the
            # metric is insensitive" and "there was no degradation to
            # detect because retrieval removed it". Those have opposite
            # interpretations, so the check is recorded per example.
            noise_reached = int(any(
                str(v) in evidence_text
                for v in ["918.7", "431.2", "77.4", "12840", "6320", "284.6",
                          "1523.9", "67.8", "984.2", "4217.5"][:injected]
            )) if injected else 0

            row = {
                "noise_level": noise_level,
                "injection_mode": args.injection_mode,
                "noise_reached_model": noise_reached,
                "id": item.get("id"),
                "answer_type": answer_type,
                "gold_answer": gold_answer,
                "predicted_value": acc["predicted_value"],
                "correct": acc["correct"],
                "parse_success": acc["parse_success"],
                "evidence_support": support.score,
                "evidence_support_strict": support.strict_score,
                "evidence_support_derived": support.derived_score,
                "hallucination": halluc["hallucination"],
                "self_reported_confidence": confidence,
            }
            level_rows.append(row)
            per_example_rows.append(row)

            conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
            print(f"  [{i}/{len(data)}] Correct: {acc['correct']} | NESS: {support.score:.2f} | "
                  f"Hallucination: {halluc['hallucination']} | Confidence: {conf_str}")

        n = len(level_rows)
        numeric = [r for r in level_rows if r["answer_type"] == "numeric"]
        n_scored = len(numeric)

        accuracy = (sum(r["correct"] for r in numeric) / n_scored) if n_scored else float("nan")
        evidence_support = sum(r["evidence_support"] for r in level_rows) / n
        evidence_support_strict = sum(r["evidence_support_strict"] for r in level_rows) / n
        evidence_support_derived = sum(r["evidence_support_derived"] for r in level_rows) / n
        hallucination_rate = sum(r["hallucination"] for r in level_rows) / n

        conf_valid = [r for r in numeric if r["self_reported_confidence"] is not None]
        confidence_missing_count = n_scored - len(conf_valid)

        if conf_valid and not math.isnan(accuracy):
            ece = nrf.calculate_ece(
                [r["self_reported_confidence"] for r in conf_valid],
                [r["correct"] for r in conf_valid],
            )
            onrs = nrf.calculate_onrs(
                accuracy=accuracy,
                faithfulness=evidence_support,
                hallucination_rate=hallucination_rate,
                ece=ece,
            )
        else:
            ece = float("nan")
            onrs = None

        noise_reached_rate = sum(r["noise_reached_model"] for r in level_rows) / n

        summary = {
            "noise_level": noise_level,
            "injection_mode": args.injection_mode,
            "injected_sentences": injected,
            "noise_reached_model_rate": noise_reached_rate,
            "n": n,
            "n_scored_numeric": n_scored,
            "accuracy": accuracy,
            "evidence_support": evidence_support,
            "evidence_support_strict": evidence_support_strict,
            "evidence_support_derived": evidence_support_derived,
            "hallucination_rate": hallucination_rate,
            "ece": ece,
            "confidence_missing_count": confidence_missing_count,
            "onrs": onrs,
        }
        level_summaries.append(summary)

        print(f"\nNoise reached model: {_fmt(noise_reached_rate)}   <- manipulation check")
        print(f"Accuracy:           {_fmt(accuracy)}")
        print(f"Evidence Support:   {_fmt(evidence_support)} "
              f"(strict {_fmt(evidence_support_strict)} / derived {_fmt(evidence_support_derived)})")
        print(f"Hallucination Rate: {_fmt(hallucination_rate)}")
        print(f"ECE:                {_fmt(ece)} ({confidence_missing_count} missing confidence)")
        print(f"ONRS:               {_fmt(onrs)}")

    print("\n" + "=" * 60)
    print("EXPERIMENT 4 RESULTS")
    print("=" * 60)
    print(f"Injection mode: {args.injection_mode}")
    for s in level_summaries:
        print(f"Noise {s['noise_level']:>2}% | Reached = {_fmt(s['noise_reached_model_rate'])} | "
              f"Accuracy = {_fmt(s['accuracy'])} | "
              f"NESS = {_fmt(s['evidence_support'])} | "
              f"Hallucination = {_fmt(s['hallucination_rate'])} | ONRS = {_fmt(s['onrs'])}")

    # Trend summary -- directly answers "how does ONRS RESPOND to noise?"
    valid = [s for s in level_summaries if s["onrs"] is not None]
    if len(valid) >= 2:
        first, last = valid[0], valid[-1]
        total_change = last["onrs"] - first["onrs"]
        monotonic = all(
            valid[i + 1]["onrs"] <= valid[i]["onrs"] + 1e-9 for i in range(len(valid) - 1)
        )
        print("-" * 60)
        print(f"ONRS at {first['noise_level']}% noise: {_fmt(first['onrs'])}")
        print(f"ONRS at {last['noise_level']}% noise: {_fmt(last['onrs'])}")
        print(f"Total change: {total_change:+.4f}")
        print(f"Monotonically non-increasing across levels: {monotonic}")
        print()
        print("  Interpretation: a NEGATIVE total change means ONRS tracked the")
        print("  injected degradation, i.e. the metric behaved as a health signal.")
        print("  A flat or positive trend would be a negative result FOR THE METRIC,")
        print("  indicating it does not respond to the degradation it should signal.")
        print()
        print("  BUT check the 'Reached' column first. If the noise did not reach the")
        print("  model, a flat ONRS says nothing about the metric -- there was no")
        print("  degradation to signal. Only levels where Reached > 0 test ONRS.")
        reached_any = any(s["noise_reached_model_rate"] > 0 for s in level_summaries[1:])
        if not reached_any:
            print()
            print("  *** WARNING: the injected noise never reached the model at ANY")
            print("  *** noise level. This run does NOT test ONRS sensitivity. Re-run")
            print("  *** with --injection-mode evidence before drawing any conclusion.")
    print("=" * 60)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(level_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(level_summaries)

    per_example_path = output_path.with_name(output_path.stem + "_per_example.csv")
    with open(per_example_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_example_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_example_rows)

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "engine": args.engine,
            "ness_mode": args.ness_mode,
            "noise_definition": f"level% of a {len(NOISE_SENTENCES)}-sentence distractor pool, "
                                f"appended to pre_text as individual sentences",
            "levels": level_summaries,
        }, f, indent=2, default=str)

    print(f"\nPer-noise-level summary saved to: {output_path}")
    print(f"Per-example results saved to:     {per_example_path}")
    print(f"Summary JSON saved to:            {summary_path}")


if __name__ == "__main__":
    main()
