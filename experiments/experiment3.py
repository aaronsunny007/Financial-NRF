"""
experiments/experiment3.py

EXPERIMENT 3: Confidence Calibration and Overconfidence Analysis

Research question:
    "Can the NRF accurately identify and quantify overconfidence in a
    Financial RAG system?"

Objective:
    Validate the Calibration module. Overconfidence in incorrect financial
    data is a high-risk failure mode, so the module's job is to SURFACE and
    QUANTIFY it.

WHAT "SUCCESS" LOOKS LIKE FOR THIS EXPERIMENT -- read this before
interpreting the numbers. This validates a measurement instrument, not the
system's performance. If the model is overconfident, a correctly-working
calibration module reports a LARGE ECE and a large confidence-accuracy
gap. An ECE of 0.0000 would mean there is no overconfidence present, which
gives the module nothing to detect and therefore validates nothing. A
large measured overconfidence gap here is the experiment succeeding at its
stated objective.

Note also that "Accuracy = 1.0 and Mean Confidence on Errors = 0.0" is not
an achievable target but a contradiction: if accuracy is 1.0 there are no
errors, so confidence-on-errors is undefined (an empty set), not zero.
This script reports it as NaN with an explicit count in that case rather
than printing a misleading 0.

FIXES IN THIS VERSION:

1. GOLD ANSWER. Now uses the shared experiments/finqa_gold.py
   resolve_gold_answer() -- the same rule experiment1 uses -- instead of
   the raw display string qa["answer"]. The previous version therefore
   scored against a different (and in percentage cases, wrong) gold value
   than experiment1 did, making the two experiments' accuracy figures
   non-comparable. See finqa_gold.py for the two scaling bugs involved.
   Non-numeric (yes/no) items are excluded from accuracy explicitly.

2. --split (default dev) instead of a hardcoded test split, so iterative
   work does not contaminate the held-out test set.

3. OVERCONFIDENCE DECOMPOSITION. The research question asks whether the
   framework can *quantify* overconfidence, so this version reports the
   quantities that actually express it, rather than ECE alone:
       - mean confidence on CORRECT answers
       - mean confidence on INCORRECT answers
       - the confidence gap between them (a discrimination measure: can
         the model's own confidence separate its right answers from its
         wrong ones at all?)
       - the overconfidence gap (mean confidence - accuracy), the direct
         quantity the phrase "overconfidence" names
       - the count of high-confidence errors (confidence >= 0.8 while
         wrong), which is the concrete high-risk failure mode in a
         financial setting
   A near-zero confidence gap is itself a strong finding: it means the
   model's self-reported confidence carries essentially no information
   about whether it is right.

Retained from the previous fixed version: missing confidence is NEVER
defaulted to 0.5. framework/nrf.py's extract_self_reported_confidence()
returns None, and such examples are excluded from calibration statistics
with the exclusion count reported, because a missing confidence is not the
same as a genuine 0.5 and defaulting it biases ECE toward looking
better-calibrated than the model is.
"""

import sys
import csv
import json
import math
import shutil
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

HIGH_CONFIDENCE_THRESHOLD = 0.8


def backup_existing_results(output_path: Path) -> None:
    if not output_path.exists():
        return
    backup_path = output_path.with_name(output_path.stem + "_prototype" + output_path.suffix)
    if backup_path.exists():
        return
    shutil.copy2(output_path, backup_path)
    print(f"Preserved previous results as: {backup_path}")


def _mean(values):
    values = list(values)
    return (sum(values) / len(values)) if values else float("nan")


def _fmt(x):
    return "N/A" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.4f}"


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Confidence Calibration")
    parser.add_argument("--mode", choices=["dev", "full"], default="dev",
                        help=f"Sample SIZE: dev={DEV_SAMPLES}, full={FULL_SAMPLES} (default: dev)")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev",
                        help="FinQA SPLIT to load (default: dev). Reserve test for final reported runs.")
    parser.add_argument("--n", type=int, default=None, help="Override sample count directly")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--engine", choices=["local", "anthropic"], default="local",
                        help="Generation backend (see rag_pipeline/rag.py).")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_examples = args.n if args.n is not None else (DEV_SAMPLES if args.mode == "dev" else FULL_SAMPLES)
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "results" / f"experiment3_results_{args.split}.csv"
    )

    if args.split == "test":
        print("!" * 60)
        print("WARNING: running against the TEST split -- final reported runs only.")
        print("!" * 60)

    print("=" * 60)
    print("EXPERIMENT 3: CONFIDENCE CALIBRATION AND OVERCONFIDENCE ANALYSIS")
    print(f"Split: {args.split} | N: {n_examples} | Seed: {args.seed} | Engine: {args.engine}")
    print("=" * 60)
    print("NOTE: 'confidence' here is the model's SELF-REPORTED value, elicited in")
    print("the prompt. It is NOT derived from token log-probabilities. Report it as")
    print("'ECE of self-reported confidence' -- see framework/nrf.py.")

    set_seed(args.seed)
    data = load_finqa(args.split)[:n_examples]
    print(f"Evaluating {len(data)} examples.")

    rag_kwargs = {"seed": args.seed, "llm_backend": args.engine}
    if args.top_k is not None:
        rag_kwargs["top_k"] = args.top_k
    rag = FinancialRAG(**rag_kwargs)
    nrf = NumericalReliabilityFramework()

    results = []
    for i, item in enumerate(data, 1):
        print(f"\n[{i}/{len(data)}] {item.get('id', '')}")
        result = rag.answer(item)
        response = result.get("response", "")
        gold_answer, answer_type = resolve_gold_answer(item)

        acc = nrf.numerical_accuracy(response, gold_answer)
        confidence = nrf.extract_self_reported_confidence(response)

        results.append({
            "id": item.get("id"),
            "gold_answer": gold_answer,
            "answer_type": answer_type,
            "predicted_value": acc["predicted_value"],
            "correct": acc["correct"],
            "parse_success": acc["parse_success"],
            "self_reported_confidence": confidence,
            "high_confidence_error": int(
                confidence is not None
                and confidence >= HIGH_CONFIDENCE_THRESHOLD
                and not acc["correct"]
                and answer_type == "numeric"
            ),
            "response": response,
        })

        conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"
        print(f"  Correct: {acc['correct']} | Confidence: {conf_str}")

    n = len(results)
    numeric = [r for r in results if r["answer_type"] == "numeric"]
    n_scored = len(numeric)
    excluded_non_numeric = n - n_scored
    accuracy = (sum(r["correct"] for r in numeric) / n_scored) if n_scored else float("nan")

    conf_valid = [r for r in numeric if r["self_reported_confidence"] is not None]
    confidence_missing_count = n_scored - len(conf_valid)

    correct_rows = [r for r in conf_valid if r["correct"]]
    error_rows = [r for r in conf_valid if not r["correct"]]

    mean_confidence = _mean(r["self_reported_confidence"] for r in conf_valid)
    mean_correct_confidence = _mean(r["self_reported_confidence"] for r in correct_rows)
    mean_error_confidence = _mean(r["self_reported_confidence"] for r in error_rows)

    # Discrimination: does self-reported confidence separate right from wrong?
    confidence_gap = (
        mean_correct_confidence - mean_error_confidence
        if correct_rows and error_rows else float("nan")
    )
    # Overconfidence: how far does stated confidence exceed actual accuracy?
    overconfidence_gap = (
        mean_confidence - accuracy
        if conf_valid and not math.isnan(accuracy) else float("nan")
    )

    high_conf_errors = sum(r["high_confidence_error"] for r in results)
    high_conf_error_rate = (high_conf_errors / n_scored) if n_scored else float("nan")

    ece = (
        nrf.calculate_ece(
            [r["self_reported_confidence"] for r in conf_valid],
            [r["correct"] for r in conf_valid],
        )
        if conf_valid else float("nan")
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 RESULTS")
    print("=" * 60)
    print(f"N examples:                          {n}")
    print(f"  scored (numeric gold):              {n_scored}")
    print(f"  excluded (non-numeric gold):        {excluded_non_numeric}")
    print(f"Accuracy:                            {_fmt(accuracy)}")
    print(f"Mean Self-Reported Confidence:       {_fmt(mean_confidence)} "
          f"({confidence_missing_count} missing, excluded)")
    print()
    print("--- Overconfidence decomposition (this is what the RQ asks for) ---")
    print(f"Mean confidence on CORRECT answers:  {_fmt(mean_correct_confidence)}  (n={len(correct_rows)})")
    print(f"Mean confidence on INCORRECT answers:{_fmt(mean_error_confidence)}  (n={len(error_rows)})")
    print(f"Confidence gap (correct - incorrect):{_fmt(confidence_gap)}")
    print("    -> near zero means self-reported confidence carries almost no")
    print("       information about whether the answer is actually right.")
    print(f"Overconfidence gap (conf - accuracy):{_fmt(overconfidence_gap)}")
    print("    -> how far stated confidence exceeds real accuracy.")
    print(f"High-confidence errors (conf>={HIGH_CONFIDENCE_THRESHOLD}):    {high_conf_errors} "
          f"({_fmt(high_conf_error_rate)} of scored)")
    print("    -> the concrete high-risk failure mode: confidently wrong.")
    print()
    print(f"ECE (of self-reported confidence):   {_fmt(ece)}")
    print("=" * 60)

    backup_existing_results(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "engine": args.engine,
            "n": n,
            "n_scored_numeric": n_scored,
            "excluded_non_numeric_count": excluded_non_numeric,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "confidence_missing_count": confidence_missing_count,
            "mean_confidence_on_correct": mean_correct_confidence,
            "mean_confidence_on_errors": mean_error_confidence,
            "confidence_gap_correct_minus_error": confidence_gap,
            "overconfidence_gap_confidence_minus_accuracy": overconfidence_gap,
            "high_confidence_error_count": high_conf_errors,
            "high_confidence_error_rate": high_conf_error_rate,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "ece_of_self_reported_confidence": ece,
            "confidence_note": "Self-reported confidence elicited in the prompt, NOT token "
                               "log-probabilities. Report as 'ECE of self-reported confidence'.",
        }, f, indent=2, default=str)

    print(f"\nPer-example results saved to: {output_path}")
    print(f"Summary saved to:             {summary_path}")


if __name__ == "__main__":
    main()
