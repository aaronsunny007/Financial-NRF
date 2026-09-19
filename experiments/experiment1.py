"""
experiments/experiment1.py

EXPERIMENT 1: Baseline Reliability Profiling

Research question:
    "What is the baseline numerical reliability of a Financial RAG system
    on FinQA?"

This script deliberately does NOT reimplement number parsing, evidence
support, or hallucination logic. All of that lives in
framework/nrf.py so there is exactly one source of truth for each metric.
The previous version of this script had its own copy of "extract_number"
and its own hallucination rule (`0 if correct else 1`), which is how the
`faithfulness = result.get("faithfulness", 0.8)` hardcoding and the
accuracy/hallucination conflation happened — two implementations of the
same concept drifting apart. Fixed here by having exactly one.

KNOWN OPEN ISSUE (see chat, not silently resolved here):
    The ONRS weighting used is the one frozen in framework/nrf.py
    (0.30 accuracy / 0.30 faithfulness / 0.20 hallucination / 0.20
    calibration), NOT the 0.40/0.35/0.25 (no-hallucination-term) formula
    that appeared in the previous version of this script. If you want the
    other formula, pass different weights to
    NumericalReliabilityFramework.calculate_onrs() — do not silently
    change framework/nrf.py's default without updating the dissertation
    text describing it.
"""

import sys
import csv
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from framework.nrf import NumericalReliabilityFramework
from rag_pipeline.rag import FinancialRAG, SEED, set_seed
from data_loader.load_finqa import load_finqa
# resolve_gold_answer now lives in experiments/finqa_gold.py so that
# experiments 1-4 all score against the SAME gold value. Previously this
# logic lived only here, and experiments 2-4 scored against the raw display
# string instead -- making their accuracy figures non-comparable with this
# one and leaving them exposed to the percentage-scaling bugs fixed here.
from experiments.finqa_gold import resolve_gold_answer


DEV_SAMPLES = 10
FULL_SAMPLES = 50


def backup_existing_results(output_path: Path) -> None:
    """If a results file already exists from a previous (prototype) run,
    preserve it rather than silently overwriting it, per the project's
    'freeze preliminary results' step. Only backs up once — if a
    '_prototype' copy already exists, a fresh run's file is just
    overwritten (the prototype snapshot is the one we care about keeping)."""
    if not output_path.exists():
        return
    backup_path = output_path.with_name(output_path.stem + "_prototype" + output_path.suffix)
    if backup_path.exists():
        return
    shutil.copy2(output_path, backup_path)
    print(f"Preserved previous results as: {backup_path}")


def evaluate_example(rag: FinancialRAG, nrf: NumericalReliabilityFramework, item: dict) -> dict:
    result = rag.answer(item)

    response = result.get("response", "")
    evidence_text = result.get("evidence_text", "")
    gold_answer, answer_type = resolve_gold_answer(item)
    gold_answer_display = str(item.get("qa", {}).get("answer", ""))

    acc = nrf.numerical_accuracy(response, gold_answer)
    support = nrf.numerical_evidence_support(response, evidence_text)
    halluc = nrf.hallucination_detection(response, evidence_text)
    confidence = nrf.extract_self_reported_confidence(response)

    # Populated by rag.answer() from framework/verifier.py -- gold-free
    # (never sees the gold answer). See that module's docstring for what
    # verified=True does and does not mean.
    verification = result.get("verification") or {}

    return {
        "id": item.get("id"),
        "question": result.get("question"),
        "gold_answer": gold_answer,                    # precise (exe_ans-derived) value used for scoring
        "gold_answer_display": gold_answer_display,     # original, possibly-rounded FinQA display string
        "answer_type": answer_type,                     # 'numeric' | 'non_numeric' | 'unknown'
        "predicted_value": acc["predicted_value"],
        "correct": acc["correct"],
        "parse_success": acc["parse_success"],       # False => fallback parser was used or no answer found
        "sign_mismatch": acc.get("sign_mismatch", False),  # diagnostic only, does NOT affect `correct`
        "evidence_support": support.score,             # NESS score, NOT "faithfulness" in the semantic sense
        # BOTH NESS definitions are recorded on every row regardless of
        # which one is driving `evidence_support`/ONRS this run, so a
        # completed run can be re-scored under either definition without
        # regenerating anything, and any reported figure can be
        # cross-checked against the other. See framework/nrf.py's
        # numerical_evidence_support docstring for the two definitions.
        "evidence_support_strict": support.strict_score,
        "evidence_support_derived": support.derived_score,
        "directly_supported_count": support.directly_supported_count,
        "derived_supported_count": support.derived_supported_count,
        "derived_numbers": json.dumps(support.derived_numbers),
        "unsupported_number_count": support.unsupported_count,
        "unsupported_numbers": json.dumps(support.unsupported_numbers),
        "hallucination": halluc["hallucination"],
        "self_reported_confidence": confidence,          # may be None -- callers must not silently default this
        "retrieved_chunk_count": len(result.get("retrieved_chunks", [])),
        "verifier_parseable": verification.get("parseable"),
        "verifier_verified": verification.get("verified"),
        "verifier_reason": verification.get("reason"),
        "retry_count": result.get("retry_count", 0),
        "evidence_text": evidence_text,
        "response": response,
    }


def run_experiment(data: list, rag: FinancialRAG, nrf: NumericalReliabilityFramework, split: str = "dev") -> tuple:
    results = []

    for index, item in enumerate(data, start=1):
        print(f"\n[{index}/{len(data)}] {item.get('id', '')}")
        row = evaluate_example(rag, nrf, item)
        results.append(row)

        conf_str = f"{row['self_reported_confidence']:.2f}" if row["self_reported_confidence"] is not None else "N/A"
        print(f"  Gold: {row['gold_answer']} ({row['answer_type']}) | Pred: {row['predicted_value']} | "
              f"Correct: {row['correct']} | ParseOK: {row['parse_success']} | NESS: {row['evidence_support']:.2f} | "
              f"Hallucination: {row['hallucination']} | Verified: {row['verifier_verified']} | "
              f"Retries: {row['retry_count']} | Confidence: {conf_str}")

    if not results:
        return results, None

    n = len(results)

    # Accuracy/ECE/ONRS are only meaningful over questions with a numeric
    # gold answer -- a 'non_numeric' (yes/no-style) question can never be
    # scored correct by a NUMERICAL accuracy metric, and silently
    # including it would just inflate the denominator with an
    # automatic miss rather than reflect model quality. These are
    # reported separately (excluded_non_numeric_count), never dropped
    # from the CSV, and never quietly folded into "correct".
    numeric_results = [r for r in results if r["answer_type"] == "numeric"]
    excluded_non_numeric_count = n - len(numeric_results)
    n_scored = len(numeric_results)

    # Evidence support / hallucination are about groundedness of the
    # response itself, independent of whether the gold answer happens to
    # be numeric -- computed over all N examples.
    faithfulness = sum(r["evidence_support"] for r in results) / n
    # Reported alongside the active figure so both NESS definitions appear
    # in every summary -- the active one is never the only number on record.
    faithfulness_strict = sum(r["evidence_support_strict"] for r in results) / n
    faithfulness_derived = sum(r["evidence_support_derived"] for r in results) / n
    hallucination_rate = sum(r["hallucination"] for r in results) / n
    invalid_format_rate = sum(1 for r in results if not r["parse_success"]) / n
    sign_mismatch_count = sum(1 for r in results if r["sign_mismatch"])
    verifier_verified_rate = sum(1 for r in results if r["verifier_verified"]) / n
    retried_count = sum(1 for r in results if r["retry_count"] > 0)

    accuracy = (sum(r["correct"] for r in numeric_results) / n_scored) if n_scored else float("nan")

    conf_valid = [r for r in numeric_results if r["self_reported_confidence"] is not None]
    confidence_missing_count = n_scored - len(conf_valid)
    mean_confidence = (
        sum(r["self_reported_confidence"] for r in conf_valid) / len(conf_valid)
        if conf_valid else float("nan")
    )

    ece = (
        nrf.calculate_ece(
            [r["self_reported_confidence"] for r in conf_valid],
            [r["correct"] for r in conf_valid],
        )
        if conf_valid else float("nan")
    )

    onrs = None
    if conf_valid:
        onrs = nrf.calculate_onrs(
            accuracy=accuracy,
            faithfulness=faithfulness,
            hallucination_rate=hallucination_rate,
            ece=ece,
        )

    summary = {
        "split": split,
        "engine": getattr(rag, "llm_backend", "local"),
        "llm_model_name": getattr(rag, "llm_model_name", None),
        "n": n,
        "n_scored_numeric": n_scored,
        "excluded_non_numeric_count": excluded_non_numeric_count,
        "accuracy": accuracy,
        "evidence_support": faithfulness,
        "ness_mode": "derived" if getattr(nrf, "credit_derived_numbers", False) else "strict",
        "evidence_support_strict": faithfulness_strict,
        "evidence_support_derived": faithfulness_derived,
        "hallucination_rate": hallucination_rate,
        "invalid_format_rate": invalid_format_rate,
        "sign_mismatch_count": sign_mismatch_count,   # diagnostic only; NOT counted as correct
        "verifier_verified_rate": verifier_verified_rate,  # gold-free: grounded + arithmetic-consistent
        "retried_count": retried_count,
        "mean_self_reported_confidence": mean_confidence,
        "confidence_missing_count": confidence_missing_count,
        "ece_of_self_reported_confidence": ece,
        "onrs": onrs,
        "onrs_weights": "0.30 accuracy / 0.30 evidence_support / 0.20 hallucination / 0.20 calibration (framework/nrf.py default)",
        "gold_answer_source": "qa.exe_ans (precise executed value), scaled x100 when the FinQA display "
                               "answer indicated a percentage; falls back to the display string only when "
                               "exe_ans is unavailable.",
    }
    return results, summary


def write_results_csv(results: list, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id", "question", "gold_answer", "gold_answer_display", "answer_type",
        "predicted_value", "correct",
        "parse_success", "sign_mismatch", "evidence_support",
        "evidence_support_strict", "evidence_support_derived",
        "directly_supported_count", "derived_supported_count", "derived_numbers",
        "unsupported_number_count",
        "unsupported_numbers", "hallucination", "self_reported_confidence",
        "retrieved_chunk_count", "verifier_parseable", "verifier_verified",
        "verifier_reason", "retry_count", "evidence_text", "response",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 RESULTS")
    print("=" * 60)
    print(f"N examples:                       {summary['n']}")
    print(f"  scored (numeric gold):           {summary['n_scored_numeric']}")
    print(f"  excluded (non-numeric gold, e.g. yes/no): {summary['excluded_non_numeric_count']}")
    print(f"Accuracy (over scored examples):  {summary['accuracy']:.4f}")
    print(f"Evidence Support (NESS):          {summary['evidence_support']:.4f} "
          f"[mode: {summary.get('ness_mode', 'strict')}]")
    print(f"  NESS strict  (literal-match only): {summary['evidence_support_strict']:.4f}")
    print(f"  NESS derived (+ derivable-from-evidence): {summary['evidence_support_derived']:.4f}")
    print(f"Hallucination Rate:               {summary['hallucination_rate']:.4f}")
    print(f"Invalid-format rate:              {summary['invalid_format_rate']:.4f}")
    print(f"Sign-mismatch count (diagnostic): {summary['sign_mismatch_count']} (not counted as correct)")
    print(f"Verifier verified rate (gold-free): {summary['verifier_verified_rate']:.4f}")
    print(f"Examples that needed a retry:     {summary['retried_count']}")
    print(f"Mean Self-Reported Confidence:    {summary['mean_self_reported_confidence']:.4f} "
          f"({summary['confidence_missing_count']} missing)")
    print(f"ECE (of self-reported confidence): {summary['ece_of_self_reported_confidence']:.4f}")
    if summary["onrs"] is not None:
        print(f"ONRS:                              {summary['onrs']:.4f}")
        print(f"  weights used:                   {summary['onrs_weights']}")
    else:
        print("ONRS:                              N/A (no examples had a parseable self-reported confidence)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Baseline Reliability Profiling")
    # NOTE: --mode controls SAMPLE SIZE (10 vs 50 examples), and is
    # unrelated to --split, which controls which FinQA file is loaded
    # (dev.json vs test.json). These are separate axes -- don't confuse
    # "--mode dev" (small sample) with "--split dev" (the dev file).
    parser.add_argument("--mode", choices=["dev", "full"], default="dev",
                         help=f"Sample SIZE: dev={DEV_SAMPLES} examples, full={FULL_SAMPLES} examples (default: dev)")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev",
                         help="FinQA SPLIT to load (default: dev). Use --split dev while iterating on the "
                              "prompt/pipeline; reserve --split test for the final, one-time reported "
                              "evaluation, so the number in the dissertation was never tuned on.")
    parser.add_argument("--n", type=int, default=None, help="Override sample count directly")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--ness-mode", choices=["strict", "derived"], default="strict",
                         help="Which NESS definition drives evidence_support, hallucination detection and "
                              "ONRS. 'strict' (default) = a response number counts as supported only if it "
                              "appears literally in the evidence -- the original definition, kept as default "
                              "so previously reported baselines stay reproducible. 'derived' = also credits "
                              "numbers arithmetically derivable from the evidence numbers the response cited "
                              "(so a correctly-computed answer is not punished for not appearing verbatim in "
                              "the source). BOTH scores are recorded in the CSV and summary either way. If you "
                              "report the 'derived' figure, say so explicitly and report the strict one "
                              "alongside it -- they are different metrics, not a corrected version of one.")
    parser.add_argument("--engine", choices=["local", "anthropic"], default="local",
                         help="Generation backend: 'local' (default -- Qwen2.5-3B 4-bit on your GPU) or "
                              "'anthropic' (calls the Anthropic API with Claude Opus instead; requires "
                              "'pip install anthropic' and an ANTHROPIC_API_KEY environment variable). "
                              "Retrieval/chunking/scoring are identical either way, so results are directly "
                              "comparable between engines -- see rag_pipeline/rag.py's module docstring, "
                              "section 11, for why this option exists.")
    parser.add_argument("--output", type=str, default=None,
                         help="Output CSV path (default: results/experiment1_results_<split>.csv, or "
                              "results/experiment1_results_<split>_<engine>.csv for a non-local engine, so "
                              "switching engines never silently overwrites the local baseline's results)")
    args = parser.parse_args()

    n_examples = args.n if args.n is not None else (DEV_SAMPLES if args.mode == "dev" else FULL_SAMPLES)
    if args.output:
        output_path = Path(args.output)
    elif args.engine == "local":
        output_path = PROJECT_ROOT / "results" / f"experiment1_results_{args.split}.csv"
    else:
        output_path = PROJECT_ROOT / "results" / f"experiment1_results_{args.split}_{args.engine}.csv"

    if args.split == "test":
        print("!" * 60)
        print("WARNING: running against the TEST split. This should only be")
        print("done for a final, reported evaluation -- not for iterative")
        print("prompt/pipeline tuning. Use --split dev while iterating.")
        print("!" * 60)

    print("=" * 60)
    print("EXPERIMENT 1: BASELINE RELIABILITY PROFILING")
    print(f"Split: {args.split} | Sample-size mode: {args.mode} | N: {n_examples} | "
          f"Seed: {args.seed} | Engine: {args.engine} | NESS mode: {args.ness_mode}")
    print("=" * 60)

    set_seed(args.seed)

    data = load_finqa(args.split)[:n_examples]
    print(f"Evaluating {len(data)} FinQA examples.")

    rag_kwargs = {"seed": args.seed, "llm_backend": args.engine}
    if args.top_k is not None:
        rag_kwargs["top_k"] = args.top_k
    rag = FinancialRAG(**rag_kwargs)
    nrf = NumericalReliabilityFramework(
        credit_derived_numbers=(args.ness_mode == "derived")
    )
    # The RAG object builds its own NRF instance for the verifier; keep the
    # NESS mode consistent across both so the scored figures and the
    # gold-free verifier are not silently using different definitions.
    if hasattr(rag, "nrf"):
        rag.nrf.credit_derived_numbers = (args.ness_mode == "derived")

    results, summary = run_experiment(data, rag, nrf, split=args.split)

    if not results:
        print("No results generated.")
        return

    print_summary(summary)

    backup_existing_results(output_path)
    write_results_csv(results, output_path)

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nPer-example results saved to: {output_path}")
    print(f"Summary saved to:             {summary_path}")


if __name__ == "__main__":
    main()