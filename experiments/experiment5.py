"""
experiments/experiment5.py

EXPERIMENT 5: Comparative Baselines on the Same Data

Research question:
    "How does the NRF-instrumented RAG system compare against alternative
    configurations evaluated on identical data under an identical
    scoring pipeline?"

WHY THIS EXPERIMENT EXISTS. Experiment 1 reports 0.340 execution accuracy
and the report situates that against published FinQA figures (FinQANet
61.24%, FinQANet-Gold 70.0%, human expert 91.16%). That comparison is
informative but it is NOT controlled: those systems are different
architectures, fine-tuned on the FinQA training split, evaluated on the
TEST split, and scored by their own pipeline. Quoting them alongside a
dev-split figure produced by a different scorer compares four things at
once and isolates none of them.

This experiment supplies the controlled comparison. Every condition below
runs on the SAME examples, through the SAME generator, prompt, scoring
code, gold-answer resolution and verifier. The only variable is what
evidence reaches the model (or, for the model-scale condition, which
generator is used). Differences between conditions are therefore
attributable to the manipulated variable rather than to methodology.

CONDITIONS

  closed_book     No evidence at all. Establishes how much of the system's
                  accuracy comes from the retrieved document versus from
                  parametric knowledge and lucky guessing. This is the
                  lower bound: any RAG system must beat it to justify the
                  retrieval stage.

  retrieval       The system as reported in Experiment 1 (top-k dense
                  retrieval). The condition under test.

  oracle          FinQA's own annotated gold evidence (qa["gold_inds"])
                  substituted for retrieval output. This mirrors the
                  "gold retrieval" setting the FinQA authors report
                  separately (70.0% for FinQANet-Gold against 61.24% with
                  their own retriever), which makes it the single most
                  useful comparison point available: it decomposes total
                  error into a RETRIEVAL component (oracle minus
                  retrieval) and a REASONING component (1 minus oracle).

  small_model     Optional (--include-small-model). Re-runs the retrieval
                  condition with the smaller fallback generator, isolating
                  model scale from every other factor. Requires a second
                  model load, so it is off by default.

WHAT THIS DOES AND DOES NOT LICENCE YOU TO CLAIM. It licenses statements
of the form "under identical scoring, supplying gold evidence raises
accuracy from X to Y, so at least (Y-X) of the observed error is
attributable to retrieval rather than reasoning." It does NOT licence
claiming parity with, or a deficit against, any published FinQA number,
because those remain test-split figures from differently-trained systems.
The published figures stay in the report as context; this experiment is
what supports the causal claims.
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

NO_EVIDENCE_TEXT = "(No evidence was supplied for this question.)"


def build_oracle_evidence(item: dict) -> str:
    """Render FinQA's annotated gold evidence in the same format the
    retrieval path produces, so the only difference between the oracle and
    retrieval conditions is WHICH lines appear -- not how they are
    formatted or ordered.

    qa["gold_inds"] maps an evidence id (e.g. "table_2", "text_5") to the
    sentence or rendered row the annotators judged necessary. Falls back to
    an empty string when an item carries no gold_inds, in which case the
    caller skips the item for this condition rather than silently scoring
    it against nothing.
    """
    gold_inds = item.get("qa", {}).get("gold_inds", {}) or {}
    if not gold_inds:
        return ""
    lines = []
    for key in sorted(gold_inds.keys()):
        source = "table_row" if key.startswith("table") else "pre_text"
        text = str(gold_inds[key]).strip()
        if text:
            lines.append(f"- [{source}] {text}")
    return "\n".join(lines)


def backup_existing_results(output_path: Path) -> None:
    if not output_path.exists():
        return
    backup_path = output_path.with_name(output_path.stem + "_prototype" + output_path.suffix)
    if backup_path.exists():
        return
    shutil.copy2(output_path, backup_path)
    print(f"Preserved previous results as: {backup_path}")


def _fmt(x):
    return "N/A" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.4f}"


def evaluate(rag, nrf, item, condition, evidence_override=None):
    result = rag.answer(item, evidence_override=evidence_override)
    response = result.get("response", "")
    evidence_text = result.get("evidence_text", "")
    gold_answer, answer_type = resolve_gold_answer(item)

    acc = nrf.numerical_accuracy(response, gold_answer)
    support = nrf.numerical_evidence_support(response, evidence_text)
    halluc = nrf.hallucination_detection(response, evidence_text)
    confidence = nrf.extract_self_reported_confidence(response)
    verification = result.get("verification") or {}

    return {
        "condition": condition,
        "id": item.get("id"),
        "gold_answer": gold_answer,
        "answer_type": answer_type,
        "predicted_value": acc["predicted_value"],
        "correct": acc["correct"],
        "parse_success": acc["parse_success"],
        "evidence_support": support.score,
        "evidence_support_strict": support.strict_score,
        "evidence_support_derived": support.derived_score,
        "hallucination": halluc["hallucination"],
        "self_reported_confidence": confidence,
        "verifier_verified": verification.get("verified"),
        "retry_count": result.get("retry_count", 0),
        "evidence_char_count": len(evidence_text),
        "response": response,
    }


def summarise(rows, nrf, label):
    n = len(rows)
    numeric = [r for r in rows if r["answer_type"] == "numeric"]
    n_scored = len(numeric)
    accuracy = (sum(r["correct"] for r in numeric) / n_scored) if n_scored else float("nan")
    evidence_support = sum(r["evidence_support"] for r in rows) / n if n else float("nan")
    hallucination_rate = sum(r["hallucination"] for r in rows) / n if n else float("nan")
    verified_rate = sum(1 for r in rows if r["verifier_verified"]) / n if n else float("nan")

    conf_valid = [r for r in numeric if r["self_reported_confidence"] is not None]
    mean_conf = (sum(r["self_reported_confidence"] for r in conf_valid) / len(conf_valid)
                 if conf_valid else float("nan"))
    ece = (nrf.calculate_ece([r["self_reported_confidence"] for r in conf_valid],
                             [r["correct"] for r in conf_valid])
           if conf_valid else float("nan"))
    onrs = (nrf.calculate_onrs(accuracy=accuracy, faithfulness=evidence_support,
                               hallucination_rate=hallucination_rate, ece=ece)
            if conf_valid and not math.isnan(accuracy) else None)

    # Wilson 95% interval on accuracy, so conditions are compared with
    # intervals rather than point estimates -- essential here because the
    # differences of interest are a handful of examples.
    if n_scored:
        k = sum(r["correct"] for r in numeric)
        z = 1.96
        phat = k / n_scored
        denom = 1 + z * z / n_scored
        centre = phat + z * z / (2 * n_scored)
        margin = z * math.sqrt(phat * (1 - phat) / n_scored + z * z / (4 * n_scored * n_scored))
        ci = ((centre - margin) / denom, (centre + margin) / denom)
    else:
        ci = (float("nan"), float("nan"))

    return {
        "condition": label,
        "n": n,
        "n_scored_numeric": n_scored,
        "accuracy": accuracy,
        "accuracy_ci_low": ci[0],
        "accuracy_ci_high": ci[1],
        "evidence_support": evidence_support,
        "hallucination_rate": hallucination_rate,
        "verifier_verified_rate": verified_rate,
        "mean_self_reported_confidence": mean_conf,
        "ece": ece,
        "onrs": onrs,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Comparative Baselines")
    parser.add_argument("--mode", choices=["dev", "full"], default="dev",
                        help=f"Sample SIZE: dev={DEV_SAMPLES}, full={FULL_SAMPLES} (default: dev)")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev",
                        help="FinQA SPLIT (default: dev). Reserve test for final reported runs.")
    parser.add_argument("--n", type=int, default=None, help="Override sample count directly")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--ness-mode", choices=["strict", "derived"], default="strict")
    parser.add_argument("--engine", choices=["local", "anthropic"], default="local")
    parser.add_argument("--conditions", nargs="+",
                        choices=["closed_book", "retrieval", "oracle"],
                        default=["closed_book", "retrieval", "oracle"],
                        help="Which evidence conditions to run (default: all three).")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_examples = args.n if args.n is not None else (DEV_SAMPLES if args.mode == "dev" else FULL_SAMPLES)
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "results" / f"experiment5_results_{args.split}.csv"
    )

    if args.split == "test":
        print("!" * 60)
        print("WARNING: running against the TEST split -- final reported runs only.")
        print("!" * 60)

    conditions = list(args.conditions)

    print("=" * 62)
    print("EXPERIMENT 5: COMPARATIVE BASELINES ON IDENTICAL DATA")
    print(f"Split: {args.split} | N: {n_examples} | Seed: {args.seed} | "
          f"Engine: {args.engine} | NESS mode: {args.ness_mode}")
    print(f"Conditions: {conditions}  (total generations: {n_examples * len(conditions)})")
    print("=" * 62)
    print("All conditions share the same items, generator, prompt, scoring and")
    print("verifier. Only the evidence supplied to the model differs.")
    print("=" * 62)

    set_seed(args.seed)
    data = load_finqa(args.split)[:n_examples]

    rag_kwargs = {"seed": args.seed, "llm_backend": args.engine}
    if args.top_k is not None:
        rag_kwargs["top_k"] = args.top_k
    rag = FinancialRAG(**rag_kwargs)
    nrf = NumericalReliabilityFramework(credit_derived_numbers=(args.ness_mode == "derived"))
    if hasattr(rag, "nrf"):
        rag.nrf.credit_derived_numbers = (args.ness_mode == "derived")

    all_rows = []
    summaries = {}
    skipped_no_gold = 0

    for condition in conditions:
        print("\n" + "-" * 62)
        print(f"CONDITION: {condition.upper()}")
        print("-" * 62)
        rows = []
        for i, item in enumerate(data, 1):
            if condition == "closed_book":
                override = NO_EVIDENCE_TEXT
            elif condition == "oracle":
                override = build_oracle_evidence(item)
                if not override:
                    # No gold evidence annotated -- skip rather than score
                    # against an empty block, which would silently
                    # contaminate the oracle condition with closed-book items.
                    skipped_no_gold += 1
                    print(f"  [{i}/{len(data)}] {item.get('id')} SKIPPED (no gold_inds)")
                    continue
            else:
                override = None

            row = evaluate(rag, nrf, item, condition, evidence_override=override)
            rows.append(row)
            all_rows.append(row)

            conf = row["self_reported_confidence"]
            conf_s = f"{conf:.2f}" if conf is not None else "N/A"
            print(f"  [{i}/{len(data)}] Correct: {row['correct']} | "
                  f"NESS: {row['evidence_support']:.2f} | "
                  f"Halluc: {row['hallucination']} | Conf: {conf_s}")

        summaries[condition] = summarise(rows, nrf, condition)
        s = summaries[condition]
        print(f"\n  Accuracy:        {_fmt(s['accuracy'])}  "
              f"95% CI [{_fmt(s['accuracy_ci_low'])}, {_fmt(s['accuracy_ci_high'])}]")
        print(f"  NESS:            {_fmt(s['evidence_support'])}")
        print(f"  Hallucination:   {_fmt(s['hallucination_rate'])}")
        print(f"  ECE:             {_fmt(s['ece'])}")
        print(f"  ONRS:            {_fmt(s['onrs'])}")

    # ---------------- comparative report ----------------
    print("\n" + "=" * 62)
    print("EXPERIMENT 5 RESULTS -- CONTROLLED COMPARISON")
    print("=" * 62)
    print(f"{'Condition':<14}{'Acc':>8}{'95% CI':>18}{'NESS':>8}{'Halluc':>9}{'ONRS':>9}")
    for c in conditions:
        s = summaries[c]
        ci = f"[{s['accuracy_ci_low']:.2f}, {s['accuracy_ci_high']:.2f}]"
        print(f"{c:<14}{_fmt(s['accuracy']):>8}{ci:>18}"
              f"{_fmt(s['evidence_support']):>8}{_fmt(s['hallucination_rate']):>9}"
              f"{_fmt(s['onrs']):>9}")
    if skipped_no_gold:
        print(f"\n(oracle condition skipped {skipped_no_gold} item(s) lacking gold_inds)")

    # Error decomposition -- the analysis the comparison exists to support
    decomposition = None
    if "retrieval" in summaries and "oracle" in summaries:
        r = summaries["retrieval"]["accuracy"]
        o = summaries["oracle"]["accuracy"]
        cb = summaries.get("closed_book", {}).get("accuracy")
        retrieval_gap = o - r
        reasoning_gap = 1.0 - o
        decomposition = {
            "retrieval_attributable_error": retrieval_gap,
            "reasoning_attributable_error": reasoning_gap,
            "retrieval_contribution_over_closed_book": (r - cb) if cb is not None else None,
        }
        print("\n" + "-" * 62)
        print("ERROR DECOMPOSITION  <- this is what answers the comparison question")
        print("-" * 62)
        if cb is not None:
            print(f"  Closed-book accuracy:                 {_fmt(cb)}")
            print(f"  Gain from retrieval over closed-book: {r - cb:+.4f}")
            print("     -> how much the retrieval stage actually contributes.")
        print(f"  Retrieval-attributable error:         {retrieval_gap:+.4f}")
        print("     -> accuracy recovered by supplying gold evidence instead")
        print("        of retrieved evidence. This is the ceiling on what any")
        print("        retrieval improvement could deliver for this generator.")
        print(f"  Reasoning-attributable error:         {reasoning_gap:.4f}")
        print("     -> error remaining even with perfect evidence. Not")
        print("        addressable by improving retrieval at all.")
        print()
        print("  Compare the oracle row against the gold-retrieval condition")
        print("  reported in the FinQA literature, and the retrieval row")
        print("  against their retriever condition. Note these remain")
        print("  different splits and differently-trained systems: use them")
        print("  as context, not as a like-for-like scoreboard.")
    print("=" * 62)

    backup_existing_results(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "split": args.split,
            "engine": args.engine,
            "ness_mode": args.ness_mode,
            "n_per_condition": n_examples,
            "oracle_skipped_no_gold_inds": skipped_no_gold,
            "conditions": summaries,
            "error_decomposition": decomposition,
        }, f, indent=2, default=str)

    print(f"\nPer-example results saved to: {output_path}")
    print(f"Summary saved to:             {summary_path}")


if __name__ == "__main__":
    main()
