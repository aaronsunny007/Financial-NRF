"""
experiments/experiment2.py

EXPERIMENT 2: Hallucination Detection Sensitivity

Research question:
    "How sensitive is the NRF's Hallucination Detection module to
    adversarial noise?"

Objective:
    Validate the Evidence Faithfulness / Hallucination module by
    introducing controlled adversarial noise and observing whether the
    NRF correctly flags the resulting unfaithful reasoning.

WHAT "SUCCESS" LOOKS LIKE FOR THIS EXPERIMENT -- read this before
interpreting the numbers. This is a DETECTOR VALIDATION experiment, not a
system-performance experiment. The hypothesis under test is that the
hallucination module's flag rate RISES when adversarial decoy numbers are
injected. A hallucination rate of 0.0000 under adversarial conditions
would mean the detector never fires, i.e. the module FAILED validation --
you cannot validate a detector on data containing no positive cases.
Degradation under noise is the expected and desired observation here.

FIXES IN THIS VERSION (all confirmed by direct measurement, not assumed):

1. CATASTROPHIC CHUNKING BUG (invalidated all previous runs of this
   script). The previous version did:
       modified_item["pre_text"] = <a single string>
   but FinQA's pre_text is a LIST of sentences, and
   rag_pipeline.rag.chunk_document() iterates it with
       for i, sentence in enumerate(pre_text)
   Iterating a STRING in Python yields one CHARACTER at a time, so every
   character of the document became its own retrieval chunk (verified:
   a 57-character string produced 57 single-character chunks). Retrieval
   was therefore matching the question against individual letters. This
   is why previous runs of this experiment reported ~2% accuracy -- that
   figure measured shredded input, not adversarial-noise sensitivity.
   Fixed by appending the adversarial sentences to the pre_text LIST, so
   each injected sentence becomes one chunk and competes with genuine
   evidence on equal footing (which is also a more faithful adversarial
   setup than dumping a blob).

2. PAIRED CONTROL CONDITION. The previous version ran only the noisy
   condition, so "sensitivity to adversarial noise" had to be inferred by
   comparing against experiment1's numbers -- a different script, a
   different split, and a different gold-answer rule. This version runs
   EVERY example twice, clean and noisy, in the same process against the
   same examples, and reports the per-condition figures plus the delta.
   Sensitivity is a within-run paired comparison, which is what the
   research question actually asks for.

3. GOLD ANSWER. Now uses the shared experiments/finqa_gold.py
   resolve_gold_answer(), the same one experiment1 uses, instead of the
   raw display string. See that module for the two percentage-scaling
   bugs this avoids. Non-numeric (yes/no) items are excluded from
   accuracy explicitly rather than silently scored.

4. --split (default dev) instead of a hardcoded test split, so iterative
   work does not contaminate the held-out test set, and --ness-mode to
   select which NESS definition drives evidence support / hallucination.

Retained from the previous fixed version: `used_decoy_number` flags
whether the model's Final Answer numerically matches an injected decoy --
a direct, checkable signal of "the model reasoned over adversarial noise
it was told to ignore", which comparing against the gold answer alone
cannot distinguish from any other wrong answer.
"""

import sys
import csv
import json
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

DECOY_VALUES = [918.7, 431.2, 77.4, 12840, 6320, 284.6, 1523.9]

# Each decoy is its OWN sentence so it becomes its own retrieval chunk,
# matching how genuine FinQA pre_text sentences are chunked. Phrased in the
# same register as real filing text so the adversarial test is meaningful:
# noise that is trivially distinguishable by wording would test nothing.
ADVERSARIAL_SENTENCES = [
    "revenue from unrelated discontinued activities was 918.7 million .",
    "operating expenses attributable to unrelated segments were 431.2 million .",
    "net income from unrelated activities was 77.4 million .",
    "total assets of unrelated activities were 12840 million .",
    "debt attributable to unrelated activities was 6320 million .",
    "cash flow from unrelated activities was 284.6 million .",
    "capital expenditure on unrelated activities was 1523.9 million .",
]


def apply_adversarial_noise(item: dict) -> dict:
    """Return a COPY of `item` with adversarial sentences appended to the
    pre_text LIST.

    Note what is deliberately NOT done here: pre_text is not flattened to a
    string, post_text is not blanked, and the table is untouched. The only
    difference between the clean and noisy conditions is the presence of
    the injected sentences -- which is what makes the paired comparison a
    controlled one.
    """
    modified = dict(item)
    modified["pre_text"] = list(item.get("pre_text", []) or []) + list(ADVERSARIAL_SENTENCES)
    return modified


def backup_existing_results(output_path: Path) -> None:
    if not output_path.exists():
        return
    backup_path = output_path.with_name(output_path.stem + "_prototype" + output_path.suffix)
    if backup_path.exists():
        return
    shutil.copy2(output_path, backup_path)
    print(f"Preserved previous results as: {backup_path}")


def evaluate_condition(rag, nrf, item: dict, scored_item: dict, condition: str,
                       extra_evidence: str = None) -> dict:
    """Score one example under one condition.

    `item` is what the model sees at the DOCUMENT level (possibly
    noise-injected); `extra_evidence` is noise injected into the EVIDENCE
    block after retrieval. `scored_item` is always the ORIGINAL item, so
    the gold answer never changes between conditions.
    """
    result = rag.answer(item, extra_evidence=extra_evidence)
    response = result.get("response", "")
    evidence_text = result.get("evidence_text", "")
    gold_answer, answer_type = resolve_gold_answer(scored_item)

    acc = nrf.numerical_accuracy(response, gold_answer)
    support = nrf.numerical_evidence_support(response, evidence_text)
    halluc = nrf.hallucination_detection(response, evidence_text)
    confidence = nrf.extract_self_reported_confidence(response)

    predicted_value = acc["predicted_value"]
    used_decoy = predicted_value is not None and any(
        abs(predicted_value - d) <= max(0.01, abs(d) * 0.01) for d in DECOY_VALUES
    )

    # Did any injected decoy value appear ANYWHERE in the reasoning, even if
    # the final answer avoided it? A stricter signal of contamination than
    # the final answer alone.
    response_numbers = [p.value for p in nrf.extract_numbers(response)]
    decoy_mentioned = any(
        any(abs(rv - d) <= max(0.01, abs(d) * 0.01) for rv in response_numbers)
        for d in DECOY_VALUES
    )

    return {
        "id": scored_item.get("id"),
        "condition": condition,
        "gold_answer": gold_answer,
        "answer_type": answer_type,
        "predicted_value": predicted_value,
        "correct": acc["correct"],
        "parse_success": acc["parse_success"],
        "evidence_support": support.score,
        "evidence_support_strict": support.strict_score,
        "evidence_support_derived": support.derived_score,
        "unsupported_number_count": support.unsupported_count,
        "unsupported_numbers": json.dumps(support.unsupported_numbers),
        "hallucination": halluc["hallucination"],
        "used_decoy_number": int(used_decoy),
        "decoy_mentioned_in_reasoning": int(decoy_mentioned),
        "self_reported_confidence": confidence,
        "retrieved_chunk_count": len(result.get("retrieved_chunks", [])),
        # Whether any decoy actually reached the model. Without this you
        # cannot distinguish "the detector didn't fire" from "there was
        # nothing to fire on" -- the two have opposite interpretations.
        "decoy_present_in_evidence": int(any(
            str(d) in evidence_text for d in DECOY_VALUES
        )),
        "evidence_text": evidence_text,
        "response": response,
    }


def summarise(rows: list, label: str) -> dict:
    n = len(rows)
    numeric = [r for r in rows if r["answer_type"] == "numeric"]
    n_scored = len(numeric)
    return {
        "condition": label,
        "n": n,
        "n_scored_numeric": n_scored,
        "accuracy": (sum(r["correct"] for r in numeric) / n_scored) if n_scored else float("nan"),
        "evidence_support": sum(r["evidence_support"] for r in rows) / n,
        "evidence_support_strict": sum(r["evidence_support_strict"] for r in rows) / n,
        "evidence_support_derived": sum(r["evidence_support_derived"] for r in rows) / n,
        "hallucination_rate": sum(r["hallucination"] for r in rows) / n,
        "decoy_usage_rate": sum(r["used_decoy_number"] for r in rows) / n,
        "decoy_mention_rate": sum(r["decoy_mentioned_in_reasoning"] for r in rows) / n,
        # The manipulation check: did the attack actually reach the model?
        "decoy_reached_model_rate": sum(r["decoy_present_in_evidence"] for r in rows) / n,
        "invalid_format_rate": sum(1 for r in rows if not r["parse_success"]) / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Hallucination Detection Sensitivity")
    parser.add_argument("--mode", choices=["dev", "full"], default="dev",
                        help=f"Sample SIZE: dev={DEV_SAMPLES}, full={FULL_SAMPLES} (default: dev)")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev",
                        help="FinQA SPLIT to load (default: dev). Reserve test for final reported runs.")
    parser.add_argument("--n", type=int, default=None, help="Override sample count directly")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--ness-mode", choices=["strict", "derived"], default="strict",
                        help="Which NESS definition drives evidence support and hallucination "
                             "detection. Both are always recorded. See framework/nrf.py.")
    parser.add_argument("--engine", choices=["local", "anthropic"], default="local",
                        help="Generation backend (see rag_pipeline/rag.py).")
    parser.add_argument("--conditions", nargs="+",
                        choices=["clean", "doc_injection", "evidence_injection"],
                        default=["clean", "doc_injection", "evidence_injection"],
                        help="Which conditions to run (default: all three). "
                             "'clean' = untouched control. "
                             "'doc_injection' = distractors added to the document, so retrieval may "
                             "filter them -- tests the pipeline END-TO-END. "
                             "'evidence_injection' = distractors forced into the evidence block after "
                             "retrieval -- tests the DETECTOR, which is what the research question asks. "
                             "Both injection conditions are needed: doc_injection alone cannot validate "
                             "the detector if retrieval removes the noise before the model sees it.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_examples = args.n if args.n is not None else (DEV_SAMPLES if args.mode == "dev" else FULL_SAMPLES)
    output_path = Path(args.output) if args.output else (
        PROJECT_ROOT / "results" / f"experiment2_results_{args.split}.csv"
    )

    if args.split == "test":
        print("!" * 60)
        print("WARNING: running against the TEST split -- final reported runs only.")
        print("!" * 60)

    print("=" * 60)
    print("EXPERIMENT 2: HALLUCINATION DETECTION SENSITIVITY")
    print(f"Split: {args.split} | N: {n_examples} | Seed: {args.seed} | "
          f"Engine: {args.engine} | NESS mode: {args.ness_mode}")
    conditions = list(args.conditions)
    print(f"Conditions: {conditions}  (total generations: {n_examples * len(conditions)})")
    print("=" * 60)

    set_seed(args.seed)
    data = load_finqa(args.split)[:n_examples]
    print(f"Evaluating {len(data)} FinQA examples per condition.")

    rag_kwargs = {"seed": args.seed, "llm_backend": args.engine}
    if args.top_k is not None:
        rag_kwargs["top_k"] = args.top_k
    rag = FinancialRAG(**rag_kwargs)
    nrf = NumericalReliabilityFramework(
        credit_derived_numbers=(args.ness_mode == "derived")
    )
    if hasattr(rag, "nrf"):
        rag.nrf.credit_derived_numbers = (args.ness_mode == "derived")

    all_rows = []
    by_condition = {}

    for condition in conditions:
        print("\n" + "-" * 60)
        print(f"CONDITION: {condition.upper()}")
        print("-" * 60)
        rows = []
        for i, item in enumerate(data, 1):
            if condition == "doc_injection":
                model_item, extra = apply_adversarial_noise(item), None
            elif condition == "evidence_injection":
                model_item = item
                extra = "\n".join(f"- [pre_text] {s}" for s in ADVERSARIAL_SENTENCES)
            else:  # clean
                model_item, extra = item, None
            row = evaluate_condition(rag, nrf, model_item, item, condition, extra_evidence=extra)
            rows.append(row)
            all_rows.append(row)

            conf_str = (f"{row['self_reported_confidence']:.2f}"
                        if row["self_reported_confidence"] is not None else "N/A")
            print(f"[{i}/{len(data)}] {row['id']}")
            print(f"  Gold: {row['gold_answer']} | Pred: {row['predicted_value']} | "
                  f"Correct: {row['correct']} | NESS: {row['evidence_support']:.2f} | "
                  f"Halluc: {row['hallucination']} | DecoyReachedModel: {row['decoy_present_in_evidence']} | "
                  f"UsedDecoy: {row['used_decoy_number']} | "
                  f"DecoyInReasoning: {row['decoy_mentioned_in_reasoning']} | Conf: {conf_str}")

        by_condition[condition] = summarise(rows, condition)

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 RESULTS")
    print("=" * 60)
    for condition in conditions:
        s = by_condition[condition]
        print(f"\n[{condition.upper()}]  n={s['n']} (scored numeric: {s['n_scored_numeric']})")
        print(f"  Decoy REACHED the model:  {s['decoy_reached_model_rate']:.4f}   <- manipulation check")
        print(f"  Accuracy:                 {s['accuracy']:.4f}")
        print(f"  Evidence Support (NESS):  {s['evidence_support']:.4f} "
              f"(strict {s['evidence_support_strict']:.4f} / derived {s['evidence_support_derived']:.4f})")
        print(f"  Hallucination Rate:       {s['hallucination_rate']:.4f}")
        print(f"  Decoy used as answer:     {s['decoy_usage_rate']:.4f}")
        print(f"  Decoy cited in reasoning: {s['decoy_mention_rate']:.4f}")
        print(f"  Invalid-format rate:      {s['invalid_format_rate']:.4f}")

    def delta(a_label, c_label):
        a, c = by_condition[a_label], by_condition[c_label]
        return {
            "accuracy_delta": a["accuracy"] - c["accuracy"],
            "evidence_support_delta": a["evidence_support"] - c["evidence_support"],
            "hallucination_rate_delta": a["hallucination_rate"] - c["hallucination_rate"],
            "decoy_usage_delta": a["decoy_usage_rate"] - c["decoy_usage_rate"],
            "decoy_mention_delta": a["decoy_mention_rate"] - c["decoy_mention_rate"],
            "decoy_reached_model_delta": a["decoy_reached_model_rate"] - c["decoy_reached_model_rate"],
        }

    deltas = {}
    for label in ("doc_injection", "evidence_injection"):
        if label in by_condition and "clean" in by_condition:
            deltas[f"{label}_vs_clean"] = delta(label, "clean")

    for label, d in deltas.items():
        print("\n" + "-" * 60)
        print(f"SENSITIVITY: {label}")
        print("-" * 60)
        print(f"  Decoy-reached-model change:   {d['decoy_reached_model_delta']:+.4f}  <- READ THIS FIRST")
        print(f"  Accuracy change:              {d['accuracy_delta']:+.4f}")
        print(f"  Evidence Support change:      {d['evidence_support_delta']:+.4f}")
        print(f"  Hallucination Rate change:    {d['hallucination_rate_delta']:+.4f}")
        print(f"  Decoy-as-answer change:       {d['decoy_usage_delta']:+.4f}")
        print(f"  Decoy-in-reasoning change:    {d['decoy_mention_delta']:+.4f}")

    print("\n" + "-" * 60)
    print("HOW TO READ THIS")
    print("-" * 60)
    print("  Check 'Decoy REACHED the model' FIRST. If it is 0 for a condition, the")
    print("  attack never got past retrieval, and that condition says NOTHING about")
    print("  the detector -- a hallucination rate of 0 there means 'nothing to detect',")
    print("  not 'detector working' and not 'detector blind'. Retrieval filtering the")
    print("  noise is itself a real finding about the pipeline's first line of defence,")
    print("  but it is a DIFFERENT finding from detector sensitivity.")
    print()
    print("  The detector is only genuinely tested where the decoys reached the model")
    print("  (normally the evidence_injection condition). There, a POSITIVE")
    print("  hallucination-rate change means the module fired on contaminated")
    print("  reasoning as intended.")
    print("=" * 60)

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
            "conditions": {k: v for k, v in by_condition.items()},
            "sensitivity_deltas": deltas,
        }, f, indent=2, default=str)

    print(f"\nPer-example results saved to: {output_path}")
    print(f"Summary saved to:             {summary_path}")


if __name__ == "__main__":
    main()
