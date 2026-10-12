"""
Weight-sensitivity sweep for the ONRS composite.

The two falsification tests in Experiments 4 and 5 were run with the frozen
default weights (0.30 accuracy / 0.30 faithfulness / 0.20 hallucination /
0.20 calibration). Those weights were assigned by judgement, so a natural
objection is that the composite failed only because the weights were wrong.

This script answers that objection without re-running the model. It re-scores
the saved per-condition component values under every weight vector on a
simplex grid and asks, for each one:

  1. Does ONRS fall as evidence noise rises?  (Experiment 4, Pearson r over
     the five noise levels, threshold configurable with --r-threshold)
  2. Does ONRS rank the three Experiment 5 conditions in their known order,
     oracle > retrieval > closed-book?

It also reports the noise correlation of each component on its own, which
bounds what any linear weighting can achieve: ONRS is a weighted sum of the
components, so it cannot track noise unless some component does.

Uses nrf.calculate_onrs() directly, and checks that the default weights
reproduce the ONRS values stored in the result files before sweeping.

Usage:
    python -m experiments.weight_sensitivity [--step 0.05] [--r-threshold -0.5]
"""

import argparse
import csv
import json
import math
import os
import statistics
import sys
from itertools import product
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from framework.nrf import NumericalReliabilityFramework  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
# Overridden by --results-dir so the same sweep can be run on another set of
# results (for example results/cpu_fp32); the defaults are the recorded runs.
EXP4_CSV = RESULTS_DIR / "experiment4_results_dev.csv"
EXP5_JSON = RESULTS_DIR / "experiment5_results_dev_summary.json"
OUT_CSV = RESULTS_DIR / "weight_sensitivity_dev.csv"
OUT_JSON = RESULTS_DIR / "weight_sensitivity_dev_summary.json"

KEYS = ("accuracy", "faithfulness", "hallucination", "calibration")
DEFAULT_WEIGHTS = {"accuracy": 0.30, "faithfulness": 0.30, "hallucination": 0.20, "calibration": 0.20}
RANK_ORDER = ("oracle", "retrieval", "closed_book")


def load_exp4():
    """One row per noise level: (noise_level, components, stored_onrs)."""
    rows = []
    with open(EXP4_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((
                float(r["noise_level"]),
                {
                    "accuracy": float(r["accuracy"]),
                    "faithfulness": float(r["evidence_support"]),
                    "hallucination_rate": float(r["hallucination_rate"]),
                    "ece": float(r["ece"]),
                },
                float(r["onrs"]),
            ))
    return rows


def load_exp5():
    """{condition: (components, stored_onrs)}."""
    with open(EXP5_JSON) as f:
        conditions = json.load(f)["conditions"]
    return {
        name: (
            {
                "accuracy": c["accuracy"],
                "faithfulness": c["evidence_support"],
                "hallucination_rate": c["hallucination_rate"],
                "ece": c["ece"],
            },
            c["onrs"],
        )
        for name, c in conditions.items()
    }


def simplex_grid(step):
    """Every weight vector over KEYS with entries on a `step` grid summing to 1."""
    n = round(1 / step)
    if not math.isclose(n * step, 1.0):
        raise ValueError(f"step must divide 1 evenly, got {step}")
    for a, b, c in product(range(n + 1), repeat=3):
        d = n - a - b - c
        if d >= 0:
            yield dict(zip(KEYS, (a / n, b / n, c / n, d / n)))


def component_scores(comp):
    """Per-component contribution in ONRS orientation (higher is better)."""
    return {
        "accuracy": comp["accuracy"],
        "faithfulness": comp["faithfulness"],
        "hallucination": 1.0 - comp["hallucination_rate"],
        "calibration": 1.0 - comp["ece"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", type=float, default=0.05, help="grid spacing on the weight simplex")
    parser.add_argument("--r-threshold", type=float, default=-0.5,
                        help="ONRS 'tracks noise' when Pearson r(noise, ONRS) <= this")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="read Experiment 4/5 results from, and write the sweep to, this directory "
                             "(default: results/)")
    args = parser.parse_args()

    if args.results_dir is not None:
        global EXP4_CSV, EXP5_JSON, OUT_CSV, OUT_JSON
        d = args.results_dir.resolve()
        EXP4_CSV, EXP5_JSON = d / EXP4_CSV.name, d / EXP5_JSON.name
        OUT_CSV, OUT_JSON = d / OUT_CSV.name, d / OUT_JSON.name

    nrf = NumericalReliabilityFramework()
    exp4 = load_exp4()
    exp5 = load_exp5()

    # Sanity check: the default weights must reproduce the stored ONRS values,
    # otherwise the sweep would be re-scoring something other than what was run.
    for level, comp, stored in exp4:
        got = nrf.calculate_onrs(**comp)
        if not math.isclose(got, stored, abs_tol=1e-9):
            raise SystemExit(f"Exp4 noise {level}: default weights give {got}, file says {stored}")
    for name, (comp, stored) in exp5.items():
        got = nrf.calculate_onrs(**comp)
        if not math.isclose(got, stored, abs_tol=1e-9):
            raise SystemExit(f"Exp5 {name}: default weights give {got}, file says {stored}")

    noise = [level for level, _, _ in exp4]

    component_r = {
        k: statistics.correlation(noise, [component_scores(comp)[k] for _, comp, _ in exp4])
        for k in KEYS
    }

    rows = []
    for w in simplex_grid(args.step):
        onrs_noise = [nrf.calculate_onrs(**comp, weights=w) for _, comp, _ in exp4]
        r = statistics.correlation(noise, onrs_noise)
        onrs5 = {name: nrf.calculate_onrs(**comp, weights=w) for name, (comp, _) in exp5.items()}
        ranking_ok = all(onrs5[a] > onrs5[b] for a, b in zip(RANK_ORDER, RANK_ORDER[1:]))
        tracks = r <= args.r_threshold
        rows.append({
            **{f"w_{k}": w[k] for k in KEYS},
            "noise_r": r,
            "onrs_oracle": onrs5["oracle"],
            "onrs_retrieval": onrs5["retrieval"],
            "onrs_closed_book": onrs5["closed_book"],
            "tracks_noise": tracks,
            "ranking_correct": ranking_ok,
            "passes_both": tracks and ranking_ok,
        })

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    passing = [r for r in rows if r["passes_both"]]
    weights_of = lambda r: {k: r[f"w_{k}"] for k in KEYS}  # noqa: E731
    default_row = next(r for r in rows if all(math.isclose(r[f"w_{k}"], DEFAULT_WEIGHTS[k]) for k in KEYS))

    summary = {
        "split": "dev",
        "source_files": [EXP4_CSV.name, EXP5_JSON.name],
        "grid_step": args.step,
        "n_weight_vectors": n,
        "r_threshold": args.r_threshold,
        "component_noise_correlation": component_r,
        "default_weights": {
            "weights": DEFAULT_WEIGHTS,
            "noise_r": default_row["noise_r"],
            "ranking_correct": default_row["ranking_correct"],
        },
        "fraction_tracking_noise": sum(r["tracks_noise"] for r in rows) / n,
        "fraction_ranking_correct": sum(r["ranking_correct"] for r in rows) / n,
        "fraction_passing_both": len(passing) / n,
        "most_negative_noise_r": {
            "noise_r": min(r["noise_r"] for r in rows),
            "weights": weights_of(min(rows, key=lambda r: r["noise_r"])),
        },
        "most_negative_noise_r_among_correct_rankings": min(
            (r["noise_r"] for r in rows if r["ranking_correct"]), default=None
        ),
        "passing_both_examples": [
            {"weights": weights_of(r), "noise_r": r["noise_r"]}
            for r in sorted(passing, key=lambda r: r["noise_r"])[:10]
        ],
        "note": (
            "Five noise levels and three conditions of n=50 each. A weighting that passes "
            "both tests here has been selected on the same data it is evaluated on, so it "
            "is a hypothesis for a held-out run, not a validated metric."
        ),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("Noise correlation of each component on its own (higher-is-better orientation):")
    for k, r in component_r.items():
        print(f"  {k:<13} r = {r:+.3f}")
    print(f"\nDefault weights: r = {default_row['noise_r']:+.3f}, "
          f"ranking correct = {default_row['ranking_correct']}")
    print(f"\nOf {n} weight vectors (step {args.step}):")
    print(f"  tracks noise (r <= {args.r_threshold}): {summary['fraction_tracking_noise']:.1%}")
    print(f"  ranks oracle > retrieval > closed-book: {summary['fraction_ranking_correct']:.1%}")
    print(f"  passes both:                            {summary['fraction_passing_both']:.1%}")
    best = summary["most_negative_noise_r"]
    print(f"\nMost negative r achievable: {best['noise_r']:+.3f} at {best['weights']}")
    best_ranked = summary["most_negative_noise_r_among_correct_rankings"]
    if best_ranked is not None:
        print(f"Most negative r among weightings that rank correctly: {best_ranked:+.3f}")
    print(f"\nWrote {OUT_CSV} and {OUT_JSON}".replace(str(RESULTS_DIR.parent) + os.sep, ""))


if __name__ == "__main__":
    main()
