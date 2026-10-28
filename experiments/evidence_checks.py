"""
Do checks on the evidence itself notice damaged evidence?

experiments/paper_analysis.py shows that checks on the response cannot. This script asks the question of
the three checks in framework/evidence_consistency.py, which look at the evidence instead:

    redundancy        needs only the evidence
    number_source     needs a trusted copy of the source (here, the clean retrieved evidence)
    pair_source       needs a trusted *structured* copy (here, the clean table cells)

For every run in results/scale/ whose Experiment 4 file records an evidence edit (numbers scaled, periods
swapped, rows swapped) it reports the AUROC for telling the edited evidence of an item from the clean
evidence of the same item, at each edit level, with a 95% bootstrap interval over items.

Usage:
    python -m experiments.evidence_checks [--root results/scale] [--out results/paper/evidence_checks.json]
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.paper_analysis import boot_auroc  # noqa: E402
from framework.evidence_consistency import number_source_check, pair_source_check, redundancy_rate  # noqa: E402

csv.field_size_limit(sys.maxsize)
MODES = ("corrupt", "period_swap", "row_swap")
CHECKS = {
    "redundancy": ("needs only the evidence", lambda ev, src: redundancy_rate(ev)),
    "number_source": ("needs a trusted copy of the source", number_source_check),
    "pair_source": ("needs a trusted structured copy", pair_source_check),
}


def run_files(root):
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in d.glob("experiment4_results_*_per_example.csv"):
            yield d.name, f


def analyse_run(path, boot, rng):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows or rows[0].get("injection_mode") not in MODES:
        return None
    mode = rows[0]["injection_mode"]
    levels = sorted({float(r["noise_level"]) for r in rows})
    clean = {r["id"]: r["evidence_text"] for r in rows if float(r["noise_level"]) == levels[0]}
    out = {"mode": mode, "n_items": len(clean), "levels": {}}
    for level in levels[1:]:
        edited = {r["id"]: r["evidence_text"] for r in rows if float(r["noise_level"]) == level}
        ids = [i for i in clean if i in edited]
        res = {}
        for name, (needs, fn) in CHECKS.items():
            score, label, group = [], [], []
            for i in ids:
                score += [fn(clean[i], clean[i]), fn(edited[i], clean[i])]
                label += [1.0, 0.0]
                group += [i, i]
            # items on which the check is defined (it needs numbers, or table cells, in the evidence)
            defined = sum(not math.isnan(fn(clean[i], clean[i])) for i in ids)
            res[name] = {"needs": needs, "coverage": defined / len(ids), **boot_auroc(score, label, group, boot, rng)}
        res["share_items_edited"] = sum(clean[i] != edited[i] for i in ids) / len(ids)
        out["levels"][str(int(level))] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/scale"))
    ap.add_argument("--out", type=Path, default=Path("results/paper/evidence_checks.json"))
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261028)
    result = {}
    for name, path in run_files(args.root):
        r = analyse_run(path, args.boot, rng)
        if r:
            result[name] = r
            top = r["levels"][max(r["levels"], key=int)]
            print(f"{name:28s} {r['mode']:12s} n={r['n_items']:3d} edited={top['share_items_edited']:.2f}  " + "  ".join(
                f"{k}={v['auroc']:.2f}(cov {v['coverage']:.2f})" for k, v in top.items() if k != "share_items_edited"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
