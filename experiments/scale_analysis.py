"""
Analysis of the larger, multi-model, multi-seed re-runs (results/scale/<name>/).

The original five-noise-level correlation is a statistic of five points, so it is fragile. This
script keeps that headline number for continuity but adds what a larger run can support:

  * bootstrap confidence intervals over items for every headline statistic,
  * item-level, paired noise effects (level 40% minus level 0%, per item) for accuracy, evidence
    support, the hallucination flag, support measured against the clean evidence, and
    entailment support,
  * the oracle-versus-retrieval comparison with a paired bootstrap on the same items,
  * the weight sweep with a bootstrap interval on the share of weightings passing both tests,
  * held-out validation: weightings chosen on a model's dev run, evaluated on its test run,
  * spread across decoding seeds, and across models.

Run directories are named <model>_<split> or <model>_<split>_s<seed>, for example qwen3b_dev,
qwen3b_test, qwen1p5b_dev_s2, and hold the per-example files written by Experiments 4 and 5
(and, optionally, the entailment scores).

Usage:
    python -m experiments.scale_analysis [--root results/scale] [--boot 2000] [--boot-sweep 200]
"""

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from framework.nrf import NumericalReliabilityFramework  # noqa: E402
from experiments.weight_sensitivity import KEYS, simplex_grid  # noqa: E402

csv.field_size_limit(sys.maxsize)
NRF = NumericalReliabilityFramework()
DEFAULT_W = np.array([0.30, 0.30, 0.20, 0.20])
RNG_SEED = 20261016


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ loading

def load_config(d, split):
    """Per-item arrays for one run: (ids, levels, tables) or None if the run is incomplete."""
    p4 = d / f"experiment4_results_{split}_per_example.csv"
    p5 = d / f"experiment5_results_{split}.csv"
    if not (p4.exists() and p5.exists()):
        return None
    rows4, rows5 = read_csv(p4), read_csv(p5)
    ids = list(dict.fromkeys(r["id"] for r in rows4))
    levels = sorted({float(r["noise_level"]) for r in rows4})
    pos = {i: k for k, i in enumerate(ids)}
    lv = {l: k for k, l in enumerate(levels)}

    def grid(rows, col, conv=num):
        a = np.full((len(ids), len(levels)), np.nan)
        for r in rows:
            a[pos[r["id"]], lv[float(r["noise_level"])]] = conv(r.get(col))
        return a

    t4 = {
        "correct": grid(rows4, "correct"),
        "ness": grid(rows4, "evidence_support"),
        "hall": grid(rows4, "hallucination"),
        "conf": grid(rows4, "self_reported_confidence"),
        "numeric": grid(rows4, "answer_type", lambda v: 1.0 if v == "numeric" else 0.0),
        "vs_clean": grid(rows4, "evidence_support_vs_clean_strict"),
    }
    conds = ["closed_book", "retrieval", "oracle"]
    cond_rows = {c: {r["id"]: r for r in rows5 if r["condition"] == c} for c in conds}
    ids5 = [i for i in ids if all(i in cond_rows[c] for c in conds)]
    pos5 = {i: k for k, i in enumerate(ids5)}
    t5 = {}
    for key, col in (("correct", "correct"), ("ness", "evidence_support"), ("hall", "hallucination"),
                     ("conf", "self_reported_confidence")):
        a = np.full((len(ids5), 3), np.nan)
        for ci, c in enumerate(conds):
            for i in ids5:
                a[pos5[i], ci] = num(cond_rows[c][i].get(col))
        t5[key] = a
    t5["numeric"] = np.array([[1.0 if cond_rows[c][i].get("answer_type") == "numeric" else 0.0
                               for c in conds] for i in ids5])

    nli4 = nli5 = None
    q4, q5 = d / f"experiment4_nli_{split}.csv", d / f"experiment5_nli_{split}.csv"
    if q4.exists():
        rows = read_csv(q4)
        nli4 = np.full((len(ids), len(levels)), np.nan)
        for r in rows:
            nli4[pos[r["id"]], lv[float(r["noise_level"])]] = num(r["nli_frac_entailed"])
    return {"ids": ids, "levels": levels, "t4": t4, "t5": t5, "ids5": ids5, "nli4": nli4}


# ---------------------------------------------------------------- statistics

def ece_of(conf, correct):
    ok = ~np.isnan(conf)
    if not ok.any():
        return float("nan")
    return NRF.calculate_ece(conf[ok].tolist(), correct[ok].tolist())


def components(t, idx):
    """Four ONRS components per column (level or condition) for the items in idx."""
    cols = t["correct"].shape[1]
    out = np.zeros((cols, 4))
    for j in range(cols):
        num_mask = t["numeric"][idx, j] > 0
        correct = t["correct"][idx, j]
        acc = float(np.nanmean(correct[num_mask])) if num_mask.any() else float("nan")
        ness = float(np.nanmean(t["ness"][idx, j]))
        hall = float(np.nanmean(t["hall"][idx, j]))
        conf = t["conf"][idx, j][num_mask]
        cor = correct[num_mask]
        ece = ece_of(conf, cor)
        out[j] = [acc, ness, 1.0 - hall, 1.0 - ece]
    return out


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0 or np.isnan(y).any():
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def weight_grid():
    return np.array([[w[k] for k in KEYS] for w in simplex_grid(0.05)])


def sweep(c4, c5, noise, W, r_thr=-0.5):
    """Shares of weightings that track noise, rank the conditions correctly, and do both."""
    s4 = W @ c4.T                       # (weights, levels)
    xs = noise - noise.mean()
    ys = s4 - s4.mean(axis=1, keepdims=True)
    denom = np.sqrt((xs ** 2).sum() * (ys ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ys @ xs) / denom
    tracks = np.nan_to_num(r, nan=0.0) <= r_thr
    s5 = W @ c5.T                       # columns: closed_book, retrieval, oracle
    ranks = (s5[:, 2] > s5[:, 1]) & (s5[:, 1] > s5[:, 0])
    return tracks, ranks, r


def pct(a, lo=2.5, hi=97.5):
    a = np.asarray([x for x in a if not (isinstance(x, float) and math.isnan(x))])
    if len(a) == 0:
        return [float("nan")] * 2
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


def analyse(cfg, boot, boot_sweep, rng):
    n4, n5 = len(cfg["ids"]), len(cfg["ids5"])
    levels = np.array(cfg["levels"])
    all4, all5 = np.arange(n4), np.arange(n5)
    W = weight_grid()

    def point(idx4, idx5, with_sweep=True):
        c4 = components(cfg["t4"], idx4)
        c5 = components(cfg["t5"], idx5)
        composite = c4 @ DEFAULT_W
        res = {
            "r_noise_composite": corr(levels, composite),
            "r_noise_component": {k: corr(levels, c4[:, i]) for i, k in enumerate(KEYS)},
            "composite_by_level": composite.tolist(),
            "accuracy_by_level": c4[:, 0].tolist(),
            "ness_by_level": c4[:, 1].tolist(),
            "halluc_rate_by_level": (1.0 - c4[:, 2]).tolist(),
            "ece_by_level": (1.0 - c4[:, 3]).tolist(),
            "exp5": {c: {"accuracy": float(c5[j, 0]), "ness": float(c5[j, 1]),
                         "halluc_rate": float(1.0 - c5[j, 2]), "composite": float(c5[j] @ DEFAULT_W)}
                     for j, c in enumerate(["closed_book", "retrieval", "oracle"])},
            "oracle_minus_retrieval_accuracy": float(c5[2, 0] - c5[1, 0]),
            "oracle_minus_retrieval_composite": float(c5[2] @ DEFAULT_W - c5[1] @ DEFAULT_W),
        }
        if with_sweep:
            tr, rk, _ = sweep(c4, c5, levels, W)
            res["sweep"] = {"tracks": float(tr.mean()), "ranks": float(rk.mean()),
                            "both": float((tr & rk).mean())}
        return res

    base = point(all4, all5)

    # item-level, paired noise effects: highest level minus level 0, per item
    t4 = cfg["t4"]
    first, last = 0, len(levels) - 1
    effects = {}
    for name, arr in (("accuracy", t4["correct"]), ("evidence_support", t4["ness"]),
                      ("hallucination_flag", t4["hall"]), ("support_vs_clean_evidence", t4["vs_clean"]),
                      ("nli_frac_entailed", cfg["nli4"])):
        if arr is None or np.isnan(arr).all():
            continue
        d = arr[:, last] - arr[:, first]
        d = d[~np.isnan(d)]
        if len(d) == 0:
            continue
        bs = [float(rng.choice(d, len(d)).mean()) for _ in range(boot)]
        effects[name] = {"mean_change": float(d.mean()), "ci95": pct(bs), "n_items": int(len(d))}

    # bootstrap over items for the headline statistics
    keys = ["r_noise_composite", "oracle_minus_retrieval_accuracy", "oracle_minus_retrieval_composite"]
    bs = {k: [] for k in keys}
    bs["r_halluc"] = []
    bs_sweep = {"tracks": [], "ranks": [], "both": []}
    for b in range(boot):
        i4 = rng.integers(0, n4, n4)
        i5 = rng.integers(0, n5, n5)
        c4 = components(t4 if False else cfg["t4"], i4)
        c5 = components(cfg["t5"], i5)
        comp = c4 @ DEFAULT_W
        bs["r_noise_composite"].append(corr(levels, comp))
        bs["r_halluc"].append(corr(levels, 1.0 - c4[:, 2]))
        bs["oracle_minus_retrieval_accuracy"].append(float(c5[2, 0] - c5[1, 0]))
        bs["oracle_minus_retrieval_composite"].append(float(c5[2] @ DEFAULT_W - c5[1] @ DEFAULT_W))
        if b < boot_sweep:
            tr, rk, _ = sweep(c4, c5, levels, W)
            bs_sweep["tracks"].append(float(tr.mean()))
            bs_sweep["ranks"].append(float(rk.mean()))
            bs_sweep["both"].append(float((tr & rk).mean()))
    base["ci95"] = {k: pct(v) for k, v in bs.items()}
    base["sweep_ci95"] = {k: pct(v) for k, v in bs_sweep.items()}
    base["r_noise_hallucination_rate"] = corr(levels, np.array(base["halluc_rate_by_level"]))
    base["p_oracle_not_better_composite"] = float(np.mean([x <= 0 for x in bs["oracle_minus_retrieval_composite"]]))
    base["item_level_noise_effects"] = effects
    base["n_items"] = n4
    base["n_items_exp5"] = n5
    return base


def held_out(dev, test):
    """Weightings that pass both tests on dev, evaluated on the test run of the same model."""
    W = weight_grid()
    out = {}
    for name, cfg in (("dev", dev), ("test", test)):
        c4 = components(cfg["t4"], np.arange(len(cfg["ids"])))
        c5 = components(cfg["t5"], np.arange(len(cfg["ids5"])))
        tr, rk, r = sweep(c4, c5, np.array(cfg["levels"]), W)
        out[name] = (tr, rk, r)
    tr_d, rk_d, r_d = out["dev"]
    tr_t, rk_t, r_t = out["test"]
    sel = tr_d & rk_d
    res = {"n_pass_both_on_dev": int(sel.sum())}
    if sel.any():
        res["of_those_pass_both_on_test"] = float((tr_t & rk_t)[sel].mean())
        res["of_those_track_noise_on_test"] = float(tr_t[sel].mean())
        res["of_those_rank_correctly_on_test"] = float(rk_t[sel].mean())
    else:
        res["note"] = "no weighting passes both tests on dev, so there is nothing to validate"
    # the single weighting chosen by dev among those ranking correctly, most negative noise r
    cand = np.where(rk_d)[0]
    if len(cand):
        k = cand[np.nanargmin(np.nan_to_num(r_d[cand], nan=9.0))]
        res["dev_choice_among_correct_rankings"] = {
            "weights": dict(zip(KEYS, W[k].tolist())), "dev_noise_r": float(r_d[k]),
            "test_noise_r": float(r_t[k]), "test_ranks_correctly": bool(rk_t[k]),
        }
    res["share_pass_both_dev"] = float((tr_d & rk_d).mean())
    res["share_pass_both_test"] = float((tr_t & rk_t).mean())
    return res


def parse_name(name):
    m = re.fullmatch(r"(?P<model>.+?)_(?P<split>dev|test)(?:_s(?P<seed>\d+))?", name)
    return (m["model"], m["split"], m["seed"]) if m else (None, None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/scale"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-sweep", type=int, default=200)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)

    cfgs, results = {}, {}
    for d in sorted(p for p in args.root.iterdir() if p.is_dir()):
        model, split, seed = parse_name(d.name)
        if model is None:
            continue
        cfg = load_config(d, split)
        if cfg is None:
            print(f"skipping {d.name}: per-example files for Experiments 4 and 5 not found")
            continue
        cfgs[d.name] = (model, split, seed, cfg)
        results[d.name] = analyse(cfg, args.boot, args.boot_sweep, rng)
        results[d.name].update({"model": model, "split": split, "seed": seed})
        r = results[d.name]
        print(f"{d.name}: n={r['n_items']}  r(noise,composite)={r['r_noise_composite']:+.2f} "
              f"{r['ci95']['r_noise_composite']}  both={r['sweep']['both']:.3f}")

    validation = {}
    for model in sorted({m for m, _, s, _ in cfgs.values()}):
        dev = next((c for n, (m, sp, s, c) in cfgs.items() if m == model and sp == "dev" and s is None), None)
        test = next((c for n, (m, sp, s, c) in cfgs.items() if m == model and sp == "test" and s is None), None)
        if dev and test:
            validation[model] = held_out(dev, test)

    seeds = {}
    for model in sorted({r["model"] for r in results.values()}):
        runs = [r for r in results.values() if r["model"] == model and r["split"] == "dev"]
        if len(runs) >= 3:
            def spread(f):
                v = [f(r) for r in runs if not math.isnan(f(r))]
                return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"),
                        "values": [float(x) for x in v]}
            seeds[model] = {
                "n_runs": len(runs),
                "r_noise_composite": spread(lambda r: r["r_noise_composite"]),
                "r_noise_hallucination_rate": spread(lambda r: r["r_noise_hallucination_rate"]),
                "accuracy_retrieval": spread(lambda r: r["exp5"]["retrieval"]["accuracy"]),
                "share_pass_both": spread(lambda r: r["sweep"]["both"]),
            }

    out = {"bootstrap_resamples": args.boot, "runs": results, "held_out_weight_validation": validation,
           "seed_spread": seeds}
    path = args.out or (args.root / "summary.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
