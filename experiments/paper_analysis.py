"""
Analyses for the paper (results/paper/): which gold-free measures notice corrupted evidence, how
they compare with baselines as error detectors, and what happens under real retrieval errors.

Three questions, each answered per run directory under results/scale/ and pooled per dataset:

1. Corruption detection. In the corrupted-evidence runs (Experiment 4, --injection-mode corrupt),
   can a measure tell a response produced from fully corrupted evidence from one produced from
   clean evidence? Reported as AUROC (0.5 = chance) with a bootstrap interval over items, for
   gold-free measures (numeric-overlap support, entailment support, lexical groundedness, the
   hallucination flag, self-reported confidence) and for two measures that need more than the
   response and its evidence: accuracy (needs the gold answer) and support measured against the
   clean evidence (needs the uncorrupted evidence).

2. Error detection. In the clean condition, how well does each gold-free measure separate correct
   from incorrect answers? The same measures, plus a combined gold-free score.

3. Real retrieval errors. In Experiment 5's retrieval condition the retriever sometimes misses the
   evidence the answer needs. Items are split by whether the retrieved evidence contains every
   number of the dataset's gold evidence (FinQA: the annotated gold facts; TAT-QA: the operands of
   the annotated derivation); the script compares accuracy and each measure between
   the two groups and reports how well each measure detects a retrieval miss.

"Lexical groundedness" is the baseline standing in for overlap-style faithfulness scoring that
needs no model: the share of the response's reasoning words (alphabetic words of four or more
letters) that occur in the evidence.

Usage:
    python -m experiments.paper_analysis [--root results/scale] [--out results/paper] [--boot 2000]
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

csv.field_size_limit(sys.maxsize)
RNG_SEED = 20261023

# run directory -> (dataset, model label); anything else is skipped by the paper tables
RUNS = {
    "qwen3bcorrupt_dev": ("FinQA", "Qwen2.5-3B fp32"),
    "qwen3bcorrupt_test": ("FinQA", "Qwen2.5-3B fp32"),
    "tat3bcorrupt_dev": ("TAT-QA", "Qwen2.5-3B fp32"),
    "tat3bcorrupt_test": ("TAT-QA", "Qwen2.5-3B fp32"),
    "q7bcorrupt_test": ("FinQA", "Qwen2.5-7B Q4"),
    "tat7bcorrupt_test": ("TAT-QA", "Qwen2.5-7B Q4"),
    "qwen3b_dev": ("FinQA", "Qwen2.5-3B fp32"),
    "qwen3b_test": ("FinQA", "Qwen2.5-3B fp32"),
    "qwen1p5b_dev": ("FinQA", "Qwen2.5-1.5B fp32"),
    "qwen1p5b_test": ("FinQA", "Qwen2.5-1.5B fp32"),
    "smollm2_dev": ("FinQA", "SmolLM2-1.7B fp32"),
    "tat3b_dev": ("TAT-QA", "Qwen2.5-3B fp32"),
    "tat3b_test": ("TAT-QA", "Qwen2.5-3B fp32"),
    "q7b_test": ("FinQA", "Qwen2.5-7B Q4"),
    "llama8bcorrupt_test": ("FinQA", "Llama-3.1-8B Q4"),
    "tat7b_test": ("TAT-QA", "Qwen2.5-7B Q4"),
}

TAT_CONSTANTS = {1.0, 2.0, 100.0, 1000.0, 1000000.0}
_WORD = re.compile(r"[A-Za-z]{4,}")
_NUM = re.compile(r"\d[\d,]*\.?\d*")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reasoning_part(response):
    text = re.split(r"final\s*answer\s*[:\-]", response or "", flags=re.I)[0]
    return re.sub(r"^\s*reasoning\s*[:\-]\s*", "", text, flags=re.I)


def lexical_groundedness(response, evidence):
    words = {w.lower() for w in _WORD.findall(reasoning_part(response))}
    if not words:
        return float("nan")
    ev = {w.lower() for w in _WORD.findall(evidence or "")}
    return len(words & ev) / len(words)


def numbers_of(text):
    out = set()
    for m in _NUM.findall(text or ""):
        m = m.replace(",", "").rstrip(".")
        if m:
            try:
                out.add(round(float(m), 6))
            except ValueError:
                pass
    return out


def auroc(score, label):
    """P(score of a label-1 item > score of a label-0 item), ties half."""
    score, label = np.asarray(score, float), np.asarray(label, float)
    ok = ~np.isnan(score) & ~np.isnan(label)
    pos, neg = score[ok & (label == 1)], score[ok & (label == 0)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def boot_auroc(score, label, groups, boot, rng):
    """AUROC with a bootstrap interval; resamples whole groups (items) so the paired rows of one
    item stay together."""
    score, label, groups = np.asarray(score, float), np.asarray(label, float), np.asarray(groups)
    point = auroc(score, label)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([index[uniq[p]] for p in pick])
        vals.append(auroc(score[idx], label[idx]))
    vals = [v for v in vals if not math.isnan(v)]
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2
    return {"auroc": point, "ci95": ci}


# --------------------------------------------------------------------------- loading

def load_nli(path, keycols):
    if not path.exists():
        return {}
    return {tuple(r[k] for k in keycols): num(r["nli_frac_entailed"]) for r in read_csv(path)}


def load_exp4(d, split):
    p = d / f"experiment4_results_{split}_per_example.csv"
    if not p.exists():
        return None
    nli = load_nli(d / f"experiment4_nli_{split}.csv", ["noise_level", "id"])
    rows = []
    for r in read_csv(p):
        rows.append({
            "id": r["id"], "level": float(r["noise_level"]),
            "correct": num(r["correct"]), "numeric": r.get("answer_type") == "numeric",
            "overlap": num(r["evidence_support"]), "vs_clean": num(r.get("evidence_support_vs_clean_strict")),
            "no_halluc": 1.0 - num(r["hallucination"]), "conf": num(r["self_reported_confidence"]),
            "lexical": lexical_groundedness(r.get("response"), r.get("evidence_text")),
            "nli": nli.get((r["noise_level"], r["id"]), float("nan")),
        })
    return rows


def load_exp5(d, split):
    p = d / f"experiment5_results_{split}.csv"
    if not p.exists():
        return None
    nli = load_nli(d / f"experiment5_nli_{split}.csv", ["condition", "id"])
    rows = []
    for r in read_csv(p):
        rows.append({
            "id": r["id"], "condition": r["condition"],
            "correct": num(r["correct"]), "numeric": r.get("answer_type") == "numeric",
            "overlap": num(r["evidence_support"]), "no_halluc": 1.0 - num(r["hallucination"]),
            "conf": num(r["self_reported_confidence"]),
            "lexical": lexical_groundedness(r.get("response"), r.get("evidence_text")),
            "nli": nli.get((r["condition"], r["id"]), float("nan")),
            "evidence_text": r.get("evidence_text", ""),
        })
    return rows


def gold_numbers(dataset, split):
    """id -> set of numbers in the dataset's gold evidence."""
    import os

    os.environ["NRF_DATASET"] = "tatqa" if dataset == "TAT-QA" else "finqa"
    from data_loader.load_finqa import load_dataset

    out = {}
    for item in load_dataset(split):
        if dataset == "TAT-QA":
            # TAT-QA has no cell-level evidence labels, so the gold evidence is the operands of its
            # annotated derivation (the scale constants and averaging divisors are not evidence)
            out[item["id"]] = numbers_of(item["qa"].get("program", "")) - TAT_CONSTANTS
        else:
            gi = item["qa"].get("gold_inds") or {}
            out[item["id"]] = numbers_of(" ".join(str(v) for v in gi.values()))
    return out


# --------------------------------------------------------------------------- analyses

GOLD_FREE = [("overlap", "numeric-overlap support"), ("nli", "entailment support"),
             ("lexical", "lexical groundedness"), ("no_halluc", "no-hallucination flag"),
             ("conf", "self-reported confidence")]
NEEDS_MORE = [("correct", "accuracy (needs gold answer)"),
              ("vs_clean", "support vs clean evidence (needs clean evidence)")]


def corruption_detection(rows, boot, rng):
    """AUROC for telling a fully corrupted response (label 0) from a clean one (label 1)."""
    levels = sorted({r["level"] for r in rows})
    lo, hi = levels[0], levels[-1]
    sel = [r for r in rows if r["level"] in (lo, hi)]
    label = [1.0 if r["level"] == lo else 0.0 for r in sel]
    groups = [r["id"] for r in sel]
    out = {}
    for key, name in GOLD_FREE + NEEDS_MORE:
        score = [r[key] for r in sel]
        if all(math.isnan(s) for s in score):
            continue
        out[key] = {"name": name, **boot_auroc(score, label, groups, boot, rng)}
    by_level = {}
    for key, _ in GOLD_FREE + NEEDS_MORE:
        by_level[key] = [float(np.nanmean([r[key] for r in rows if r["level"] == l])) for l in levels]
    return {"levels": levels, "n_items": len({r["id"] for r in rows}), "auroc": out, "mean_by_level": by_level}


def error_detection(rows, boot, rng, level_key="level", clean=0.0):
    sel = [r for r in rows if (r[level_key] == clean if level_key in r else r["condition"] == "retrieval")
           and r["numeric"] and not math.isnan(r["correct"])]
    label = [r["correct"] for r in sel]
    groups = [r["id"] for r in sel]
    out = {"n": len(sel), "accuracy": float(np.mean(label)) if label else float("nan"), "auroc": {}}
    for key, name in GOLD_FREE:
        score = [r[key] for r in sel]
        if all(math.isnan(s) for s in score):
            continue
        out["auroc"][key] = {"name": name, **boot_auroc(score, label, groups, boot, rng)}
    return out


def retrieval_errors(rows5, gold, boot, rng):
    """Split the retrieval-condition items by whether the evidence holds the gold numbers."""
    sel = []
    for r in rows5:
        if r["condition"] != "retrieval" or r["id"] not in gold or not gold[r["id"]]:
            continue
        have = numbers_of(r["evidence_text"])
        r = dict(r)
        r["recall"] = len(gold[r["id"]] & have) / len(gold[r["id"]])
        sel.append(r)
    if not sel:
        return None
    miss = [1.0 if r["recall"] < 1.0 else 0.0 for r in sel]
    groups = [r["id"] for r in sel]
    out = {"n": len(sel), "miss_rate": float(np.mean(miss)), "n_miss": int(sum(miss)), "groups": {}, "auroc": {}}
    for label, flag in (("hit", 0.0), ("miss", 1.0)):
        g = [r for r, m in zip(sel, miss) if m == flag]
        if not g:
            continue
        out["groups"][label] = {"n": len(g)}
        for key in ("correct", "overlap", "nli", "lexical", "no_halluc", "conf"):
            vals = [r[key] for r in g if not math.isnan(r[key])]
            out["groups"][label][key] = float(np.mean(vals)) if vals else float("nan")
    hit_label = [1.0 - m for m in miss]  # AUROC: does a higher score mean the evidence was complete?
    for key, name in GOLD_FREE + [("correct", "accuracy (needs gold answer)")]:
        score = [r[key] for r in sel]
        if all(math.isnan(s) for s in score):
            continue
        out["auroc"][key] = {"name": name, **boot_auroc(score, hit_label, groups, boot, rng)}
    return out


# --------------------------------------------------------------------------- report

def fmt_auc(v):
    return f"{v['auroc']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" if v and not math.isnan(v["auroc"]) else "n/a"


def pooled(rows_by_run):
    out = []
    for name, rows in rows_by_run.items():
        for r in rows:
            out.append({**r, "id": f"{name}:{r['id']}"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("results/scale"))
    ap.add_argument("--out", type=Path, default=Path("results/paper"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--skip-retrieval", action="store_true", help="skip the gold-evidence comparison (needs the dataset files)")
    args = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)
    args.out.mkdir(parents=True, exist_ok=True)

    result = {"corruption": {}, "error_detection": {}, "retrieval_errors": {}}
    corrupt_rows, clean_rows, exp5_rows = {}, {}, {}
    for name, (dataset, model) in RUNS.items():
        d = args.root / name
        if not d.is_dir():
            continue
        split = name.rsplit("_", 1)[-1]
        r4 = load_exp4(d, split)
        if r4 is None:
            continue
        mode = "corrupt" if "corrupt" in name else "evidence"
        if mode == "corrupt":
            corrupt_rows[name] = r4
            result["corruption"][name] = {"dataset": dataset, "model": model, **corruption_detection(r4, args.boot, rng)}
        clean_rows[name] = [r for r in r4 if r["level"] == 0.0]
        result["error_detection"][name] = {"dataset": dataset, "model": model,
                                           **error_detection(r4, args.boot, rng)}
        r5 = load_exp5(d, split)
        if r5 is not None:
            exp5_rows[name] = r5

    # pooled over runs of one dataset, so a table row has more items behind it
    for dataset in ("FinQA", "TAT-QA"):
        names = [n for n in corrupt_rows if RUNS[n][0] == dataset]
        if len(names) > 1:
            rows = pooled({n: corrupt_rows[n] for n in names})
            result["corruption"][f"pooled_{dataset}"] = {"dataset": dataset, "model": "pooled",
                                                         **corruption_detection(rows, args.boot, rng)}
        names = [n for n in clean_rows if RUNS[n][0] == dataset]
        if len(names) > 1:
            rows = pooled({n: clean_rows[n] for n in names})
            result["error_detection"][f"pooled_{dataset}"] = {"dataset": dataset, "model": "pooled",
                                                              **error_detection(rows, args.boot, rng)}

    if not args.skip_retrieval:
        gold_cache = {}
        pool = {"FinQA": {}, "TAT-QA": {}}
        for name, rows in exp5_rows.items():
            dataset, model = RUNS[name]
            split = name.rsplit("_", 1)[-1]
            if (dataset, split) not in gold_cache:
                try:
                    gold_cache[(dataset, split)] = gold_numbers(dataset, split)
                except FileNotFoundError as e:
                    print(f"skipping retrieval-error analysis for {name}: {e}")
                    gold_cache[(dataset, split)] = None
            gold = gold_cache[(dataset, split)]
            if gold is None:
                continue
            res = retrieval_errors(rows, gold, args.boot, rng)
            if res:
                result["retrieval_errors"][name] = {"dataset": dataset, "model": model, **res}
                pool[dataset][name] = [r for r in rows if r["condition"] == "retrieval"
                                       for r in [{**r, "id": f"{name}:{r['id']}"}]]
        for dataset, byrun in pool.items():
            if len(byrun) > 1:
                gold = {}
                for n in byrun:
                    split = n.rsplit("_", 1)[-1]
                    for k, v in gold_cache[(dataset, split)].items():
                        gold[f"{n}:{k}"] = v
                rows = [r for rs in byrun.values() for r in rs]
                res = retrieval_errors(rows, gold, args.boot, rng)
                if res:
                    result["retrieval_errors"][f"pooled_{dataset}"] = {"dataset": dataset, "model": "pooled", **res}

    (args.out / "analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    write_markdown(result, args.out / "ANALYSIS.md")
    print(f"wrote {args.out / 'analysis.json'} and ANALYSIS.md")


def write_markdown(res, path):
    L = ["# Paper analyses", "",
         "Generated by `python -m experiments.paper_analysis`. AUROC with a 95% bootstrap interval over items; 0.5 is chance.", ""]
    L += ["## 1. Can a measure tell fully corrupted evidence from clean evidence?", "",
          "AUROC for separating a response produced from 100%-corrupted evidence from one produced from the clean evidence of the same item.", ""]
    keys = [k for k, _ in GOLD_FREE + NEEDS_MORE]
    names = dict(GOLD_FREE + NEEDS_MORE)
    L.append("| Run | Dataset | Model | Items | " + " | ".join(names[k] for k in keys) + " |")
    L.append("|---|---|---|---|" + "---|" * len(keys))
    for run, v in res["corruption"].items():
        L.append(f"| {run} | {v['dataset']} | {v['model']} | {v['n_items']} | "
                 + " | ".join(fmt_auc(v["auroc"].get(k)) for k in keys) + " |")
    L += ["", "## 2. Can a gold-free measure tell a correct answer from an incorrect one? (clean evidence)", "",
          "| Run | Dataset | Model | Items | Accuracy | " + " | ".join(n for _, n in GOLD_FREE) + " |",
          "|---|---|---|---|---|" + "---|" * len(GOLD_FREE)]
    for run, v in res["error_detection"].items():
        L.append(f"| {run} | {v['dataset']} | {v['model']} | {v['n']} | {v['accuracy']:.2f} | "
                 + " | ".join(fmt_auc(v["auroc"].get(k)) for k, _ in GOLD_FREE) + " |")
    L += ["", "## 3. Real retrieval errors (retrieved evidence missing numbers of the gold evidence)", "",
          "Groups: `hit` = every number of the gold evidence is in the retrieved evidence, `miss` = at least one is absent. "
          "AUROC: can the measure tell hit from miss (higher score = evidence complete)?", "",
          "| Run | Dataset | Model | Items | Miss rate | Accuracy hit / miss | " + " | ".join(n for _, n in GOLD_FREE) + " | accuracy (needs gold) |",
          "|---|---|---|---|---|---|" + "---|" * (len(GOLD_FREE) + 1)]
    for run, v in res["retrieval_errors"].items():
        g = v["groups"]
        acc = " / ".join(f"{g[k]['correct']:.2f}" if k in g else "n/a" for k in ("hit", "miss"))
        L.append(f"| {run} | {v['dataset']} | {v['model']} | {v['n']} | {v['miss_rate']:.2f} ({v['n_miss']}) | {acc} | "
                 + " | ".join(fmt_auc(v["auroc"].get(k)) for k, _ in GOLD_FREE) + " | "
                 + fmt_auc(v["auroc"].get("correct")) + " |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
