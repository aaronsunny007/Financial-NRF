"""
Builds paper/figures/*.pdf and paper/generated/{tables,numbers}.tex from results/paper/analysis.json
(written by experiments.paper_analysis), so every number in the paper comes from a result file.

Usage:
    python -m experiments.paper_figures [--analysis results/paper/analysis.json] [--paper paper]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Okabe-Ito, colour-blind safe
COL = {"judge": "#882255", "overlap": "#0072B2", "nli": "#009E73", "lexical": "#CC79A7", "no_halluc": "#E69F00",
       "conf": "#56B4E9", "correct": "#D55E00", "vs_clean": "#000000"}
LABEL = {"judge": "LLM-judge faithfulness", "overlap": "numeric-overlap support", "nli": "entailment support", "lexical": "lexical groundedness",
         "no_halluc": "no-hallucination flag", "conf": "self-reported confidence",
         "correct": "accuracy (gold answer)", "vs_clean": "support vs clean evidence"}
GOLD_FREE = ["overlap", "nli", "lexical", "judge", "no_halluc", "conf"]
ALL = GOLD_FREE + ["correct", "vs_clean"]


def tex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def groups(corruption):
    """(dataset, model) -> list of run names, excluding pooled rows."""
    g = {}
    for name, v in corruption.items():
        if name.startswith("pooled_") or v.get("mode", "corrupt") != "corrupt":
            continue
        g.setdefault((v["dataset"], v["model"]), []).append(name)
    return g


def fig_corruption(corruption, out):
    g = groups(corruption)
    n = len(g)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.0 * nrows), sharey=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, ((dataset, model), names) in zip(axes.flat, g.items()):
        levels = np.array(corruption[names[0]]["levels"])
        for key in ["correct", "vs_clean"] + GOLD_FREE:
            ys = np.mean([corruption[nm]["mean_by_level"].get(key, [float("nan")] * len(levels)) for nm in names], axis=0)
            ok = ~np.isnan(ys)
            if not ok.any():
                continue
            ax.plot(levels[ok], ys[ok], marker="o", ms=3.5, lw=1.8 if key in ("correct", "vs_clean") else 1.2,
                    ls="-" if key in ("correct", "vs_clean") else "--", color=COL[key], label=LABEL[key])
        ax.set_title(f"{dataset}, {model}", fontsize=9)
        ax.set_xlabel("numbers in evidence replaced (%)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    for row in axes:
        row[0].set_ylabel("mean value")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_auroc(corruption, out):
    g = groups(corruption)
    n = len(g)
    ncols = min(3, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 2.9 * nrows), sharey=True, squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax, ((dataset, model), names) in zip(axes.flat, g.items()):
        for yi, key in enumerate(ALL[::-1]):
            vals = [corruption[nm]["auroc"][key] for nm in names if key in corruption[nm]["auroc"]]
            if not vals:
                continue
            for k, v in enumerate(vals):
                ax.errorbar(v["auroc"], yi + (k - (len(vals) - 1) / 2) * 0.18,
                            xerr=[[v["auroc"] - v["ci95"][0]], [v["ci95"][1] - v["auroc"]]],
                            fmt="o" if k == 0 else "s", ms=4, color=COL[key], capsize=2, lw=1.2)
        ax.axvline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_yticks(range(len(ALL)))
        ax.set_yticklabels([LABEL[k] for k in ALL[::-1]], fontsize=7.5)
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel("AUROC, corrupted vs clean")
        ax.set_title(f"{dataset}, {model}", fontsize=9)
        ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def f2(v):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"


def auc(v):
    if not v or math.isnan(v["auroc"]):
        return "--"
    return f"{v['auroc']:.2f} \\small[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]"


def table_corruption(corruption):
    rows = []
    for name, v in corruption.items():
        if v.get("mode", "corrupt") != "corrupt":
            continue
        run = f"{v['dataset']} / {v['model']}" if not name.startswith("pooled_") else f"{v['dataset']} pooled"
        split = "" if name.startswith("pooled_") else (" dev" if name.endswith("dev") else " test")
        rows.append((name.startswith("pooled_"), run + split, v["n_items"], [auc(v["auroc"].get(k)) for k in ALL]))
    head = " & ".join(["Run", "$n$"] + [tex_escape(LABEL[k]).replace(" (gold answer)", "$^\\dagger$")
                                         .replace("support vs clean evidence", "support vs clean$^\\ddagger$") for k in ALL])
    lines = [r"\begin{tabular}{@{}l r " + "c" * len(ALL) + r"@{}}", r"\toprule", head + r" \\", r"\midrule"]
    for pooled, run, n, cells in rows:
        if pooled:
            lines.append(r"\midrule")
        lines.append(" & ".join([tex_escape(run), str(n)] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_error(error):
    keys = GOLD_FREE
    lines = [r"\begin{tabular}{@{}l r r " + "c" * len(keys) + r"@{}}", r"\toprule",
             " & ".join(["Run", "$n$", "Acc."] + [tex_escape(LABEL[k]) for k in keys]) + r" \\", r"\midrule"]
    for name, v in error.items():
        if name.startswith("pooled_"):
            lines.append(r"\midrule")
            run = f"{v['dataset']} pooled"
        else:
            run = f"{v['dataset']} / {v['model']} " + ("dev" if name.endswith("dev") else "test")
        lines.append(" & ".join([tex_escape(run), str(v["n"]), f2(v["accuracy"])]
                                + [auc(v["auroc"].get(k)) for k in keys]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_retrieval(ret):
    keys = GOLD_FREE
    lines = [r"\begin{tabular}{@{}l r r r " + "c" * (len(keys) + 1) + r"@{}}", r"\toprule",
             " & ".join(["Run", "$n$", "Miss", "Acc. hit / miss"] + [tex_escape(LABEL[k]) for k in keys]
                        + ["accuracy$^\\dagger$"]) + r" \\", r"\midrule"]
    for name, v in ret.items():
        if name.startswith("pooled_"):
            lines.append(r"\midrule")
            run = f"{v['dataset']} pooled"
        else:
            run = f"{v['dataset']} / {v['model']} " + ("dev" if name.endswith("dev") else "test")
        g = v["groups"]
        acc = " / ".join(f2(g[k]["correct"]) if k in g else "--" for k in ("hit", "miss"))
        lines.append(" & ".join([tex_escape(run), str(v["n"]), f"{v['miss_rate']:.2f}", acc]
                                + [auc(v["auroc"].get(k)) for k in keys] + [auc(v["auroc"].get("correct"))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_realistic(corruption):
    """Response-side measures under realistic evidence errors (periods or rows swapped)."""
    keys = ALL
    lines = [r"\begin{tabular}{@{}l l r " + "c" * len(keys) + r"@{}}", r"\toprule",
             " & ".join(["Run", "Error", "$n$"] + [tex_escape(LABEL[k]).replace(" (gold answer)", "$^\\dagger$")
                                                   .replace("support vs clean evidence", "numbers vs clean$^\\ddagger$") for k in keys]) + r" \\", r"\midrule"]
    names = {"period_swap": "periods swapped", "row_swap": "rows swapped"}
    for name, v in corruption.items():
        if v.get("mode") not in names:
            continue
        lines.append(" & ".join([tex_escape(f"{v['dataset']} / {v['model']}"), names[v["mode"]], str(v["n_items"])]
                                + [auc(v["auroc"].get(k)) for k in keys]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_evidence_checks(checks, runs):
    """Evidence-side checks: one row per dataset and edit mode (the edit depends only on the item, not on the model)."""
    best = {}
    for name, v in checks.items():
        key = (runs[name][0], v["mode"])
        if key not in best or v["n_items"] > best[key][1]["n_items"]:
            best[key] = (name, v)
    modes = {"corrupt": "numbers scaled", "period_swap": "periods swapped", "row_swap": "rows swapped"}
    lines = [r"\begin{tabular}{@{}l l r r c c c@{}}", r"\toprule",
             r"Dataset & Error & $n$ & edited & redundancy & numbers vs trusted copy & cells vs trusted table$^\S$ \\", r"\midrule"]
    for (dataset, mode), (name, v) in sorted(best.items(), key=lambda kv: (kv[0][0], list(modes).index(kv[0][1]))):
        top = v["levels"][max(v["levels"], key=int)]
        lines.append(" & ".join([dataset, modes[mode], str(v["n_items"]), f"{100 * top['share_items_edited']:.0f}\\%",
                                 auc(top["redundancy"]), auc(top["number_source"]),
                                 auc(top["pair_source"]) + f" ({100 * top['pair_source']['coverage']:.0f}\\%)"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines), best


def table_paired(summary, res):
    """Paired item-level change from 0% to 100% corruption, from results/scale/summary.json."""
    cols = [("accuracy", "accuracy$^\\dagger$"), ("evidence_support", "numeric-overlap support"),
            ("nli_frac_entailed", "entailment support"), ("hallucination_flag", "hallucination flag"),
            ("support_vs_clean_evidence", "support vs clean$^\\ddagger$")]
    lines = [r"\begin{tabular}{@{}l r " + "c" * len(cols) + r"@{}}", r"\toprule",
             " & ".join(["Run", "$n$"] + [c[1] for c in cols]) + r" \\", r"\midrule"]
    for name, v in res["corruption"].items():
        if name.startswith("pooled_") or v.get("mode", "corrupt") != "corrupt":
            continue
        eff = summary["runs"].get(name, {}).get("item_level_noise_effects")
        if not eff:
            continue
        cells = []
        for key, _ in cols:
            e = eff.get(key)
            cells.append(f"{e['mean_change']:+.2f} \\small[{e['ci95'][0]:+.2f}, {e['ci95'][1]:+.2f}]" if e else "--")
        split = "dev" if name.endswith("dev") else "test"
        lines.append(" & ".join([tex_escape(f"{v['dataset']} / {v['model']} {split}"), str(v["n_items"])] + cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def numbers(res):
    """\\newcommand macros for the figures quoted in the prose."""
    out = []

    def mac(name, val):
        out.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    for ds, tag in (("FinQA", "Fin"), ("TAT-QA", "Tat")):
        runs = [v for k, v in res["corruption"].items() if v["dataset"] == ds and not k.startswith("pooled_")]
        if runs:
            for key, name in (("correct", "acc"), ("overlap", "ov"), ("nli", "nli"), ("vs_clean", "vsc")):
                first = np.mean([r["mean_by_level"][key][0] for r in runs])
                last = np.mean([r["mean_by_level"][key][-1] for r in runs])
                mac(f"{name}Zero{tag}", f"{first:.2f}")
                mac(f"{name}Full{tag}", f"{last:.2f}")
        c = res["corruption"].get(f"pooled_{ds}")
        if c:
            mac(f"n{tag}Corrupt", c["n_items"])
            for k in ALL:
                v = c["auroc"].get(k)
                if v:
                    mac(f"auc{tag}{k.replace('_', '')}", f"{v['auroc']:.2f}")
                    mac(f"auc{tag}{k.replace('_', '')}ci", f"[{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
        e = res["error_detection"].get(f"pooled_{ds}")
        if e:
            mac(f"n{tag}Err", e["n"])
            for k in GOLD_FREE:
                v = e["auroc"].get(k)
                if v:
                    mac(f"err{tag}{k.replace('_', '')}", f"{v['auroc']:.2f}")
        r = res["retrieval_errors"].get(f"pooled_{ds}")
        if r:
            mac(f"n{tag}Ret", r["n"])
            mac(f"miss{tag}", f"{100 * r['miss_rate']:.0f}")
            for k in GOLD_FREE + ["correct"]:
                v = r["auroc"].get(k)
                if v:
                    mac(f"ret{tag}{k.replace('_', '')}", f"{v['auroc']:.2f}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, default=Path("results/paper/analysis.json"))
    ap.add_argument("--paper", type=Path, default=Path("paper"))
    args = ap.parse_args()
    res = json.loads(args.analysis.read_text(encoding="utf-8"))
    (args.paper / "figures").mkdir(parents=True, exist_ok=True)
    (args.paper / "generated").mkdir(parents=True, exist_ok=True)
    fig_corruption(res["corruption"], args.paper / "figures" / "corruption.pdf")
    fig_auroc(res["corruption"], args.paper / "figures" / "auroc.pdf")
    gen = args.paper / "generated"
    (gen / "table_corruption.tex").write_text(table_corruption(res["corruption"]), encoding="utf-8")
    summary_path = Path("results/scale/summary.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        (gen / "table_paired.tex").write_text(table_paired(summary, res), encoding="utf-8")
    (gen / "table_realistic.tex").write_text(table_realistic(res["corruption"]), encoding="utf-8")
    ev_path = Path("results/paper/evidence_checks.json")
    if ev_path.exists():
        from experiments.paper_analysis import RUNS
        checks = json.loads(ev_path.read_text(encoding="utf-8"))
        tex, best = table_evidence_checks(checks, RUNS)
        (gen / "table_evidence.tex").write_text(tex, encoding="utf-8")
        macros = []
        for (dataset, mode), (name, v) in best.items():
            top = v["levels"][max(v["levels"], key=int)]
            tag = ("Fin" if dataset == "FinQA" else "Tat") + {"corrupt": "Scaled", "period_swap": "Period", "row_swap": "Row"}[mode]
            for key, short in (("redundancy", "Red"), ("number_source", "Num"), ("pair_source", "Pair")):
                macros.append(f"\\newcommand{{\\ev{short}{tag}}}{{{top[key]['auroc']:.2f}}}")
        (gen / "numbers_evidence.tex").write_text("\n".join(macros), encoding="utf-8")
    (gen / "table_error.tex").write_text(table_error(res["error_detection"]), encoding="utf-8")
    (gen / "table_retrieval.tex").write_text(table_retrieval(res["retrieval_errors"]), encoding="utf-8")
    (gen / "numbers.tex").write_text(numbers(res), encoding="utf-8")
    print("wrote figures and tables under", args.paper)


if __name__ == "__main__":
    main()
