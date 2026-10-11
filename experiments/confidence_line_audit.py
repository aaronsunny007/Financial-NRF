"""
Audit of the "Confidence: 0.9" line in evidence-support scoring.

Until this was found, the evidence-support score (NESS) counted the number in
the model's own "Confidence: <x>" statement as one more number the retrieved
evidence had to support. It never could, so every response was scored as
carrying one extra unsupported number, and a borderline response could be
pushed under the hallucination threshold by it. framework/nrf.py now strips
the statement before scoring (exclude_confidence_from_support, default True).

The recorded results were produced with the old behaviour. This script
measures what the difference was, from the saved files alone (no GPU):

  Exact   Experiments 1 and 2. Their CSVs keep the response and the retrieved
          evidence, so every row is re-scored both ways. The old behaviour
          must reproduce the stored scores before the new one is trusted.

  Bounded Experiment 5. Its CSV keeps the response and each row's stored
          strict NESS but not the evidence, so the evidence cannot be re-read.
          The number of directly supported numbers is recovered from the
          stored score, and the corrected score is bracketed by the only two
          possibilities for the confidence value: it was unsupported (the
          normal case) or it happened to match an evidence number.

  Not possible  Experiment 4 keeps neither responses nor evidence per item;
          correcting it needs the model re-run.

Usage:
    python -m experiments.confidence_line_audit
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from framework.nrf import NumericalReliabilityFramework  # noqa: E402
from experiments.weight_sensitivity import (  # noqa: E402
    KEYS, RANK_ORDER, simplex_grid,
)

csv.field_size_limit(sys.maxsize)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_JSON = RESULTS_DIR / "confidence_line_audit_dev.json"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def load_rows(name):
    with open(RESULTS_DIR / f"{name}_results_dev.csv", newline="") as f:
        return list(csv.DictReader(f))


def ness_mode(name):
    with open(RESULTS_DIR / f"{name}_results_dev_summary.json") as f:
        return json.load(f)["ness_mode"]


# ---------------------------------------------------------------- exact part

def exact_audit(name, group_key=None):
    recorded_mode = ness_mode(name)
    thr = NumericalReliabilityFramework().hallucination_threshold
    old = NumericalReliabilityFramework(exclude_confidence_from_support=False)
    new = NumericalReliabilityFramework(exclude_confidence_from_support=True)

    scored = []
    for r in load_rows(name):
        res_old = old.numerical_evidence_support(r["response"], r["evidence_text"])
        res_new = new.numerical_evidence_support(r["response"], r["evidence_text"])
        # The old behaviour must reproduce both stored definitions before the
        # corrected one is trusted.
        for got, col in ((res_old.strict_score, "evidence_support_strict"),
                         (res_old.derived_score, "evidence_support_derived")):
            if not math.isclose(got, float(r[col]), abs_tol=1e-9):
                raise SystemExit(f"{name} {r['id']}: old behaviour gives {col}={got}, file says {r[col]}")
        stored_score = res_old.derived_score if recorded_mode == "derived" else res_old.strict_score
        if int(stored_score < thr) != int(float(r["hallucination"])):
            raise SystemExit(f"{name} {r['id']}: stored hallucination flag not reproduced")
        scored.append({
            "group": r[group_key] if group_key else "all",
            # True when the confidence value coincided with a number in the evidence.
            "conf_matched": res_old.directly_supported_count - res_new.directly_supported_count == 1,
            "strict": (res_old.strict_score, res_new.strict_score),
            "derived": (res_old.derived_score, res_new.derived_score),
        })

    def block(rows, mode):
        o = [x[mode][0] for x in rows]
        n = [x[mode][1] for x in rows]
        return {
            "n": len(rows),
            "mean_ness_recorded": mean(o),
            "mean_ness_corrected": mean(n),
            "hallucination_rate_recorded": mean(int(v < thr) for v in o),
            "hallucination_rate_corrected": mean(int(v < thr) for v in n),
            "flags_changed": sum(int(a < thr) != int(b < thr) for a, b in zip(o, n)),
        }

    out = {"recorded_mode": recorded_mode,
           "rows_where_confidence_value_matched_evidence": sum(x["conf_matched"] for x in scored)}
    for mode in ("strict", "derived"):
        out[mode] = {"overall": block(scored, mode)}
        if group_key:
            out[mode]["by_condition"] = {
                g: block([x for x in scored if x["group"] == g], mode)
                for g in sorted({x["group"] for x in scored})
            }
    return out


# -------------------------------------------------------------- bounded part

def exp5_audit():
    old = NumericalReliabilityFramework(exclude_confidence_from_support=False)
    new = NumericalReliabilityFramework(exclude_confidence_from_support=True)
    thr = new.hallucination_threshold

    by_cond = {}
    for r in load_rows("experiment5"):
        by_cond.setdefault(r["condition"], []).append(r)

    out, comps = {}, {}
    for cond, rows in by_cond.items():
        n = len(rows)
        numeric = [r for r in rows if r["answer_type"] == "numeric"]
        accuracy = mean(float(r["correct"]) for r in numeric)
        conf = [r for r in numeric if r["self_reported_confidence"] not in ("", "None")]
        ece = old.calculate_ece([float(r["self_reported_confidence"]) for r in conf],
                                [float(r["correct"]) for r in conf])

        rec_ness, rec_flags = [], []
        mid_ness, mid_flags = [], []   # confidence value was unsupported (normal case)
        low_ness, low_flags = [], []   # confidence value happened to match evidence
        stripped = 0
        for r in rows:
            s = float(r["evidence_support_strict"])
            rec_ness.append(float(r["evidence_support"]))
            rec_flags.append(int(float(r["hallucination"])))
            t_old = len(old.extract_numbers(r["response"]))
            t_new = len(new.extract_numbers(new.strip_confidence_statement(r["response"])))
            if t_new == t_old or t_old == 0:
                mid = low = s
            else:
                stripped += 1
                k = t_old - t_new
                direct = round(s * t_old)
                mid = direct / t_new if t_new else 0.0
                low = max(direct - k, 0) / t_new if t_new else 0.0
            mid_ness.append(mid); mid_flags.append(int(mid < thr))
            low_ness.append(low); low_flags.append(int(low < thr))

        def onrs(ness, flags):
            return new.calculate_onrs(accuracy=accuracy, faithfulness=mean(ness),
                                      hallucination_rate=mean(flags), ece=ece)

        # Sanity: the recorded values must reproduce the stored ONRS.
        recorded = onrs(rec_ness, rec_flags)
        out[cond] = {
            "n": n, "responses_with_confidence_statement": stripped,
            "accuracy": accuracy, "ece": ece,
            "recorded": {"ness": mean(rec_ness), "hallucination_rate": mean(rec_flags), "onrs": recorded},
            "corrected": {"ness": mean(mid_ness), "hallucination_rate": mean(mid_flags), "onrs": onrs(mid_ness, mid_flags)},
            "worst_case": {"ness": mean(low_ness), "hallucination_rate": mean(low_flags), "onrs": onrs(low_ness, low_flags)},
        }
        comps[cond] = {
            "accuracy": accuracy, "faithfulness": mean(mid_ness),
            "hallucination": 1 - mean(mid_flags), "calibration": 1 - ece,
        }

    stored = json.load(open(RESULTS_DIR / "experiment5_results_dev_summary.json"))["conditions"]
    for cond, v in out.items():
        if not math.isclose(v["recorded"]["onrs"], stored[cond]["onrs"], abs_tol=1e-9):
            raise SystemExit(f"Exp5 {cond}: rebuilt ONRS {v['recorded']['onrs']} != stored {stored[cond]['onrs']}")

    # Ranking, under the default weights and across the weight grid.
    def ranks(key):
        o = {c: out[c][key]["onrs"] for c in RANK_ORDER}
        return all(o[a] > o[b] for a, b in zip(RANK_ORDER, RANK_ORDER[1:]))

    def grid_fraction(components):
        ok = total = 0
        for w in simplex_grid(0.05):
            sc = {c: sum(w[k] * components[c][k] for k in KEYS) for c in RANK_ORDER}
            ok += all(sc[a] > sc[b] for a, b in zip(RANK_ORDER, RANK_ORDER[1:]))
            total += 1
        return ok / total

    recorded_comps = {c: {"accuracy": out[c]["accuracy"], "faithfulness": out[c]["recorded"]["ness"],
                          "hallucination": 1 - out[c]["recorded"]["hallucination_rate"],
                          "calibration": 1 - out[c]["ece"]} for c in out}
    comps["_recorded"] = recorded_comps
    return {
        "conditions": out,
        "_components": comps,
        "default_weights_rank_correctly": {"recorded": ranks("recorded"), "corrected": ranks("corrected"),
                                           "worst_case": ranks("worst_case")},
        "fraction_of_weightings_ranking_correctly": {"recorded": grid_fraction(recorded_comps),
                                                     "corrected": grid_fraction(comps)},
    }


# ------------------------------------------------------ Experiment 4 bounds

T_MAX = 50   # largest plausible count of numbers in one response (saved responses: median 12, max 40)


def candidates(strict):
    """Possible (directly_supported, total) pairs behind a stored strict score."""
    from fractions import Fraction
    fr = Fraction(strict).limit_denominator(T_MAX)
    k = 1
    while fr.denominator * k <= T_MAX:
        yield fr.numerator * k, fr.denominator * k
        k += 1


def corrected_options(strict):
    """Corrected strict scores a row could have had, assuming the confidence
    value was one of the unsupported numbers (checked on Experiments 1 and 2)."""
    opts = set()
    for d, t in candidates(strict):
        if d == t:                 # everything supported, including the confidence value
            opts.add(1.0)
        elif t <= 1:
            opts.add(0.0)
        else:
            opts.add(d / (t - 1))
    return opts or {strict}


def pearson(xs, ys):
    return statistics.correlation(xs, ys)


def exp4_audit(exp5_corrected_components):
    thr = NumericalReliabilityFramework().hallucination_threshold
    nrf = NumericalReliabilityFramework()

    # Validate the interval method where the truth is known: every Experiment 5
    # row's exact corrected score must lie inside the options derived from its
    # stored score alone.
    old = NumericalReliabilityFramework(exclude_confidence_from_support=False)
    new = NumericalReliabilityFramework()
    checked = 0
    for r in load_rows("experiment5"):
        t_old = len(old.extract_numbers(r["response"]))
        t_new = len(new.extract_numbers(new.strip_confidence_statement(r["response"])))
        if t_old == t_new or t_old == 0:
            continue
        s_old = float(r["evidence_support_strict"])
        direct = round(s_old * t_old)
        exact = direct / t_new if t_new else 0.0
        if not any(math.isclose(exact, o, abs_tol=1e-9) for o in corrected_options(s_old)):
            raise SystemExit(f"interval method missed Exp5 row {r['id']}: exact {exact}")
        checked += 1

    # Experiment 4's 0% level reproduces Experiment 5's retrieval condition
    # exactly (greedy decoding), so a noisy-level row with the same item id and
    # the same stored score is taken to be the same response, and its exact
    # corrected score is read from the Experiment 5 response. Only rows whose
    # score differs from the baseline response keep the interval treatment.
    baseline = {}
    for r in load_rows("experiment5"):
        if r["condition"] != "retrieval":
            continue
        t_old = len(old.extract_numbers(r["response"]))
        t_new = len(new.extract_numbers(new.strip_confidence_statement(r["response"])))
        s_old = float(r["evidence_support_strict"])
        direct = round(s_old * t_old)
        fixed = s_old if t_old == t_new or t_old == 0 else (direct / t_new if t_new else 0.0)
        baseline[r["id"]] = (s_old, fixed)

    with open(RESULTS_DIR / "experiment4_results_dev_per_example.csv", newline="") as f:
        per = list(csv.DictReader(f))
    summary = {float(r["noise_level"]): r for r in csv.DictReader(open(RESULTS_DIR / "experiment4_results_dev.csv"))}
    levels = sorted(summary)

    resolved = {}
    scen = {k: {"ness": [], "hall": []} for k in ("recorded", "most_flags_lowest_ness", "middle", "fewest_flags_highest_ness")}
    detail = []
    for lv in levels:
        rows = [r for r in per if float(r["noise_level"]) == lv]
        n = len(rows)
        lo_n, hi_n, mid_n, lo_f, hi_f, mid_f = [], [], [], [], [], []
        for r in rows:
            s_row = float(r["evidence_support_strict"])
            base = baseline.get(r["id"])
            if base and math.isclose(base[0], s_row, abs_tol=1e-9):
                opts = [base[1]]
                resolved[lv] = resolved.get(lv, 0) + 1
            else:
                opts = sorted(corrected_options(s_row))
            lo_n.append(opts[0]); hi_n.append(opts[-1]); mid_n.append(sum(opts) / len(opts))
            flags = [int(o < thr) for o in opts]
            lo_f.append(min(flags)); hi_f.append(max(flags)); mid_f.append(sum(flags) / len(flags))
        rec_n = mean(float(r["evidence_support_strict"]) for r in rows)
        rec_f = mean(int(float(r["hallucination"])) for r in rows)
        if not math.isclose(rec_n, float(summary[lv]["evidence_support_strict"]), abs_tol=1e-9):
            raise SystemExit(f"Exp4 level {lv}: rebuilt strict NESS {rec_n} != stored")
        scen["recorded"]["ness"].append(rec_n); scen["recorded"]["hall"].append(rec_f)
        scen["most_flags_lowest_ness"]["ness"].append(mean(lo_n)); scen["most_flags_lowest_ness"]["hall"].append(mean(hi_f))
        scen["middle"]["ness"].append(mean(mid_n)); scen["middle"]["hall"].append(mean(mid_f))
        scen["fewest_flags_highest_ness"]["ness"].append(mean(hi_n)); scen["fewest_flags_highest_ness"]["hall"].append(mean(lo_f))
        detail.append({"noise_level": lv, "rows_resolved_from_baseline_response": resolved.get(lv, 0),
                       "hallucination_rate_recorded": rec_f,
                       "hallucination_rate_corrected_range": [mean(lo_f), mean(hi_f)],
                       "ness_recorded": rec_n, "ness_corrected_range": [mean(lo_n), mean(hi_n)]})

    noise = levels
    acc = [float(summary[lv]["accuracy"]) for lv in levels]
    ece = [float(summary[lv]["ece"]) for lv in levels]
    out = {"interval_method_validated_on_experiment5_rows": checked, "by_noise_level": detail, "scenarios": {}}
    for name, v in scen.items():
        exp4_comp = [{"accuracy": a, "faithfulness": nn, "hallucination": 1 - h, "calibration": 1 - e}
                     for a, nn, h, e in zip(acc, v["ness"], v["hall"], ece)]
        comp_r = {k: pearson(noise, [c[k] for c in exp4_comp]) for k in KEYS}
        exp5 = exp5_corrected_components if name != "recorded" else exp5_corrected_components["_recorded"]
        exp5 = {c: exp5[c] for c in RANK_ORDER}
        n = tracks = ranking = both = 0
        for w in simplex_grid(0.05):
            r = pearson(noise, [sum(w[k] * c[k] for k in KEYS) for c in exp4_comp])
            sc = {c: sum(w[k] * exp5[c][k] for k in KEYS) for c in RANK_ORDER}
            ok = all(sc[a] > sc[b] for a, b in zip(RANK_ORDER, RANK_ORDER[1:]))
            n += 1; tracks += r <= -0.5; ranking += ok; both += (r <= -0.5 and ok)
        dflt = {"accuracy": 0.30, "faithfulness": 0.30, "hallucination": 0.20, "calibration": 0.20}
        out["scenarios"][name] = {
            "component_noise_correlation": comp_r,
            "default_weights_noise_r": pearson(noise, [sum(dflt[k] * c[k] for k in KEYS) for c in exp4_comp]),
            "fraction_tracking_noise": tracks / n,
            "fraction_ranking_correct": ranking / n,
            "fraction_passing_both": both / n,
        }
    return out


def main():
    result = {
        "what": "Effect of counting the 'Confidence: x' value as an evidence-support number",
        "experiment1_exact": exact_audit("experiment1"),
        "experiment2_exact": exact_audit("experiment2", group_key="condition"),
        "experiment5_bounded": exp5_audit(),
    }
    comps = result["experiment5_bounded"].pop("_components")
    result["experiment4_bounded"] = exp4_audit(comps)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    for name in ("experiment1_exact", "experiment2_exact"):
        for mode in ("strict", "derived"):
            b = result[name][mode]["overall"]
            tag = " [recorded mode]" if result[name]["recorded_mode"] == mode else ""
            print(f"{name} {mode}{tag} (n={b['n']}): NESS {b['mean_ness_recorded']:.4f} -> "
                  f"{b['mean_ness_corrected']:.4f}, hallucination {b['hallucination_rate_recorded']:.3f} -> "
                  f"{b['hallucination_rate_corrected']:.3f}, flags changed {b['flags_changed']}")
    e5 = result["experiment5_bounded"]
    for c, v in e5["conditions"].items():
        print(f"experiment5 {c}: NESS {v['recorded']['ness']:.4f} -> {v['corrected']['ness']:.4f}, "
              f"halluc {v['recorded']['hallucination_rate']:.2f} -> {v['corrected']['hallucination_rate']:.2f}, "
              f"ONRS {v['recorded']['onrs']:.4f} -> {v['corrected']['onrs']:.4f} "
              f"(worst case {v['worst_case']['onrs']:.4f})")
    print("Exp5 default weights rank oracle>retrieval>closed-book:", e5["default_weights_rank_correctly"])
    print("Exp5 share of weightings ranking correctly:", e5["fraction_of_weightings_ranking_correctly"])
    e4 = result["experiment4_bounded"]
    print(f"\nExperiment 4 interval method validated on {e4['interval_method_validated_on_experiment5_rows']} Experiment 5 rows")
    for d in e4["by_noise_level"]:
        lo, hi = d["hallucination_rate_corrected_range"]
        print(f"  noise {d['noise_level']:>4}: hallucination {d['hallucination_rate_recorded']:.2f} -> [{lo:.2f}, {hi:.2f}]"
              f"  ({d['rows_resolved_from_baseline_response']}/50 rows resolved exactly)")
    for name, v in e4["scenarios"].items():
        print(f"  {name:<27} halluc-noise r {v['component_noise_correlation']['hallucination']:+.3f}  "
              f"default-weights r {v['default_weights_noise_r']:+.3f}  tracks {v['fraction_tracking_noise']:.1%}  "
              f"ranks {v['fraction_ranking_correct']:.1%}  both {v['fraction_passing_both']:.1%}")
    print(f"\nWrote {OUT_JSON.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
