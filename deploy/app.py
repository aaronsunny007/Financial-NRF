"""
deploy/app.py

Gradio application for the Numerical Reliability Framework (NRF).

DEPLOYMENT REALITY -- read before hosting. This application has two modes,
separated deliberately because they have very different resource profiles:

  1. RESULTS EXPLORER. Loads the recorded experiment CSV/JSON files and
     renders every metric, per-example row and cross-experiment comparison.
     Pure pandas -- no model, no GPU, instant. This works on any free CPU
     tier and is the mode that should carry a demonstration.

  2. LIVE PIPELINE. Runs the real retrieval + generation + verification
     stack on a user-supplied question. This needs the LLM in memory. On a
     free CPU tier (2 vCPU) generation runs at roughly 1-3 tokens/second,
     so a single answer takes minutes. It is functional but slow, and the
     interface says so rather than appearing to hang. On a machine with a
     CUDA GPU it runs at normal speed.

The split exists so that a hosted deployment degrades honestly: the
evidence is always available even where the generator is impractical.

ENVIRONMENT VARIABLES
  NRF_ENABLE_LIVE   "1" to enable the live tab (default "0" -- explorer only)
  NRF_MODEL         override the generator (default: small model on CPU)
  NRF_RESULTS_DIR   path to the results directory (default: ../results)
"""

import os
import json
import glob
from pathlib import Path

import gradio as gr
import pandas as pd

RESULTS_DIR = Path(os.environ.get("NRF_RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
ENABLE_LIVE = os.environ.get("NRF_ENABLE_LIVE", "0") == "1"

# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------

def _load_json(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_csv(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def headline_table():
    """Cross-experiment headline metrics, assembled from whatever summary
    files are actually present. Missing experiments are reported as absent
    rather than silently omitted, so the display never implies a run
    happened that did not."""
    rows = []

    s1 = _load_json("experiment1_results_dev_summary.json")
    if s1:
        rows.append(["1. Baseline profiling", s1.get("n", "-"),
                     f"{s1.get('accuracy', float('nan')):.4f}",
                     f"{s1.get('evidence_support', float('nan')):.4f}",
                     f"{s1.get('hallucination_rate', float('nan')):.4f}",
                     f"{s1.get('ece_of_self_reported_confidence', float('nan')):.4f}",
                     f"{s1.get('onrs', float('nan')):.4f}"])
    else:
        rows.append(["1. Baseline profiling", "not run", "-", "-", "-", "-", "-"])

    s2 = _load_json("experiment2_results_dev_summary.json")
    if s2 and s2.get("conditions"):
        for cond, c in s2["conditions"].items():
            rows.append([f"2. Adversarial · {cond}", c.get("n", "-"),
                         f"{float(c.get('accuracy', 'nan')):.4f}",
                         f"{float(c.get('evidence_support', 'nan')):.4f}",
                         f"{float(c.get('hallucination_rate', 'nan')):.4f}", "-", "-"])
    else:
        rows.append(["2. Adversarial sensitivity", "not run", "-", "-", "-", "-", "-"])

    s3 = _load_json("experiment3_results_dev_summary.json")
    if s3:
        rows.append(["3. Calibration", s3.get("n_scored_numeric", s3.get("n", "-")),
                     f"{float(s3.get('accuracy', 'nan')):.4f}", "-", "-",
                     f"{float(s3.get('ece_of_self_reported_confidence', 'nan')):.4f}", "-"])
    else:
        rows.append(["3. Calibration", "not run", "-", "-", "-", "-", "-"])

    s5 = _load_json("experiment5_results_dev_summary.json")
    if s5 and s5.get("conditions"):
        for cond, c in s5["conditions"].items():
            rows.append([f"5. Comparison · {cond}", c.get("n_scored_numeric", "-"),
                         f"{float(c.get('accuracy', 'nan')):.4f}",
                         f"{float(c.get('evidence_support', 'nan')):.4f}",
                         f"{float(c.get('hallucination_rate', 'nan')):.4f}",
                         f"{float(c.get('ece', 'nan')):.4f}",
                         f"{float(c.get('onrs', 'nan')):.4f}" if c.get("onrs") is not None else "-"])
    else:
        rows.append(["5. Comparative baselines", "not run", "-", "-", "-", "-", "-"])

    return pd.DataFrame(rows, columns=["Experiment", "n", "Accuracy", "NESS",
                                       "Halluc.", "ECE", "ONRS"])


def noise_curve():
    df = _load_csv("experiment4_results_dev.csv")
    if df is None or "noise_level" not in df.columns:
        return None, "Experiment 4 results not found in the results directory."
    keep = [c for c in ["noise_level", "noise_reached_model_rate", "accuracy",
                        "evidence_support", "hallucination_rate", "ece", "onrs"]
            if c in df.columns]
    note = ("Read the 'reached' column first: where injected noise did not reach the model, "
            "a flat ONRS says nothing about the metric because there was no degradation to signal.")
    return df[keep], note


def comparison_table():
    s5 = _load_json("experiment5_results_dev_summary.json")
    if not s5:
        return None, ("Experiment 5 has not been run yet. Run:\n"
                      "    python -m experiments.experiment5 --split dev --n 50")
    rows = []
    for cond, c in s5.get("conditions", {}).items():
        rows.append([cond, c.get("n_scored_numeric", "-"),
                     f"{float(c.get('accuracy', 'nan')):.4f}",
                     f"[{float(c.get('accuracy_ci_low', 'nan')):.3f}, {float(c.get('accuracy_ci_high', 'nan')):.3f}]",
                     f"{float(c.get('evidence_support', 'nan')):.4f}",
                     f"{float(c.get('hallucination_rate', 'nan')):.4f}"])
    df = pd.DataFrame(rows, columns=["Condition", "n", "Accuracy", "95% CI", "NESS", "Halluc."])

    d = s5.get("error_decomposition") or {}
    lines = ["**Error decomposition** — all conditions share the same items, generator, "
             "prompt, scoring and verifier; only the evidence differs.\n"]
    if d.get("retrieval_contribution_over_closed_book") is not None:
        lines.append(f"- Gain from retrieval over closed-book: "
                     f"**{d['retrieval_contribution_over_closed_book']:+.4f}** — what the retrieval stage contributes.")
    if d.get("retrieval_attributable_error") is not None:
        lines.append(f"- Retrieval-attributable error: **{d['retrieval_attributable_error']:+.4f}** — "
                     f"accuracy recoverable by perfect retrieval; the ceiling on any retrieval improvement.")
    if d.get("reasoning_attributable_error") is not None:
        lines.append(f"- Reasoning-attributable error: **{d['reasoning_attributable_error']:.4f}** — "
                     f"error remaining with gold evidence; not addressable by retrieval at all.")
    return df, "\n".join(lines)


def per_example(exp_choice, only_wrong):
    mapping = {
        "Experiment 1 — baseline": "experiment1_results_dev.csv",
        "Experiment 2 — adversarial": "experiment2_results_dev.csv",
        "Experiment 3 — calibration": "experiment3_results_dev.csv",
        "Experiment 4 — noise (per example)": "experiment4_results_dev_per_example.csv",
        "Experiment 5 — comparison": "experiment5_results_dev.csv",
    }
    df = _load_csv(mapping.get(exp_choice, ""))
    if df is None:
        return pd.DataFrame({"status": [f"{exp_choice} results not found."]})
    drop = [c for c in ["response", "evidence_text", "unsupported_numbers", "derived_numbers"] if c in df.columns]
    view = df.drop(columns=drop)
    if only_wrong and "correct" in view.columns:
        view = view[view["correct"] == 0.0]
    return view.head(400)


def inspect_row(exp_choice, row_index):
    mapping = {
        "Experiment 1 — baseline": "experiment1_results_dev.csv",
        "Experiment 2 — adversarial": "experiment2_results_dev.csv",
        "Experiment 3 — calibration": "experiment3_results_dev.csv",
        "Experiment 5 — comparison": "experiment5_results_dev.csv",
    }
    df = _load_csv(mapping.get(exp_choice, ""))
    if df is None:
        return "Results file not found."
    try:
        r = df.iloc[int(row_index)]
    except Exception:
        return f"Row {row_index} out of range (file has {0 if df is None else len(df)} rows)."
    out = [f"### {r.get('id', '(no id)')}"]
    if "gold_answer" in r: out.append(f"**Gold:** {r['gold_answer']}")
    if "predicted_value" in r: out.append(f"**Predicted:** {r['predicted_value']}")
    if "correct" in r: out.append(f"**Correct:** {r['correct']}")
    if "verifier_reason" in r and pd.notna(r.get("verifier_reason")):
        out.append(f"**Verifier:** {r['verifier_reason']}")
    if "evidence_text" in r and pd.notna(r.get("evidence_text")):
        out.append("\n**Retrieved evidence**\n```\n" + str(r["evidence_text"])[:2500] + "\n```")
    if "response" in r and pd.notna(r.get("response")):
        out.append("\n**Model response**\n```\n" + str(r["response"])[:2500] + "\n```")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Live pipeline (optional)
# ---------------------------------------------------------------------------

_RAG = None

def _get_rag():
    global _RAG
    if _RAG is None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from rag_pipeline.rag import FinancialRAG
        _RAG = FinancialRAG()
    return _RAG


def run_live(question, evidence):
    if not ENABLE_LIVE:
        return ("Live mode is disabled in this deployment.\n\n"
                "It is off by default because generation on a free CPU tier takes "
                "minutes per question. Set NRF_ENABLE_LIVE=1 to turn it on, or run "
                "locally on a CUDA machine for normal speed.")
    if not question.strip():
        return "Enter a question."
    try:
        rag = _get_rag()
        item = {"id": "live", "qa": {"question": question},
                "pre_text": [l for l in evidence.split("\n") if l.strip()],
                "post_text": [], "table": []}
        result = rag.answer(item, evidence_override=evidence if evidence.strip() else None)
        v = result.get("verification") or {}
        return (f"**Response**\n```\n{result.get('response','')}\n```\n\n"
                f"**Verifier:** parseable={v.get('parseable')} verified={v.get('verified')}\n\n"
                f"{v.get('reason','')}")
    except Exception as e:
        return f"Live run failed: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

INTRO = """
# Numerical Reliability Framework for RAG-Based Financial QA

Evaluation framework decomposing reliability into **execution accuracy**, a
**Numerical Evidence Support Score (NESS)**, **rule-based hallucination detection**,
and **calibration of self-reported confidence**, aggregated into a proposed
**Overall Numerical Reliability Score (ONRS)**.

Evaluated on FinQA with a locally hosted 3B instruction-tuned generator under
4-bit quantisation. All figures below are loaded from recorded experiment runs.

*MSc project — Aaron Prakash Gandi, School of Computing, Ulster University.*
"""

with gr.Blocks(title="NRF — Financial RAG Reliability", theme=gr.themes.Soft()) as demo:
    gr.Markdown(INTRO)

    with gr.Tab("Headline results"):
        gr.Markdown("Cross-experiment summary. Experiments not yet run are marked as such "
                    "rather than omitted.")
        gr.DataFrame(value=headline_table, interactive=False, wrap=True)

    with gr.Tab("Comparative baselines"):
        gr.Markdown("### Controlled comparison on identical data\n"
                    "Closed-book, retrieval and oracle (gold-evidence) conditions run over the "
                    "same items with the same generator and scorer, so differences are "
                    "attributable to the evidence supplied rather than to methodology.")
        cmp_df = gr.DataFrame(interactive=False, wrap=True)
        cmp_md = gr.Markdown()
        demo.load(comparison_table, outputs=[cmp_df, cmp_md])

    with gr.Tab("Reliability under noise"):
        gr.Markdown("### ONRS response to graded context corruption")
        noise_df = gr.DataFrame(interactive=False, wrap=True)
        noise_md = gr.Markdown()
        demo.load(noise_curve, outputs=[noise_df, noise_md])

    with gr.Tab("Per-example inspection"):
        gr.Markdown("Every scored example, with the option to filter to failures only. "
                    "Use the row inspector to read the retrieved evidence and the model's "
                    "full response for any single item.")
        with gr.Row():
            exp_pick = gr.Dropdown(
                ["Experiment 1 — baseline", "Experiment 2 — adversarial",
                 "Experiment 3 — calibration", "Experiment 4 — noise (per example)",
                 "Experiment 5 — comparison"],
                value="Experiment 1 — baseline", label="Experiment")
            wrong_only = gr.Checkbox(label="Failures only", value=False)
        table = gr.DataFrame(interactive=False, wrap=True)
        exp_pick.change(per_example, [exp_pick, wrong_only], table)
        wrong_only.change(per_example, [exp_pick, wrong_only], table)
        demo.load(per_example, [exp_pick, wrong_only], table)

        gr.Markdown("---\n### Inspect a single row")
        with gr.Row():
            row_idx = gr.Number(value=0, precision=0, label="Row index")
            inspect_btn = gr.Button("Inspect", variant="primary")
        detail = gr.Markdown()
        inspect_btn.click(inspect_row, [exp_pick, row_idx], detail)

    with gr.Tab("Live pipeline"):
        status = ("**Enabled.** Generation on CPU is slow — expect minutes per question."
                  if ENABLE_LIVE else
                  "**Disabled in this deployment.** Free CPU hosting makes live generation "
                  "impractically slow (minutes per answer). The recorded results in the other "
                  "tabs are the evidence; enable this only where a GPU is available.")
        gr.Markdown(f"### Run the pipeline on your own question\n{status}")
        q_in = gr.Textbox(label="Question", placeholder="what was the percentage change in net revenue?")
        e_in = gr.Textbox(label="Evidence (one line per fact; leave blank to use retrieval)",
                          lines=6, placeholder="- net revenue was 5829 million in 2015\n- net revenue was 5735 million in 2014")
        run_btn = gr.Button("Run", variant="primary", interactive=ENABLE_LIVE)
        out = gr.Markdown()
        run_btn.click(run_live, [q_in, e_in], out)

    with gr.Tab("Method"):
        gr.Markdown("""
### Metric definitions

**Execution accuracy** — the final answer is compared against FinQA's precise executed
answer (`exe_ans`), not its rounded display string, following the benchmark's own
methodology. Tolerance is `max(0.01, 2% of gold)`. A percentage/non-percentage unit
mismatch is scored as incorrect even when magnitudes agree; this is a deliberate
conservative policy.

**NESS** — the proportion of numeric values in a response that are supported by the
retrieved evidence. Reported under two definitions. *Strict* requires literal
appearance in the evidence. *Revised* additionally credits values derivable from
evidence numbers the response itself cited, because a computed answer does not appear
verbatim in the source text and the strict definition therefore penalises the system
for performing the arithmetic the task requires. The revision was validated against
gaming: an unrestricted implementation credited invented numbers 86.5% of the time,
while the restricted version admits them 0.2% of the time over 400 adversarial trials.

**Hallucination detection** — a threshold on NESS. A rule-based baseline, not a
trained classifier.

**Calibration** — binned expected calibration error over *self-reported* confidence,
elicited in the prompt. This is not a probability derived from token log-probabilities
and must not be described as one. Missing values are excluded and counted, never
imputed.

**ONRS** — 0.30 accuracy + 0.30 evidence support + 0.20 (1 − hallucination rate) +
0.20 (1 − ECE). Proposed by this project, not established in the literature. The
experiments demonstrate a structural weakness in this aggregation: where self-reported
confidence is near-constant, ECE becomes an approximately deterministic function of
accuracy, double-counting it and masking signal present in the components.

**Gold-free verifier** — re-executes the model's own stated arithmetic in a restricted
AST evaluator and checks operand grounding, without ever consulting the gold answer.
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
