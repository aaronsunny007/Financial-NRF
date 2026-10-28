"""
A model-based faithfulness judge, in the style of RAGAs' faithfulness score.

For each response a language model is shown the evidence and the response, lists the factual claims the
response makes, and marks each as SUPPORTED or UNSUPPORTED by the evidence. The score of a response is
the share of its claims marked SUPPORTED (NaN when the judge lists no claim). This is the model-based
counterpart of the heuristic measures in experiments/paper_analysis.py (numeric overlap, entailment with
a small NLI model, lexical overlap).

The judge reads the same evidence the generator saw, so it is a reference-free check of the same kind:
whether it notices corrupted evidence is exactly what the paper tests.

Only the first and last level of Experiment 4 are judged (clean and fully edited evidence), which is
all that the corruption-detection, error-detection and retrieval-miss analyses need.

Runs a 4-bit GGUF model through llama.cpp, so it needs no GPU:

    python -m experiments.llm_judge --dir results/scale/q7bcorrupt_test --split test \\
        --gguf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
        --shard 0/10 --out judge_0.jsonl

    python -m experiments.llm_judge --self-test       # prompt and parser checks, no model
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

csv.field_size_limit(sys.maxsize)
MAX_EVIDENCE_CHARS = 12000
MAX_TOKENS = 500

SYSTEM = "You are a careful fact-checker. You only use the evidence you are given."
USER = """Evidence:
{evidence}

Answer to check:
{answer}

Step 1. List each distinct factual claim the answer makes (a number together with what it refers to, or a statement taken from the evidence). Do not list the question or the confidence statement.
Step 2. For each claim, say SUPPORTED if the evidence states it or it follows from the evidence by simple arithmetic on figures that appear in the evidence, otherwise UNSUPPORTED.

Reply with JSON only, in this form:
{{"claims": [{{"claim": "...", "verdict": "SUPPORTED"}}, {{"claim": "...", "verdict": "UNSUPPORTED"}}]}}"""

_VERDICT = re.compile(r'"verdict"\s*:\s*"\s*(SUPPORTED|UNSUPPORTED)\s*"', re.IGNORECASE)
_CONFIDENCE = re.compile(r"confidence\s*[:\-]\s*(?:\d+(?:\.\d+)?|\.\d+)\s*%?", re.IGNORECASE)


def build_messages(evidence: str, response: str):
    answer = _CONFIDENCE.sub("", response or "").strip()
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(evidence=(evidence or "")[:MAX_EVIDENCE_CHARS], answer=answer)}]


def parse_verdicts(text: str):
    """(supported, total) counted from the judge's JSON, tolerant of broken or truncated JSON."""
    verdicts = [v.upper() for v in _VERDICT.findall(text or "")]
    return verdicts.count("SUPPORTED"), len(verdicts)


def faithfulness(supported: int, total: int) -> float:
    return supported / total if total else float("nan")


def load_rows(directory: Path, split: str):
    path = directory / f"experiment4_results_{split}_per_example.csv"
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    levels = sorted({float(r["noise_level"]) for r in rows})
    keep = {levels[0], levels[-1]}
    return [r for r in rows if float(r["noise_level"]) in keep]


def self_test():
    m = build_messages("- [table_row] a -- 2015: 5", "Reasoning: x = 5\nFinal Answer: 5\nConfidence: 0.9")
    assert "Confidence" not in m[1]["content"].split("Answer to check:")[1].split("Step 1")[0]
    assert parse_verdicts('{"claims": [{"claim": "a", "verdict": "SUPPORTED"}, {"claim": "b", "verdict": "unsupported"}]}') == (1, 2)
    assert parse_verdicts('{"claims": [{"claim": "a", "verdict": "SUPPORTED"}, {"claim": "b", "verd') == (1, 1)
    assert parse_verdicts("no json at all") == (0, 0)
    assert math.isnan(faithfulness(0, 0)) and faithfulness(1, 2) == 0.5
    print("llm_judge: self-test passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--gguf", help="<hf repo id>:<file name> of the judge model")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="judge only the first N rows of this shard (pilot runs)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return

    k, n = (int(x) for x in args.shard.split("/"))
    rows = [r for i, r in enumerate(load_rows(args.dir, args.split)) if i % n == k]
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.out.exists():
        done = {(json.loads(l)["level"], json.loads(l)["id"]) for l in open(args.out) if l.strip()}

    from llama_cpp import Llama

    repo_id, filename = args.gguf.split(":", 1)
    llm = Llama.from_pretrained(repo_id=repo_id, filename=filename, n_ctx=8192, n_threads=4, seed=0, verbose=False)
    print(f"judge {filename}: {len(rows)} rows in shard {args.shard}", flush=True)
    for j, r in enumerate(rows, 1):
        key = (float(r["noise_level"]), r["id"])
        if key in done:
            continue
        t0 = time.time()
        out = llm.create_chat_completion(messages=build_messages(r["evidence_text"], r["response"]),
                                         max_tokens=MAX_TOKENS, temperature=0.0)
        text = out["choices"][0]["message"]["content"]
        sup, tot = parse_verdicts(text)
        rec = {"level": key[0], "id": key[1], "claims": tot, "supported": sup,
               "faithfulness": faithfulness(sup, tot), "raw": text[:600], "seconds": round(time.time() - t0, 1)}
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{j}/{len(rows)}] claims={tot} supported={sup} ({rec['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
