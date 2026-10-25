"""
TAT-QA (Zhu et al., 2021) arithmetic questions, converted to the FinQA item layout.

TAT-QA pairs a financial-report table with its surrounding paragraphs and asks questions whose
answers are spans, counts or the result of arithmetic. Only the arithmetic questions with a
numeric answer are kept, because the framework scores numerical answers. Each kept question
becomes one item with the same fields the FinQA loader returns, so the retrieval pipeline and
all five experiments run on it unchanged:

    id, pre_text (the paragraphs), table, post_text (empty),
    qa = {question, answer, exe_ans, program, gold_inds}

Splits (TAT-QA's own test labels are hidden, so its train and dev files are used):
    "dev"   arithmetic questions from tatqa_dataset_train.json
    "test"  arithmetic questions from tatqa_dataset_dev.json
Each is shuffled once with a fixed seed so that a prefix of n items spans many reports.

Answers are in the table's own units. A "percent" answer is already in percent (12.5 means 12.5%),
which is converted to the FinQA convention (exe_ans 0.125, display answer "12.5%").

gold_inds is the document-level oracle: the paragraphs TAT-QA marks as relevant plus, when the
answer comes from the table, the whole table. TAT-QA has no cell-level evidence annotation, so
this oracle is coarser than FinQA's.

Data files are fetched by the workflow (data_loader/tatqa_train.json, tatqa_dev.json).
"""

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
FILES = {"dev": "tatqa_train.json", "test": "tatqa_dev.json"}
SHUFFLE_SEED = 7
_WS = re.compile(r"\s+")


def _clean(cell):
    return _WS.sub(" ", str(cell)).strip()


def _split_header(rows):
    """Merge the leading header rows (those whose first cell is empty) into one header row."""
    k = 0
    while k < len(rows) - 1 and not _clean(rows[k][0]):
        k += 1
    k = max(k, 1)
    width = max(len(r) for r in rows)
    header = []
    for c in range(width):
        parts = []
        for r in rows[:k]:
            v = _clean(r[c]) if c < len(r) else ""
            if v and v not in parts:
                parts.append(v)
        header.append(" ".join(parts))
    if header and header[0]:
        header[0] = ""
    return header, rows[k:]


def _format_row(label, header, cells):
    pairs = [f"{h}: {v}" for h, v in zip(header[1:], cells[1:]) if h and v]
    return f"{label} -- {' | '.join(pairs)}" if pairs else label


def _table_rows(raw):
    rows = [[_clean(c) for c in r] for r in raw if any(_clean(c) for c in r)]
    if len(rows) < 2:
        return rows, []
    header, body = _split_header(rows)
    width = len(header)
    table = [header] + [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in body]
    return table, [(r[0], _format_row(r[0], header, r)) for r in table[1:] if r and r[0]]


def _convert(doc, q):
    table, formatted = _table_rows(doc["table"]["table"])
    paras = sorted(doc["paragraphs"], key=lambda p: p["order"])
    ans, scale = q["answer"], q.get("scale", "")
    if scale == "percent":
        display, exe = f"{ans}%", ans / 100.0
    else:
        display, exe = str(ans), ans
    gold = {}
    rel = {int(o) for o in q.get("rel_paragraphs", []) or []}
    for p in paras:
        if p["order"] in rel:
            gold[f"text_{p['order']}"] = p["text"]
    if q.get("answer_from") in ("table", "table-text"):
        for i, (_, text) in enumerate(formatted):
            gold[f"table_{i + 1}"] = text
    return {
        "id": q["uid"],
        "pre_text": [p["text"] for p in paras],
        "post_text": [],
        "table": table,
        "qa": {
            "question": q["question"],
            "answer": display,
            "exe_ans": exe,
            "program": q.get("derivation", ""),
            "gold_inds": gold,
            "scale": scale,
            "answer_from": q.get("answer_from", ""),
        },
    }


def load_tatqa(split="test"):
    path = DATA_DIR / FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"TAT-QA file {path} not found; fetch it from "
                                "https://github.com/NExTplusplus/TAT-QA/tree/master/dataset_raw")
    with open(path, encoding="utf-8") as f:
        docs = json.load(f)
    items = [_convert(d, q) for d in docs for q in d["questions"]
             if q["answer_type"] == "arithmetic" and isinstance(q["answer"], (int, float))
             and not isinstance(q["answer"], bool)]
    random.Random(SHUFFLE_SEED).shuffle(items)
    return items
