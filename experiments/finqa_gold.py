"""
experiments/finqa_gold.py

Single source of truth for turning a FinQA item into the gold string used
for numerical scoring.

WHY THIS MODULE EXISTS: this logic originally lived inside experiment1.py
only. Experiments 2, 3 and 4 each scored against `str(item["qa"]["answer"])`
-- the human-readable DISPLAY string -- instead. That is a different (and
worse) gold value than experiment1 used, which meant the four experiments
were not comparable with each other, and experiments 2-4 did not benefit
from either of the percentage-scaling bug fixes made to experiment1. Having
one implementation imported by all four is the point.
"""


def resolve_gold_answer(item: dict):
    """Return (gold_string_for_scoring, answer_type) for a FinQA item.

    Prefers the dataset's precise EXECUTED answer (qa['exe_ans']) over the
    human-readable display string (qa['answer']), which FinQA frequently
    rounds for readability (e.g. display "14%" vs exe_ans 0.14464). This is
    the standard "execution accuracy" methodology used by FinQA itself and
    by the papers in the literature review -- NOT a loosened tolerance.

    PERCENTAGE SCALING. FinQA is inconsistent about whether exe_ans is a
    raw ratio (needing x100 to match a "%" display answer) or is already
    percent-scaled. Two distinct bugs were found and fixed here:

      1. exe_ans already scaled by the program itself, e.g.
         DVN/2007/page_58.pdf-2: program "divide(60, 243), multiply(#0,
         const_100)", exe_ans 24.69136, answer "24.69%". Multiplying again
         produced a gold of "2469.136%" and marked a correct model answer
         wrong.

      2. exe_ans computed from operands that are ALREADY percentages, e.g.
         ZBH/2003/page_40.pdf-1: program "subtract(51.2, 47.4)", exe_ans
         3.8, answer "3.8%". No multiply-by-const_100 step appears here, so
         a rule that keyed off the program string still mis-scaled it.

    Rather than pattern-matching the program string (which missed case 2),
    the scaling decision is made by comparing exe_ans against the numeric
    magnitude of FinQA's own human-written display answer and choosing
    whichever of {exe_ans, exe_ans * 100} is closer to it. The display
    string is a human annotation and is a far more reliable signal than
    program syntax.

    answer_type is 'numeric' for ordinary FinQA questions, 'non_numeric'
    for yes/no comparison questions (exe_ans is literally 'yes'/'no' --
    see programs using greater()/lesser()), which cannot be meaningfully
    scored by a NUMERICAL accuracy metric and are excluded from accuracy
    explicitly rather than silently counted either way.
    """
    qa = item.get("qa", {})
    display_answer = str(qa.get("answer", "")).strip()
    exe_ans = qa.get("exe_ans")

    def _display_numeric_magnitude(s: str):
        cleaned = s.replace("$", "").replace(",", "").replace("%", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    if isinstance(exe_ans, bool):
        return str(exe_ans), "non_numeric"

    if isinstance(exe_ans, (int, float)):
        if display_answer.endswith("%"):
            display_mag = _display_numeric_magnitude(display_answer)
            if display_mag is None:
                # Can't verify against the display string -- fall back to the
                # historical default rather than guessing further.
                return f"{exe_ans * 100}%", "numeric"
            raw_diff = abs(exe_ans - display_mag)
            scaled_diff = abs((exe_ans * 100) - display_mag)
            if raw_diff <= scaled_diff:
                return f"{exe_ans}%", "numeric"
            return f"{exe_ans * 100}%", "numeric"
        return str(exe_ans), "numeric"

    if isinstance(exe_ans, str) and exe_ans.strip():
        return exe_ans.strip(), "non_numeric"

    # No usable exe_ans -- fall back to the display string (may be empty;
    # nrf.numerical_accuracy will then correctly report parse_success=False
    # rather than silently guessing "correct").
    return display_answer, ("numeric" if display_answer else "unknown")


def _run_tests():
    """Regression tests for both scaling bugs. Run: python experiments/finqa_gold.py"""
    cases = [
        # (label, program, display answer, exe_ans, expected gold magnitude)
        ("DVN  program already x100",   "divide(60, 243), multiply(#0, const_100)", "24.69%", 24.69136, 24.69136),
        ("ZBH  operands already pct",   "subtract(51.2, 47.4)",                     "3.8%",   3.8,      3.8),
        ("C    ratio needs x100",       "subtract(193.5, const_100), divide(#0, const_100)", "93.5%", 0.935, 93.5),
        ("JPM  ratio needs x100",       "divide(136104, 1244659)",                  "10.94%", 0.10935,  10.935),
        ("PNC  ratio needs x100",       "divide(36197, 1189)",                      "3044%",  30.44323, 3044.323),
        ("ETR  ratio needs x100",       "divide(59.1, 98.0)",                       "60.3%",  0.60306,  60.306),
        ("MSI  ratio needs x100",       "divide(1451, 4134)",                       "35.1%",  0.35099,  35.099),
        ("CME  plain value",            "subtract(339235, 338240)",                 "995",    995.0,    995.0),
        ("AMT  plain value",            "add(2157503, 2418012)",                    "4575515", 4575515.0, 4575515.0),
    ]
    failures = []
    for label, program, answer, exe_ans, expected in cases:
        item = {"qa": {"program": program, "answer": answer, "exe_ans": exe_ans}}
        gold, kind = resolve_gold_answer(item)
        got = float(gold.rstrip("%"))
        ok = abs(got - expected) < 1e-6 and kind == "numeric"
        if not ok:
            failures.append(f"{label}: got {gold!r}, expected ~{expected}")
        print(f"  {'OK  ' if ok else 'FAIL'} {label:28s} -> {gold}")

    # non-numeric (yes/no) handling
    yn = {"qa": {"program": "greater(1, 2)", "answer": "no", "exe_ans": "no"}}
    gold, kind = resolve_gold_answer(yn)
    if kind != "non_numeric":
        failures.append(f"yes/no item classified as {kind}, expected non_numeric")
    print(f"  {'OK  ' if kind == 'non_numeric' else 'FAIL'} yes/no excluded from numeric  -> {gold!r} ({kind})")

    print()
    if failures:
        print(f"FAILED {len(failures)}:")
        for f in failures:
            print("   ", f)
    else:
        print("All FinQA gold-resolution tests passed.")


if __name__ == "__main__":
    _run_tests()
