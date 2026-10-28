"""
Checks on the retrieved evidence itself, as opposed to checks on the response.

The response-side measures in framework/nrf.py compare an answer with the evidence it was given, so they
cannot notice that the evidence is wrong. These functions look at the evidence. They differ in what they
need:

    redundancy_rate        needs nothing but the evidence. The share of its numbers that occur on at least
                           two different lines. Clean filings state the same figure in a table and in the
                           text; independently damaged copies stop matching.
    number_source_check    needs a trusted copy of the source. The share of the evidence's numbers that occur
                           in that copy. Catches changed figures, but not figures that were merely moved.
    arithmetic_explained_rate  needs nothing but the evidence. The share of its numbers that are the sum,
                           difference, ratio-as-percent or percent change of two numbers of one table row, to
                           the precision they are printed at. A negative result: about one number in ten is
                           "explained" in clean and in corrupted evidence alike.
    pair_source_check      needs a trusted *structured* copy. The share of the evidence's table cells, as
                           (row label, column header, value) triples, that match the copy. Catches moved figures
                           (a swapped period or row) as well.

All three return a number in [0, 1] where higher means the evidence looks more trustworthy, or NaN when the
evidence has nothing to check. Evidence lines use the format of rag_pipeline.rag.FinancialRAG.build_evidence_text:

    - [table_row] net revenue -- 2015: $5829 | 2014: $5735
    - [pre_text] revenue grew in 2015 .

Run the tests with:  python framework/evidence_consistency.py
"""

import math
import re
from collections import Counter
from typing import List, Set, Tuple

_NUM = re.compile(r"(?<![\w.])\$?\s?(\d+(?:,\d{3})*)(\.\d+)?(%?)")
NAN = float("nan")


def extract_numbers(line: str) -> List[float]:
    """Numeric values on one line, leaving out four-digit years and single digits (list markers, '1 )')."""
    out = []
    for m in _NUM.finditer(line):
        whole = m.group(1).replace(",", "")
        if not m.group(2) and len(whole) == 4 and 1900 <= int(whole) <= 2099:
            continue
        if not m.group(2) and int(whole) < 10:
            continue
        out.append(float(whole + (m.group(2) or "")))
    return out


def _lines(evidence: str) -> List[str]:
    return [l for l in (evidence or "").split("\n") if l.strip()]


def redundancy_rate(evidence: str) -> float:
    per_line = [extract_numbers(l) for l in _lines(evidence)]
    values = [v for p in per_line for v in p]
    if not values:
        return NAN
    on_lines: Counter = Counter()
    for p in per_line:
        for v in {round(x, 4) for x in p}:
            on_lines[v] += 1
    return sum(on_lines[round(v, 4)] >= 2 for v in values) / len(values)


def number_source_check(evidence: str, source: str) -> float:
    known = {round(v, 4) for l in _lines(source) for v in extract_numbers(l)}
    values = [v for l in _lines(evidence) for v in extract_numbers(l)]
    if not values:
        return NAN
    return sum(round(v, 4) in known for v in values) / len(values)


def table_cells(evidence: str) -> List[Tuple[str, str, float]]:
    """(row label, column header, value) triples of the table rows in the evidence."""
    cells = []
    for line in _lines(evidence):
        if not line.startswith("- [table_row] ") or " -- " not in line:
            continue
        label, payload = line[len("- [table_row] "):].split(" -- ", 1)
        for part in payload.split(" | "):
            if ": " not in part:
                continue
            header, value = part.split(": ", 1)
            nums = extract_numbers(value)
            if nums:
                cells.append((label.strip(), header.strip(), round(nums[0], 4)))
    return cells


def pair_source_check(evidence: str, source: str) -> float:
    known: Set[Tuple[str, str, float]] = set(table_cells(source))
    cells = table_cells(evidence)
    if not cells:
        return NAN
    return sum(c in known for c in cells) / len(cells)


def _numbers_with_decimals(line: str):
    out = []
    for m in _NUM.finditer(line):
        whole = m.group(1).replace(",", "")
        if not m.group(2) and len(whole) == 4 and 1900 <= int(whole) <= 2099:
            continue
        if not m.group(2) and int(whole) < 10:
            continue
        out.append((float(whole + (m.group(2) or "")), len(m.group(2)) - 1 if m.group(2) else 0))
    return out


def arithmetic_explained_rate(evidence: str) -> float:
    """Share of the evidence's numbers explained by an arithmetic relation of two numbers of one table row."""
    import itertools

    lines = _lines(evidence)
    per_line = [_numbers_with_decimals(l) for l in lines]
    rows = [[v for v, _ in per_line[i]] for i, l in enumerate(lines) if l.startswith("- [table_row]")]
    total = explained = 0
    for numbers in per_line:
        for c, decimals in numbers:
            total += 1
            tol = 0.5 * 10 ** (-decimals) + 1e-9
            found = False
            for vals in rows:
                for a, b in itertools.permutations(vals, 2):
                    if a == c or b == c:
                        continue
                    if abs(a + b - c) <= tol or abs(a - b - c) <= tol:
                        found = True
                    elif b != 0 and (abs((a - b) / b * 100 - c) <= tol or abs(a / b * 100 - c) <= tol):
                        found = True
                    if found:
                        break
                if found:
                    break
            explained += found
    return explained / total if total else NAN


def _run_tests() -> None:
    clean = ("- [table_row] net revenue -- 2015: $5829 | 2014: $5735\n"
             "- [pre_text] net revenue was $ 5829 million in 2015 and $ 5735 million in 2014 .")
    scaled = ("- [table_row] net revenue -- 2015: $2914.5 | 2014: $9176\n"
              "- [pre_text] net revenue was $ 7286.25 million in 2015 and $ 4588 million in 2014 .")
    swapped = ("- [table_row] net revenue -- 2015: $5735 | 2014: $5829\n"
               "- [pre_text] net revenue was $ 5829 million in 2015 and $ 5735 million in 2014 .")
    assert redundancy_rate(clean) == 1.0, redundancy_rate(clean)
    assert redundancy_rate(scaled) == 0.0
    assert number_source_check(clean, clean) == 1.0
    assert number_source_check(scaled, clean) == 0.0
    assert number_source_check(swapped, clean) == 1.0          # moved figures are still real figures
    assert pair_source_check(clean, clean) == 1.0
    assert pair_source_check(swapped, clean) == 0.0            # but the (label, period, value) pairs are wrong
    assert pair_source_check(scaled, clean) == 0.0
    assert math.isnan(pair_source_check("- [pre_text] nothing numeric here .", clean))
    assert math.isnan(redundancy_rate(""))
    table = "- [table_row] revenue -- 2015: 120.5 | 2014: 100.0 | change: 20.5"
    assert arithmetic_explained_rate(table) > 0.0          # 20.5 = 120.5 - 100.0
    assert arithmetic_explained_rate("- [table_row] revenue -- 2015: 120.5 | 2014: 100.0 | change: 77.7") < 1.0
    print("evidence_consistency: all tests passed")


if __name__ == "__main__":
    _run_tests()
