"""
framework/verifier.py

Post-hoc Arithmetic Verifier (implements the "Multi-layer Reasoning and
Cross-Verification" idea from the research design document's Stage 4,
and mirrors the compile-and-execute / verifier-guided execution pattern
used by DCRC and LedgerRAG in the literature review).

IMPORTANT SCOPE NOTE -- read before citing this in the dissertation:

This verifier NEVER looks at the gold answer or gold program. It only
checks two things that can be checked WITHOUT gold labels, which is
exactly what makes it usable in real deployment (where there is no gold
answer to compare against):

  1. Operand grounding: are the numbers the model says it used in its
     "Reasoning:" section actually present in the retrieved evidence
     text (within numeric tolerance)? This catches invented/hallucinated
     inputs -- the model using a number that doesn't come from the
     evidence at all.

  2. Execution consistency: if the Reasoning section states an
     arithmetic expression (e.g. "the calculation is:
     ((100690000 - 92710000) / 92710000) * 100." -- note: no "=result"
     in that sentence, the model states the number separately in
     "Final Answer: 10.04%"), does Python's own deterministic evaluation
     of that expression match the model's separately-stated Final
     Answer? This catches arithmetic slips even when the operands are
     correctly grounded (the model picked the right numbers but did the
     math wrong).

     REVISION NOTE: the first version of this function only matched an
     expression when its result was written inline in the SAME sentence
     ("expr = result"). Empirically, Qwen2.5-3B often writes the
     expression as a plain sentence in Reasoning and states the number
     only in the separate Final Answer line (see Experiment 1 dev-run:
     example SYY/2006/page_71.pdf-1) -- the old pattern silently failed
     to parse these and reported "unverifiable" instead of catching a
     real arithmetic error (8.61% recomputed vs 10.04% stated). Fixed by
     decoupling expression-finding from result-finding: find the last
     arithmetic-looking span in Reasoning (operators + >=2 numbers, no
     "=result" required), and compare its recomputation against
     nrf.extract_final_answer()'s value directly, wherever that appears
     in the response.

This verifier CANNOT and does not claim to verify that the model picked
the numbers that are actually semantically relevant to the question
(e.g. the correct year, the correct line item, the correct denominator).
Detecting that would require either gold labels (test-set leakage,
invalid at inference time) or a much larger semantic verification model,
both out of scope here. A verified=True result means "internally
consistent and grounded in evidence," NOT "correct" -- correctness is
still, and only, measured by comparing to the gold answer in
framework/nrf.py's numerical_accuracy(), completely separately from this
module. Do not conflate the two in the write-up.
"""

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import List, Optional

from framework.nrf import NumericalReliabilityFramework


# ---------------------------------------------------------------------------
# Safe arithmetic evaluation (no eval() on untrusted model output)
# ---------------------------------------------------------------------------

class UnsafeExpressionError(Exception):
    """Raised when the extracted expression contains anything beyond
    plain arithmetic (+, -, *, /, parentheses, unary +/-, numeric
    literals). Deliberately conservative: anything not on this list is
    rejected rather than guessed at."""


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise UnsafeExpressionError(f"Non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval_node(node.operand))
    raise UnsafeExpressionError(f"Disallowed expression node: {type(node).__name__}")


def safe_eval_arithmetic(expr: str) -> float:
    """Evaluate a plain arithmetic expression using Python's own
    arithmetic (never trusting the model's stated result). Raises
    UnsafeExpressionError / SyntaxError / ZeroDivisionError on anything
    that isn't clean +,-,*,/ arithmetic -- callers must treat that as
    'could not verify', never as 'verified correct'."""
    tree = ast.parse(expr, mode="eval")
    return _safe_eval_node(tree)


# ---------------------------------------------------------------------------
# Extracting a verifiable equation from the model's Reasoning section
# ---------------------------------------------------------------------------

_REASONING_SECTION_RE = re.compile(
    r"reasoning\s*[:\-]\s*(.*?)(?:final\s*answer\s*[:\-]|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# A span made only of digits, decimal points, commas, $, %, parentheses,
# arithmetic operators and whitespace. Deliberately does NOT require a
# trailing '=result' -- letters (ordinary prose) naturally bound each
# span, so "the calculation is: ((100690000 - 92710000) / 92710000) * 100."
# is captured whole even with no inline result, and a separate "= 9.86"
# elsewhere becomes its own (operator-less, filtered-out) span rather
# than being required to attach to the expression before it.
_EXPR_SPAN_RE = re.compile(r"[\d.,\$%()+\-*/\s]{3,}")


def _clean_expr(expr: str) -> str:
    return expr.replace("$", "").replace(",", "").replace("%", "").strip(" .\t\n")


def find_last_expression(reasoning_text: str) -> Optional[str]:
    """Find the LAST span in the reasoning text that looks like a real
    arithmetic expression: contains at least one operator AND at least
    two numeric tokens (so a bare restated value like "net revenue 94"
    or a lone "= 9.86" fragment doesn't qualify). The concluding
    calculation is usually the one most worth verifying, hence "last".
    Returns the cleaned expression string, or None if nothing qualifies."""
    best = None
    for match in _EXPR_SPAN_RE.finditer(reasoning_text):
        span = match.group(0)
        # A '.' serves as both a decimal point and a sentence terminator, so
        # a raw span can run across a sentence boundary and yield something
        # syntactically invalid. Confirmed case:
        #     "Total facilities = 56.0. 8.1 / 56.0 * 100 = 14.46"
        # extracted as "56.0. 8.1 / 56.0 * 100", which fails to parse, and
        # the response was reported as unverifiable rather than as the
        # correct calculation it actually contains. A period followed by
        # whitespace cannot be a decimal point, so it is treated as a
        # boundary and only the final fragment is retained.
        # Take the LAST fragment that still looks like an expression, not
        # simply the last fragment: an expression ending in a period (e.g.
        # "... * 100.") would otherwise split to an empty trailing fragment
        # and be discarded, which is how an earlier version of this fix
        # broke the two regression tests below.
        fragments = re.split(r"\.\s+", span)
        qualified = [
            frag for frag in fragments
            if re.search(r"[+\-*/]", frag) and len(re.findall(r"\d+(?:\.\d+)?", frag)) >= 2
        ]
        span = qualified[-1] if qualified else span

        if not re.search(r"[+\-*/]", span):
            continue
        if len(re.findall(r"\d+(?:\.\d+)?", span)) < 2:
            continue
        best = span  # keep overwriting -- last qualifying match wins

    if best is None:
        return None

    cleaned = _clean_expr(best)
    if not cleaned or not re.search(r"[+\-*/]", cleaned):
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    parseable: bool
    expression: Optional[str]
    stated_result: Optional[float]
    recomputed_result: Optional[float]
    execution_consistent: Optional[bool]     # None if not parseable
    operand_count: int
    grounded_operand_count: int
    ungrounded_operands: List[float] = field(default_factory=list)
    operands_grounded: Optional[bool] = None  # None if not parseable / no operands
    verified: bool = False
    reason: str = ""

    def as_dict(self):
        return {
            "parseable": self.parseable,
            "expression": self.expression,
            "stated_result": self.stated_result,
            "recomputed_result": self.recomputed_result,
            "execution_consistent": self.execution_consistent,
            "operand_count": self.operand_count,
            "grounded_operand_count": self.grounded_operand_count,
            "ungrounded_operands": self.ungrounded_operands,
            "operands_grounded": self.operands_grounded,
            "verified": self.verified,
            "reason": self.reason,
        }


class PostHocArithmeticVerifier:
    """Gold-free, deterministic verification of a RAG response's own
    stated arithmetic. See module docstring for exactly what this does
    and does not check."""

    # Common arithmetic/unit-conversion constants (percent conversion,
    # months per year, day-count conventions, thousand/million scaling)
    # that legitimately appear in a financial calculation without being
    # sourced from the retrieved evidence. These are exempt from the
    # operand-grounding check -- flagging "* 100" in a percentage
    # calculation as a "hallucinated input" would be a false positive,
    # not a real hallucination.
    _CONVERSION_CONSTANTS = {100.0, 1000.0, 1000000.0, 12.0, 4.0, 360.0, 365.0}

    def __init__(
        self,
        nrf: Optional[NumericalReliabilityFramework] = None,
        relative_tolerance: float = 0.02,
        absolute_tolerance: float = 0.01,
    ):
        self.nrf = nrf if nrf is not None else NumericalReliabilityFramework()
        self.relative_tolerance = relative_tolerance
        self.absolute_tolerance = absolute_tolerance

    def _close(self, a: float, b: float) -> bool:
        tolerance = max(self.absolute_tolerance, abs(b) * self.relative_tolerance)
        return abs(a - b) <= tolerance

    def verify(self, response: str, evidence_text: str) -> VerificationResult:
        reasoning_match = _REASONING_SECTION_RE.search(response or "")
        reasoning_text = reasoning_match.group(1) if reasoning_match else (response or "")

        expr = find_last_expression(reasoning_text)
        if expr is None:
            return VerificationResult(
                parseable=False, expression=None, stated_result=None,
                recomputed_result=None, execution_consistent=None,
                operand_count=0, grounded_operand_count=0,
                reason="No arithmetic expression (with an operator and >=2 numbers) found in the Reasoning section.",
            )

        try:
            recomputed = safe_eval_arithmetic(expr)
        except (UnsafeExpressionError, SyntaxError, ZeroDivisionError, TypeError, ValueError) as e:
            return VerificationResult(
                parseable=False, expression=expr, stated_result=None,
                recomputed_result=None, execution_consistent=None,
                operand_count=0, grounded_operand_count=0,
                reason=f"Extracted expression could not be safely evaluated: {e}",
            )

        # Compare Python's independent recomputation of the Reasoning
        # section's expression against the model's separately-stated
        # Final Answer -- the two are not required to appear together in
        # one "expr = result" clause (see module docstring's REVISION
        # NOTE for why the earlier, stricter version missed real errors).
        final_answer = self.nrf.extract_final_answer(response or "")

        if final_answer is None:
            return VerificationResult(
                parseable=True, expression=expr, stated_result=None,
                recomputed_result=recomputed, execution_consistent=None,
                operand_count=0, grounded_operand_count=0,
                reason="Expression recomputed but no 'Final Answer:' value found to compare it against.",
            )

        stated_result = final_answer.value
        execution_consistent = self._close(recomputed, stated_result)

        operands = self.nrf.extract_numbers(expr)
        evidence_numbers = self.nrf.extract_numbers(evidence_text or "")

        grounded_count = 0
        ungrounded: List[float] = []
        for op in operands:
            if not op.is_percentage and op.value in self._CONVERSION_CONSTANTS:
                grounded_count += 1
                continue
            found = any(
                op.is_percentage == en.is_percentage and self._close(op.value, en.value)
                for en in evidence_numbers
            )
            if found:
                grounded_count += 1
            else:
                ungrounded.append(op.value)

        operands_grounded = (len(ungrounded) == 0) if operands else None
        verified = bool(execution_consistent and (operands_grounded is not False))

        if verified:
            reason = "Operands are grounded in retrieved evidence and Python's recomputation matches the stated answer."
        elif operands_grounded is False:
            reason = f"Operand(s) not found in retrieved evidence (possible hallucinated input): {ungrounded}"
        else:
            reason = (
                f"Python recomputation of '{expr}' = {recomputed:.4f}, which does not match "
                f"the stated result ({stated_result})."
            )

        return VerificationResult(
            parseable=True, expression=expr, stated_result=stated_result,
            recomputed_result=recomputed, execution_consistent=execution_consistent,
            operand_count=len(operands), grounded_operand_count=grounded_count,
            ungrounded_operands=ungrounded, operands_grounded=operands_grounded,
            verified=verified, reason=reason,
        )


# ---------------------------------------------------------------------------
# Explicit parsing tests (run directly: `python framework/verifier.py`)
# ---------------------------------------------------------------------------

def _run_tests():
    v = PostHocArithmeticVerifier()
    failures = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    evidence = "Net revenue was $5829 million in 2015 and $5735 million in 2014."

    r1 = v.verify(
        "Reasoning: $5829 - $5735 = $94 million\n\nFinal Answer: $94 million\n\nConfidence: 1",
        evidence,
    )
    check("correct_grounded_calculation_verifies", r1.verified is True)

    r2 = v.verify(
        "Reasoning: $5829 - $5735 = $104 million\n\nFinal Answer: $104 million\n\nConfidence: 1",
        evidence,
    )
    check("arithmetic_slip_caught", r2.verified is False and r2.execution_consistent is False)

    r3 = v.verify(
        "Reasoning: $5829 - $918.7 = $4910.3 million\n\nFinal Answer: $4910.3 million\n\nConfidence: 1",
        evidence,
    )
    check("ungrounded_operand_caught", r3.verified is False and r3.operands_grounded is False)

    r4 = v.verify("Reasoning: The net revenue increased.\n\nFinal Answer: 94\n\nConfidence: 1", evidence)
    check("unparseable_reasoning_flagged", r4.parseable is False and r4.verified is False)

    r5 = v.verify(
        "Reasoning: (153.7 - 139.9) / 139.9 * 100 = 9.86\n\nFinal Answer: 9.86%\n\nConfidence: 1",
        "as of 2011 and 2010 were $153.7 million and $139.9 million, respectively.",
    )
    check("chained_expression_verifies", r5.verified is True)

    # Regression test for the real bug found in Experiment 1's dev run
    # (SYY/2006/page_71.pdf-1): the expression is stated as a plain
    # sentence with NO inline "=result", and the model's Final Answer
    # (10.04%) does not actually match Python's recomputation of its own
    # stated expression (8.61%) -- the fixed verifier must catch this
    # real arithmetic error instead of reporting "unparseable".
    syy_evidence = (
        "total rental expense under operating leases was $100690000, $92710000, "
        "and $86842000 in fiscal 2006, 2005 and 2004, respectively."
    )
    r6 = v.verify(
        "Reasoning: The total rental expense for fiscal 2005 was $92,710,000 and for fiscal 2006 it was "
        "$100,690,000. To find the percentage change, I use the formula: "
        "((New Value - Original Value) / Original Value) * 100. So, the calculation is: "
        "((100690000 - 92710000) / 92710000) * 100.\n\nFinal Answer: 10.04%\n\nConfidence: 0.9",
        syy_evidence,
    )
    check("expression_without_inline_result_is_parsed", r6.parseable is True)
    check("real_arithmetic_slip_is_caught_not_missed", r6.verified is False and r6.execution_consistent is False)

    if failures:
        print(f"FAILED {len(failures)} test(s): {failures}")
    else:
        print("All verifier tests passed.")


if __name__ == "__main__":
    _run_tests()
