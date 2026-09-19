"""
framework/nrf.py

Numerical Reliability Framework (NRF) for Financial-NRF.

IMPORTANT SCOPE NOTES (read before citing this in the dissertation):

1. "Numerical Evidence Support Score" (NESS) — this file's `numerical_evidence_support()`
   — is NOT semantic faithfulness and NOT entailment. It checks whether numbers that
   appear in the model's response also appear (within tolerance) somewhere in the
   retrieved evidence text. It distinguishes:
       - "supported": the number appears directly in the evidence
       - "unsupported": the number does not appear in the evidence
   It does NOT currently attempt a third "derived from evidence" category (e.g.
   recognizing that 94 is derived from 5829 - 5735). Doing that correctly requires
   symbolic re-execution of an arithmetic program against the evidence, which this
   implementation does not do. Do not claim it does.

2. `self_reported_confidence()` extracts a confidence value the model was asked to
   output in text ("Confidence: 0.9"). This is NOT a calibrated probability derived
   from token log-probabilities. It is a self-reported number and must be referred to
   as such in any writeup (e.g. "ECE of self-reported confidence").

3. `hallucination_detection()` is a rule-based detector derived from NESS
   (NESS < threshold => flagged). It is a baseline heuristic, not a trained
   classifier. Document it as such.

4. ONRS (Overall Numerical Reliability Score) is a PROPOSED metric introduced by
   this project. It is not an established metric from the literature. The weights
   (0.30 / 0.30 / 0.20 / 0.20) are preserved from the original design and are not
   changed here without explicit justification.
"""

import re
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedNumber:
    """A number extracted from text, with enough metadata to compare it
    correctly against another number (in particular: was it a percentage?)."""
    raw_text: str
    value: float
    is_percentage: bool


@dataclass
class EvidenceSupportResult:
    supported_count: int
    unsupported_count: int
    total_count: int
    unsupported_numbers: List[float]
    score: float  # supported_count / total_count, or None-safe 0.0 if total_count == 0

    # --- Derived-number accounting (see numerical_evidence_support) ---
    # These are ALWAYS populated, regardless of which scoring mode is
    # active, so any result row can be audited/re-scored either way after
    # the fact without re-running generation.
    directly_supported_count: int = 0     # number appears literally in evidence
    derived_supported_count: int = 0      # number is arithmetically derivable from evidence
    derived_numbers: List[float] = field(default_factory=list)
    strict_score: float = 0.0             # directly_supported / total  (original NESS definition)
    derived_score: float = 0.0            # (direct + derived) / total  (revised NESS definition)


@dataclass
class SampleResult:
    """Per-example structured result, matching the dict shape requested in the brief."""
    accuracy: float
    faithfulness: float           # NESS score (see module docstring)
    hallucination: int
    confidence: float             # self-reported confidence
    predicted_value: Optional[float]
    gold_value: Optional[float]
    parse_success: bool           # whether a Final Answer section was found at all

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "faithfulness": self.faithfulness,
            "hallucination": self.hallucination,
            "confidence": self.confidence,
            "predicted_value": self.predicted_value,
            "gold_value": self.gold_value,
            "parse_success": self.parse_success,
        }


# ---------------------------------------------------------------------------
# Core framework
# ---------------------------------------------------------------------------

class NumericalReliabilityFramework:
    """
    Implements:
        extract_final_answer()
        numerical_accuracy()
        numerical_evidence_support()   (renamed from "faithfulness")
        hallucination_detection()
        extract_confidence()           (self-reported confidence)
        calculate_ece()
        calculate_onrs()

    All number parsing goes through `_parse_number_token` so percentage
    handling is consistent everywhere.
    """

    # Regex for the "Final Answer:" line. Captures everything up to end of
    # line (or start of "Confidence"/"Reasoning" if the model runs sections
    # together without a newline).
    _FINAL_ANSWER_LINE_RE = re.compile(
        r"final\s*answer\s*[:\-]\s*(.*?)(?:\n|$|confidence\s*[:\-]|reasoning\s*[:\-])",
        re.IGNORECASE | re.DOTALL,
    )

    # A single numeric token, optionally with a leading currency symbol,
    # thousands separators, decimal point, and a trailing percent sign.
    # Note: no whitespace is allowed between an optional +/- sign and the
    # digits that follow. This is deliberate: in text like "5829 - 5735"
    # the "-" is a subtraction operator with a space after it, not a
    # negative sign, and must NOT be absorbed into the following number
    # (doing so previously caused "5735" to fail to parse at all).
    _NUMBER_TOKEN_RE = re.compile(
        r"[-+]?\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
    )

    _CONFIDENCE_RE_LIST = [
        re.compile(r"confidence\s*[:\-]\s*(\d+(?:\.\d+)?)\s*%"),   # "Confidence: 92%"
        re.compile(r"confidence\s*[:\-]\s*(0?\.\d+)\b"),           # "Confidence: 0.9"
        re.compile(r"confidence\s*[:\-]\s*(\d+(?:\.\d+)?)\b"),     # "Confidence: 1" or "Confidence: 1.0"
    ]

    # Single-operand rescalings treated as legitimate derivations (unit
    # conversions that appear constantly in financial text: dollars <->
    # thousands <-> millions, ratio <-> percent).
    _DERIVED_SCALE_FACTORS = (100.0, 1000.0, 1_000_000.0)

    # Unit-conversion constants treated as legitimately available to the
    # model rather than as invented facts about the document. Kept in sync
    # with framework/verifier.py's _CONVERSION_CONSTANTS so the two modules
    # agree on what counts as an ungrounded operand.
    _CONVERSION_CONSTANTS = frozenset({100.0, 1000.0, 1_000_000.0, 12.0, 4.0, 360.0, 365.0})

    def __init__(
        self,
        hallucination_threshold: float = 0.5,
        ece_bins: int = 10,
        credit_derived_numbers: bool = False,
        derivation_relative_tolerance: float = 0.001,
        derivation_absolute_tolerance: float = 0.005,
    ):
        """
        credit_derived_numbers controls which NESS definition drives
        `score` (and therefore hallucination_detection and ONRS):

            False (default, unchanged behaviour): a response number counts
                as supported ONLY if it appears literally in the evidence.
            True (revised definition): a response number ALSO counts as
                supported if it is arithmetically derivable from evidence
                numbers (see numerical_evidence_support for the exact,
                closed list of derivation forms).

        The default is False so that the original, already-reported
        baseline numbers remain reproducible. Both scores are ALWAYS
        computed and returned regardless of this setting, so switching
        modes never requires re-running generation, and any reported
        figure can be cross-checked against the other definition.

        derivation_*_tolerance are DELIBERATELY tighter than the 2%
        relative tolerance used for literal evidence matching. Literal
        matching compares a number against a specific evidence number, so
        a loose tolerance costs little. Derivation matching compares a
        number against thousands of candidate arithmetic combinations, so
        a loose tolerance would start matching numbers by coincidence and
        silently inflate NESS. These defaults are set just wide enough to
        absorb the model's display rounding (e.g. stating 14.46 for a true
        value of 14.464286) and no wider. See _run_tests() for the
        empirical false-positive check on this choice.
        """
        self.hallucination_threshold = hallucination_threshold
        self.ece_bins = ece_bins
        self.credit_derived_numbers = credit_derived_numbers
        self.derivation_relative_tolerance = derivation_relative_tolerance
        self.derivation_absolute_tolerance = derivation_absolute_tolerance

    # ------------------------------------------------------------------
    # Low-level number parsing
    # ------------------------------------------------------------------

    def _parse_number_token(self, token: str) -> Optional[ParsedNumber]:
        """Parse a single numeric token (e.g. '$94', '32.5%', '1,234.5')
        into a ParsedNumber, or None if it can't be parsed."""
        raw = token.strip()
        if not raw:
            return None

        is_percentage = raw.endswith("%")
        cleaned = raw.replace("$", "").replace(",", "").replace("%", "").strip()

        if cleaned in ("", "-", "+", "."):
            return None

        try:
            value = float(cleaned)
        except ValueError:
            return None

        return ParsedNumber(raw_text=raw, value=value, is_percentage=is_percentage)

    def extract_numbers(self, text: str) -> List[ParsedNumber]:
        """Extract all numeric tokens from arbitrary text (used for evidence
        support checking, NOT for answer extraction)."""
        if not text:
            return []

        results = []
        for match in self._NUMBER_TOKEN_RE.finditer(str(text)):
            parsed = self._parse_number_token(match.group(0))
            if parsed is not None:
                results.append(parsed)
        return results

    # ------------------------------------------------------------------
    # Final answer extraction (THE key fix)
    # ------------------------------------------------------------------

    def extract_final_answer(self, response: str) -> Optional[ParsedNumber]:
        """
        Extract ONLY the number associated with a 'Final Answer:' section.
        Does NOT fall back to scanning the whole response for a trailing
        number, because that is what caused Confidence values to be
        mistaken for the answer in the original implementation.

        If no 'Final Answer:' section is found at all, returns None
        (the fallback parser, `extract_final_answer_with_fallback`, handles
        that case explicitly and marks it as a parse failure rather than
        silently guessing).
        """
        if not response:
            return None

        match = self._FINAL_ANSWER_LINE_RE.search(response)
        if not match:
            return None

        segment = match.group(1)

        # Within the Final Answer segment, there may be an expression like
        # "$5829 - $5735 = $94 million". The actual answer is the LAST
        # number in that segment (the result of the calculation), not the
        # first. This is a deliberate, scoped use of "last number" — scoped
        # to the Final Answer segment only, not the whole response.
        numbers_in_segment = self.extract_numbers(segment)
        if not numbers_in_segment:
            return None

        return numbers_in_segment[-1]

    def extract_final_answer_with_fallback(self, response: str) -> (Optional[ParsedNumber], bool):
        """
        Robust wrapper: tries the strict Final Answer parse first.
        If that fails (model didn't follow the format), falls back to the
        last number in the entire response, but flags parse_success=False
        so that this fallback path is auditable/reportable rather than
        silently indistinguishable from a clean parse.

        Returns (parsed_number_or_None, parse_success_bool).
        """
        strict = self.extract_final_answer(response)
        if strict is not None:
            return strict, True

        # Fallback: whole-response scan (same rough behaviour as the
        # original buggy implementation), but only used as a last resort
        # and clearly marked as such downstream.
        all_numbers = self.extract_numbers(response)
        if not all_numbers:
            return None, False

        return all_numbers[-1], False

    # ------------------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------------------

    def numerical_accuracy(
        self,
        prediction: str,
        gold: str,
        relative_tolerance: float = 0.02,
        absolute_tolerance: float = 0.01,
        use_fallback_parser: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare the model's Final Answer to the gold answer.

        Percentage handling: if exactly one of (prediction, gold) is a
        percentage and the other is not, they are NOT treated as equal
        even if the numeric magnitudes match (e.g. predicted '32.5' vs
        gold '32.5%' is a mismatch), because that conflates two different
        units. This is a deliberate, conservative choice — flag it in your
        writeup as a chosen policy, not an implementation accident.

        Returns a dict with the correctness flag plus the parsed values,
        so you can audit exactly what was compared.
        """
        if use_fallback_parser:
            pred_num, pred_parse_ok = self.extract_final_answer_with_fallback(prediction)
        else:
            pred_num = self.extract_final_answer(prediction)
            pred_parse_ok = pred_num is not None

        gold_numbers = self.extract_numbers(gold)
        gold_num = gold_numbers[-1] if gold_numbers else None

        if pred_num is None or gold_num is None:
            return {
                "correct": 0.0,
                "predicted_value": pred_num.value if pred_num else None,
                "gold_value": gold_num.value if gold_num else None,
                "parse_success": pred_parse_ok,
                "sign_mismatch": False,
            }

        if pred_num.is_percentage != gold_num.is_percentage:
            correct = 0.0
        else:
            tolerance = max(absolute_tolerance, abs(gold_num.value) * relative_tolerance)
            correct = float(abs(pred_num.value - gold_num.value) <= tolerance)

        # Diagnostic only -- does NOT change `correct`. Flags cases where
        # the predicted value has the right magnitude but the opposite
        # sign of the gold value (e.g. predicted -9.9, gold 9.9). This is
        # reported separately so a genuine sign/direction-convention
        # mismatch is visible and auditable in results rather than either
        # (a) silently counted as correct, which would be scoring
        # inflation, or (b) invisible among ordinary wrong answers, which
        # would hide a real, distinct failure mode from the write-up.
        sign_mismatch = False
        if pred_num.is_percentage == gold_num.is_percentage and correct == 0.0:
            tolerance = max(absolute_tolerance, abs(gold_num.value) * relative_tolerance)
            sign_mismatch = abs(-pred_num.value - gold_num.value) <= tolerance

        return {
            "correct": correct,
            "predicted_value": pred_num.value,
            "gold_value": gold_num.value,
            "parse_success": pred_parse_ok,
            "sign_mismatch": sign_mismatch,
        }

    # ------------------------------------------------------------------
    # Numerical Evidence Support Score (formerly mislabeled "faithfulness")
    # ------------------------------------------------------------------

    def _build_derivable_value_set(self, seed_values: List[float]) -> np.ndarray:
        """
        Precompute every value reachable from `seed_values` by ONE
        elementary arithmetic step, returned as a sorted array for fast
        lookup.

        CRITICAL DESIGN POINT -- what goes into `seed_values`. An earlier
        version of this method seeded the search with EVERY number in the
        retrieved evidence. That was measured (see _run_tests) to produce
        an ~86% false-positive rate: with ~40 evidence numbers there are
        ~15,000 candidate combinations, densely enough packed that an
        arbitrary invented number lands within tolerance of one of them
        almost every time. Crediting numbers that way would have inflated
        NESS on invented values -- the exact opposite of what the metric
        is for.

        The seed set is therefore restricted to the evidence numbers the
        model ITSELF cited in this response (i.e. response numbers that
        already matched the evidence literally), plus a small closed set
        of unit-conversion constants. This is both far tighter -- a
        handful of seeds instead of forty -- and semantically the right
        question: "is this number computed from the evidence values this
        response actually used?" rather than "does some combination of
        anything in the document happen to hit this value?"

        Derivation forms (closed list; each corresponds to an operation
        FinQA questions actually ask for):

            unary:   -a,  a*k,  a/k        for k in {100, 1000, 1e6}
            binary:  a+b,  a-b,  a*b,  a/b
                     (a+b)/2                    [average of two periods]
                     (a/b)*100                  [share/ratio as a percent]
                     (a-b)/b,  ((a-b)/b)*100    [change and percent change]

        NOT included (documented limitation, not an oversight): any
        derivation needing three or more distinct source numbers, e.g. the
        mean of three years. Such numbers stay classified as unsupported.
        """
        values = set()
        n = len(seed_values)

        for i in range(n):
            a = seed_values[i]
            values.add(-a)
            for scale in self._DERIVED_SCALE_FACTORS:
                values.add(a * scale)
                values.add(a / scale)

            for j in range(n):
                if i == j:
                    continue
                b = seed_values[j]
                values.add(a + b)
                values.add(a - b)
                values.add(a * b)
                values.add((a + b) / 2.0)
                if b != 0.0:
                    quotient = a / b
                    values.add(quotient)
                    values.add(quotient * 100.0)
                    change = (a - b) / b
                    values.add(change)
                    values.add(change * 100.0)

        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return np.array([], dtype=float)
        return np.sort(np.array(finite, dtype=float))

    @staticmethod
    def _stated_precision_tolerance(parsed: "ParsedNumber") -> float:
        """
        Tolerance derived from the precision the number was WRITTEN at,
        rather than a flat relative tolerance.

        A model that computes 8.1/56.0*100 = 14.464286 and writes "14.46"
        has not made an error -- it has rounded for display. So a derived
        candidate should count as matching a stated value if it agrees to
        the precision that value was stated at: half a unit in the last
        written decimal place ("14.46" -> 0.005, "94" -> 0.5).

        This matters more than it sounds: it is what keeps the derivation
        search honest. A flat relative tolerance is far too generous on
        large values (0.1% of 2,500 is +/-2.5, a wide net to cast over
        thousands of candidates), whereas stated-precision tolerance stays
        tight exactly where the candidate set is densest.
        """
        cleaned = parsed.raw_text.strip().replace("$", "").replace(",", "").replace("%", "")
        if "." in cleaned:
            decimals = len(cleaned.split(".")[-1])
            tolerance = 0.5 * (10.0 ** -decimals)
        else:
            tolerance = 0.5
        # Tiny floating-point slack so exact derivations aren't lost to
        # binary representation error.
        return tolerance + abs(parsed.value) * 1e-9

    def _matches_sorted(self, target: float, sorted_values: np.ndarray, tolerance: float) -> bool:
        """Tolerance-aware membership test against a sorted array, via
        binary search rather than a linear scan."""
        if sorted_values.size == 0:
            return False
        idx = int(np.searchsorted(sorted_values, target))
        for k in (idx - 1, idx):
            if 0 <= k < sorted_values.size and abs(float(sorted_values[k]) - target) <= tolerance:
                return True
        return False

    def numerical_evidence_support(
        self,
        response: str,
        evidence: str,
        credit_derived: Optional[bool] = None,
    ) -> EvidenceSupportResult:
        """
        Numerical Evidence Support Score (NESS).

        Two definitions are computed on every call; `credit_derived`
        (defaulting to the instance's `credit_derived_numbers`) selects
        which one is returned as `.score`. Both are always available as
        `.strict_score` and `.derived_score`.

        STRICT (original definition): the fraction of numeric values in the
        response that also appear literally, within tolerance, in the
        retrieved evidence.

        DERIVED-CREDIT (revised definition): a response number counts as
        supported if it appears literally in the evidence OR is reachable
        from evidence numbers by one elementary arithmetic step (see
        _build_derivable_value_set for the exact closed list).

        WHY THE REVISION EXISTS -- state this in any write-up, do not
        present the revised score as if it were the original metric:
        under the strict definition, a CORRECT answer to an arithmetic
        question is counted as unsupported, because the computed result
        does not itself appear in the source text. Measured on this
        project's n=50 dev run, zero of the 15 fully-correct answers
        achieved a strict NESS of 1.0 (mean 0.686). The strict metric
        therefore penalises the system for doing the arithmetic the task
        requires, and a perfect system cannot score 1.0 on it -- which
        also caps ONRS well below 1.0 by construction. The revised
        definition measures what "evidence support" was intended to mean:
        every number the model used is either read from the evidence or
        computed from it, and nothing was invented.

        WHAT THE REVISION DOES NOT DO: it still does not check that a
        supported number was used in a LOGICALLY correct way. A model
        that divides the right two numbers in the wrong order produces a
        derivable -- and therefore "supported" -- number. NESS remains a
        grounding measure, not a correctness measure; accuracy and the
        post-hoc verifier are what catch that case.
        """
        if credit_derived is None:
            credit_derived = self.credit_derived_numbers

        response_numbers = self.extract_numbers(response)
        evidence_numbers = self.extract_numbers(evidence)

        if not response_numbers:
            return EvidenceSupportResult(0, 0, 0, [], 0.0)

        total = len(response_numbers)

        if not evidence_numbers:
            return EvidenceSupportResult(
                supported_count=0,
                unsupported_count=total,
                total_count=total,
                unsupported_numbers=[n.value for n in response_numbers],
                score=0.0,
                directly_supported_count=0,
                derived_supported_count=0,
                derived_numbers=[],
                strict_score=0.0,
                derived_score=0.0,
            )

        direct_count = 0
        not_direct: List[ParsedNumber] = []
        cited_evidence_values: List[float] = []

        for rn in response_numbers:
            found = False
            for en in evidence_numbers:
                # Only compare like-with-like on percentage-ness to avoid
                # spuriously matching e.g. "32.5" against "32.5%".
                if rn.is_percentage != en.is_percentage:
                    continue
                tolerance = max(0.01, abs(en.value) * 0.02)
                if abs(rn.value - en.value) <= tolerance:
                    found = True
                    break
            if found:
                direct_count += 1
                cited_evidence_values.append(rn.value)
            else:
                not_direct.append(rn)

        derived_values: List[float] = []
        unsupported_values: List[float] = []

        if not_direct:
            # Seed the derivation search ONLY with the evidence numbers this
            # response actually cited, plus unit-conversion constants -- see
            # _build_derivable_value_set for why seeding it with the whole
            # evidence set was measured to be unusable (~86% false positives).
            #
            # A conversion constant (100 for percent, 1000/1e6 for unit
            # scaling, 12/4 for months/quarters, 365/360 for day-count) is
            # treated as legitimately available rather than invented: writing
            # "* 100" to convert a ratio to a percent is not an unsupported
            # factual claim about the document. This mirrors the same
            # exemption already applied in framework/verifier.py, so the two
            # modules agree on what counts as an ungrounded operand.
            seeds = sorted(set(cited_evidence_values) | set(self._CONVERSION_CONSTANTS))
            derivable = self._build_derivable_value_set(seeds)
            conversion_array = np.sort(np.array(sorted(self._CONVERSION_CONSTANTS), dtype=float))

            for rn in not_direct:
                tol = self._stated_precision_tolerance(rn)
                # NOTE: derivation matching compares MAGNITUDES only and does
                # not require percentage-ness to match, unlike literal
                # matching above. That is deliberate and is the whole point:
                # a derived percentage (14.46%) is computed FROM
                # non-percentage evidence numbers (8.1 and 56.0), so requiring
                # the flags to agree would reject exactly the cases this is
                # meant to credit.
                if self._matches_sorted(rn.value, conversion_array, tol) or \
                        self._matches_sorted(rn.value, derivable, tol):
                    derived_values.append(rn.value)
                else:
                    unsupported_values.append(rn.value)

        derived_count = len(derived_values)
        strict_score = direct_count / total
        derived_score = (direct_count + derived_count) / total

        if credit_derived:
            supported = direct_count + derived_count
            score = derived_score
            reported_unsupported = unsupported_values
        else:
            supported = direct_count
            score = strict_score
            # Under the strict definition a derived number IS unsupported,
            # so it must still be reported as such -- otherwise the strict
            # score and its own unsupported list would disagree.
            reported_unsupported = derived_values + unsupported_values

        return EvidenceSupportResult(
            supported_count=supported,
            unsupported_count=total - supported,
            total_count=total,
            unsupported_numbers=reported_unsupported,
            score=score,
            directly_supported_count=direct_count,
            derived_supported_count=derived_count,
            derived_numbers=derived_values,
            strict_score=strict_score,
            derived_score=derived_score,
        )

    # ------------------------------------------------------------------
    # Hallucination detection (rule-based baseline, derived from NESS)
    # ------------------------------------------------------------------

    def hallucination_detection(self, response: str, evidence: str) -> Dict[str, Any]:
        """
        Baseline rule-based hallucination flag: NESS score below
        `self.hallucination_threshold` => flagged as hallucinated.

        This is NOT a trained/learned hallucination classifier. It is a
        threshold rule on top of the numeric evidence support score.
        """
        support = self.numerical_evidence_support(response, evidence)
        flagged = int(support.score < self.hallucination_threshold)
        return {
            "hallucination": flagged,
            "ness_score": support.score,
            "unsupported_numbers": support.unsupported_numbers,
        }

    # ------------------------------------------------------------------
    # Confidence (self-reported)
    # ------------------------------------------------------------------

    def extract_self_reported_confidence(self, response: str) -> Optional[float]:
        """
        Extract the self-reported confidence value the model was asked to
        output. Returns None if no confidence value is found — callers
        should decide explicitly how to handle missing confidence (the
        previous implementation silently defaulted to 0.5, which biases
        ECE; that default is NOT applied here).
        """
        if not response:
            return None

        lower = response.lower()

        # Percent form first (e.g. "Confidence: 92%")
        m = self._CONFIDENCE_RE_LIST[0].search(lower)
        if m:
            value = float(m.group(1)) / 100.0
            return min(max(value, 0.0), 1.0)

        # Decimal < 1 form (e.g. "Confidence: 0.9")
        m = self._CONFIDENCE_RE_LIST[1].search(lower)
        if m:
            value = float(m.group(1))
            return min(max(value, 0.0), 1.0)

        # Bare number form (e.g. "Confidence: 1" meaning 1.0, or possibly "Confidence: 85")
        m = self._CONFIDENCE_RE_LIST[2].search(lower)
        if m:
            value = float(m.group(1))
            if value > 1.0:
                value /= 100.0
            return min(max(value, 0.0), 1.0)

        return None

    # ------------------------------------------------------------------
    # Expected Calibration Error (of self-reported confidence)
    # ------------------------------------------------------------------

    def calculate_ece(self, confidences: List[float], correctness: List[float]) -> float:
        """
        Standard binned ECE:
            ECE = sum_over_bins [ (n_bin / N) * |acc_bin - conf_bin| ]

        NOTE: when `confidences` come from `extract_self_reported_confidence`,
        this is the "ECE of self-reported confidence", NOT a calibration
        measure of a genuine model probability. Label it that way in any
        report/figure.
        """
        if not confidences or not correctness:
            return 0.0
        if len(confidences) != len(correctness):
            raise ValueError("confidences and correctness must be the same length")

        confidences_arr = np.array(confidences, dtype=float)
        correctness_arr = np.array(correctness, dtype=float)

        if np.any((confidences_arr < 0) | (confidences_arr > 1)):
            raise ValueError("confidences must be in [0, 1]")

        bins = np.linspace(0.0, 1.0, self.ece_bins + 1)
        ece = 0.0
        n = len(confidences_arr)

        for lower, upper in zip(bins[:-1], bins[1:]):
            if upper == bins[-1]:
                mask = (confidences_arr >= lower) & (confidences_arr <= upper)
            else:
                mask = (confidences_arr >= lower) & (confidences_arr < upper)

            if not mask.any():
                continue

            bin_acc = np.mean(correctness_arr[mask])
            bin_conf = np.mean(confidences_arr[mask])
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)

        return float(ece)

    # ------------------------------------------------------------------
    # ONRS
    # ------------------------------------------------------------------

    def calculate_onrs(
        self,
        accuracy: float,
        faithfulness: float,
        hallucination_rate: float,
        ece: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Overall Numerical Reliability Score (ONRS) — a PROPOSED metric,
        not an established one from the literature.

        Default weights (unchanged from the original design):
            0.30 * accuracy
            + 0.30 * faithfulness (NESS)
            + 0.20 * (1 - hallucination_rate)
            + 0.20 * (1 - ece)

        All four inputs must be in [0, 1]; this is validated explicitly
        rather than silently clamped, so a bug upstream (e.g. an ECE > 1)
        surfaces immediately instead of being masked.
        """
        default_weights = {"accuracy": 0.30, "faithfulness": 0.30, "hallucination": 0.20, "calibration": 0.20}
        w = weights if weights is not None else default_weights

        expected_keys = {"accuracy", "faithfulness", "hallucination", "calibration"}
        if set(w.keys()) != expected_keys:
            raise ValueError(f"weights must have exactly keys {expected_keys}, got {set(w.keys())}")

        total_weight = sum(w.values())
        if not math.isclose(total_weight, 1.0, abs_tol=1e-6):
            raise ValueError(f"weights must sum to 1.0, got {total_weight}")

        for name, value in [
            ("accuracy", accuracy),
            ("faithfulness", faithfulness),
            ("hallucination_rate", hallucination_rate),
            ("ece", ece),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        calibration = 1.0 - ece
        hallucination_score = 1.0 - hallucination_rate

        onrs = (
            w["accuracy"] * accuracy
            + w["faithfulness"] * faithfulness
            + w["hallucination"] * hallucination_score
            + w["calibration"] * calibration
        )

        onrs = max(0.0, min(1.0, onrs))
        assert 0.0 <= onrs <= 1.0
        return onrs

    # ------------------------------------------------------------------
    # Convenience: evaluate a single sample end-to-end
    # ------------------------------------------------------------------

    def evaluate_sample(self, response: str, gold: str, evidence: str) -> SampleResult:
        acc_result = self.numerical_accuracy(response, gold)
        support = self.numerical_evidence_support(response, evidence)
        halluc = self.hallucination_detection(response, evidence)
        confidence = self.extract_self_reported_confidence(response)

        return SampleResult(
            accuracy=acc_result["correct"],
            faithfulness=support.score,
            hallucination=halluc["hallucination"],
            confidence=confidence if confidence is not None else float("nan"),
            predicted_value=acc_result["predicted_value"],
            gold_value=acc_result["gold_value"],
            parse_success=acc_result["parse_success"],
        )


# ---------------------------------------------------------------------------
# Explicit parsing tests (run directly: `python framework/nrf.py`)
# ---------------------------------------------------------------------------

def _run_tests():
    nrf = NumericalReliabilityFramework()
    failures = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # --- extract_final_answer: the core bug fix ---
    r1 = "Reasoning:\n5829 - 5735 = 94\n\nFinal Answer: 94\n\nConfidence: 1"
    p1 = nrf.extract_final_answer(r1)
    check("final_answer_not_confused_with_confidence", p1 is not None and p1.value == 94.0)

    r2 = "Final Answer: $5829 - $5735 = $94 million\n\nConfidence: 1"
    p2 = nrf.extract_final_answer(r2)
    check("final_answer_expression_takes_last_number", p2 is not None and p2.value == 94.0)

    r3 = "Final Answer: 94.0\nConfidence: 0.85"
    p3 = nrf.extract_final_answer(r3)
    check("final_answer_plain_decimal", p3 is not None and p3.value == 94.0)

    r4 = "Final Answer: 94 million\nConfidence: 1"
    p4 = nrf.extract_final_answer(r4)
    check("final_answer_with_unit_word", p4 is not None and p4.value == 94.0)

    r5 = "Final Answer: 32.5%\nConfidence: 0.7"
    p5 = nrf.extract_final_answer(r5)
    check("final_answer_percentage", p5 is not None and p5.value == 32.5 and p5.is_percentage)

    r6 = "Final Answer: 1,234.5\nConfidence: 1"
    p6 = nrf.extract_final_answer(r6)
    check("final_answer_thousands_separator", p6 is not None and p6.value == 1234.5)

    r7 = "no structured output here, just 42"
    p7 = nrf.extract_final_answer(r7)
    check("final_answer_missing_returns_none", p7 is None)

    fallback_num, fallback_ok = nrf.extract_final_answer_with_fallback(r7)
    check("fallback_used_when_no_final_answer_section", fallback_ok is False and fallback_num is not None)

    # --- numerical_accuracy ---
    acc1 = nrf.numerical_accuracy("Final Answer: 94\nConfidence: 1", "94")
    check("accuracy_exact_match", acc1["correct"] == 1.0)

    acc2 = nrf.numerical_accuracy("Final Answer: 1\nConfidence: 1", "94")
    check("accuracy_confidence_not_mistaken_for_answer", acc2["correct"] == 0.0 and acc2["predicted_value"] == 1.0)

    acc3 = nrf.numerical_accuracy("Final Answer: 32.5%\nConfidence: 1", "32.5")
    check("accuracy_percentage_vs_nonpercentage_mismatch", acc3["correct"] == 0.0)

    acc4 = nrf.numerical_accuracy("Final Answer: 32.5%\nConfidence: 1", "32.5%")
    check("accuracy_percentage_vs_percentage_match", acc4["correct"] == 1.0)

    # --- numerical_evidence_support ---
    evidence = "Net revenue was $5829 million in 2015 and $5735 million in 2014."
    support_good = nrf.numerical_evidence_support(
        "Final Answer: 5829 - 5735 = 94", evidence
    )
    check(
        "ness_flags_unsupported_derived_number",
        support_good.supported_count == 2 and 94.0 in support_good.unsupported_numbers,
    )

    support_all_supported = nrf.numerical_evidence_support("5829 and 5735", evidence)
    check("ness_all_supported_scores_1", support_all_supported.score == 1.0)

    # --- NESS derived-number credit (revised definition) ---
    # Same input as support_good above: 94 is 5829 - 5735, i.e. correctly
    # derived from evidence but not literally present in it.
    derived_result = nrf.numerical_evidence_support(
        "Final Answer: 5829 - 5735 = 94", evidence, credit_derived=True
    )
    check("ness_derived_credits_correct_subtraction", derived_result.score == 1.0)
    check("ness_derived_records_which_number_was_derived", 94.0 in derived_result.derived_numbers)
    check(
        "ness_both_scores_always_populated",
        math.isclose(derived_result.strict_score, 2 / 3, abs_tol=1e-9)
        and derived_result.derived_score == 1.0,
    )
    # The strict path must ALSO report both, so a run can be re-scored
    # either way after the fact without regenerating anything.
    strict_result = nrf.numerical_evidence_support(
        "Final Answer: 5829 - 5735 = 94", evidence, credit_derived=False
    )
    check(
        "ness_strict_path_still_reports_derived_score",
        strict_result.score == strict_result.strict_score and strict_result.derived_score == 1.0,
    )

    # Percentage derived from two non-percentage evidence numbers: the
    # canonical case the strict metric was penalising.
    pct_evidence = "leased total 8.1 and total facilities 56.0 square feet"
    pct_result = nrf.numerical_evidence_support(
        "8.1 / 56.0 * 100 = 14.46. Final Answer: 14.46%", pct_evidence, credit_derived=True
    )
    check("ness_derived_credits_percentage_computation", pct_result.score == 1.0)

    # An INVENTED number must still be flagged even in derived mode --
    # this is the property that stops the revision being a score hack.
    invented = nrf.numerical_evidence_support(
        "Final Answer: 7777777", evidence, credit_derived=True
    )
    check("ness_derived_still_flags_invented_number", invented.score == 0.0)

    # Empirical false-positive check -- the property that decides whether
    # derived-credit is a legitimate metric revision or a score hack.
    #
    # This models the REALISTIC adversarial case, not an easy one: a
    # response that correctly cites several real evidence numbers (so the
    # derivation search is fully seeded and active) and then states one
    # additional number that is NOT derivable from them -- i.e. exactly
    # what a hallucinated figure looks like. That invented number must
    # still be flagged as unsupported.
    #
    # Recorded for the write-up: seeding this search with ALL evidence
    # numbers instead of only the cited ones was measured at ~86% false
    # positives, i.e. it credited almost any invented value. That is why
    # _build_derivable_value_set is restricted as it is.
    rng = np.random.default_rng(12345)
    false_positives = 0
    trials = 400
    for _ in range(trials):
        evidence_vals = [round(float(v), 2) for v in rng.uniform(1, 5000, size=40)]
        fp_evidence = " ".join(str(v) for v in evidence_vals)
        cited = [evidence_vals[int(k)] for k in rng.integers(0, len(evidence_vals), size=4)]
        invented = round(float(rng.uniform(1, 5000)), 2)
        fp_response = (
            "Reasoning: using "
            + ", ".join(str(c) for c in cited)
            + f".\n\nFinal Answer: {invented}"
        )
        probe_result = nrf.numerical_evidence_support(
            fp_response, fp_evidence, credit_derived=True
        )
        if invented in probe_result.derived_numbers:
            false_positives += 1
    fp_rate = false_positives / trials
    check("ness_derived_false_positive_rate_is_low", fp_rate <= 0.10)
    print(f"  [info] derived-credit false-positive rate (invented number alongside "
          f"4 correctly-cited evidence numbers): {fp_rate:.1%}")

    # --- extract_self_reported_confidence ---
    check("confidence_decimal", nrf.extract_self_reported_confidence("Confidence: 0.9") == 0.9)
    check("confidence_bare_one", nrf.extract_self_reported_confidence("Confidence: 1") == 1.0)
    check("confidence_percent", nrf.extract_self_reported_confidence("Confidence: 85%") == 0.85)
    check("confidence_missing_returns_none", nrf.extract_self_reported_confidence("no confidence here") is None)

    # --- calculate_ece ---
    ece_perfect = nrf.calculate_ece([1.0, 1.0], [1.0, 1.0])
    check("ece_perfect_calibration_is_zero", math.isclose(ece_perfect, 0.0, abs_tol=1e-9))

    ece_worst = nrf.calculate_ece([1.0, 1.0], [0.0, 0.0])
    check("ece_worst_case_is_one", math.isclose(ece_worst, 1.0, abs_tol=1e-9))

    # --- calculate_onrs ---
    onrs_perfect = nrf.calculate_onrs(1.0, 1.0, 0.0, 0.0)
    check("onrs_perfect_case_is_one", math.isclose(onrs_perfect, 1.0, abs_tol=1e-9))

    onrs_worst = nrf.calculate_onrs(0.0, 0.0, 1.0, 1.0)
    check("onrs_worst_case_is_zero", math.isclose(onrs_worst, 0.0, abs_tol=1e-9))

    try:
        nrf.calculate_onrs(1.5, 1.0, 0.0, 0.0)
        check("onrs_rejects_out_of_range_input", False)
    except ValueError:
        check("onrs_rejects_out_of_range_input", True)

    if failures:
        print(f"FAILED {len(failures)} test(s): {failures}")
    else:
        print("All NRF parsing tests passed.")


if __name__ == "__main__":
    _run_tests()