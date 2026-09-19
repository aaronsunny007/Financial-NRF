"""
rag_pipeline/rag.py

Genuine retrieval-augmented pipeline for Financial-NRF.

Architecture (matches the corrected diagram in the project brief):

    FinQA item
        -> chunk_document()          [document/table processing + chunking]
        -> DocumentIndex.build()     [embeddings + vector index]
        -> DocumentIndex.search()    [top-k retrieval]
        -> build_evidence_text()     [retrieved financial evidence]
        -> generate()                [Qwen2.5-3B-Instruct, 4-bit, fallback 1.5B fp16]
        -> structured Reasoning / Final Answer / Confidence output

Design choices, stated explicitly (per project requirements: don't change
things silently):

1. CHUNKING GRANULARITY: FinQA's `pre_text` and `post_text` fields are
   already lists of sentences/lines as segmented by the dataset itself.
   Rather than re-splitting that text with a heuristic sentence splitter
   (which risks breaking numeric expressions like "$5,829 million" across
   chunks), each existing list element is used as one chunk. Each table
   row is its own chunk, formatted as "header: value | header: value" so
   that a retrieved row is interpretable in isolation without needing the
   full table.

2. EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2, as suggested
   in the brief. It is small (~80MB), fast on CPU or GPU, and has a
   384-dim output, which keeps memory/index overhead trivial compared to
   the LLM itself.

3. VECTOR INDEX: FAISS (IndexFlatIP over L2-normalized embeddings, i.e.
   exact cosine similarity) is used when the `faiss` package is
   installed. Because the corpus being indexed here is a SINGLE
   document's chunks (typically tens, not millions, of chunks), exact
   search is cheap and there is no need for an approximate index (IVF,
   HNSW, etc.) — that would add complexity without a measurable speed
   benefit at this scale, and would also make retrieval non-exact/less
   reproducible. If `faiss` is not installed, a pure NumPy cosine
   similarity fallback (`_NumpyFlatIndex`) is used automatically, giving
   IDENTICAL retrieval results (both are exact nearest-neighbor search
   over the same normalized vectors) with zero loss of correctness. This
   fallback also means the pipeline still runs on a machine where the
   FAISS wheel fails to install (a real, common issue on Windows), which
   is why it's implemented rather than treated as a hard dependency.

4. DETERMINISM: generation uses `do_sample=False` (greedy decoding), and
   a global seed is set for anything else that touches RNG state (e.g.
   library-level nondeterminism in some FAISS builds). Retrieval itself
   is exact nearest-neighbor search, which is already deterministic given
   fixed embeddings.

5. CACHING: chunk embeddings are cached per source document (keyed by a
   hash of pre_text + table + post_text) rather than per FinQA question,
   because multiple FinQA questions can share the same underlying report.
   This avoids redundant embedding computation without needing to know
   FinQA's internal document-id scheme.

UPDATES (after the Experiment 1 dev-run diagnosis):

6. MODEL: switched from Qwen2.5-1.5B-Instruct (fp16) to
   Qwen2.5-3B-Instruct (4-bit, via bitsandbytes), with automatic fallback
   to the 1.5B model in fp16 if 4-bit loading fails. See
   FinancialRAG._load_llm() for exactly what is loaded and why, and the
   printed log line for which model actually loaded on a given run.

7. FEW-SHOT PROMPTING: generate() now includes two worked examples as
   prior conversation turns, each targeting a specific reasoning error
   observed in the 1.5B baseline (wrong-denominator selection on
   ratio-of-total questions; wrong-period value attribution when evidence
   lists two values "respectively" for two dates). See
   FinancialRAG._FEW_SHOT_EXAMPLES.

8. VERIFIER-GUIDED RETRY: answer() now runs the response through
   framework/verifier.py's PostHocArithmeticVerifier, which independently
   re-executes the model's own stated arithmetic in Python and checks
   that its stated operands appear in the retrieved evidence -- WITHOUT
   ever looking at the gold answer. If verification fails, one
   corrective regeneration is attempted (MAX_VERIFICATION_RETRIES). This
   implements the research design document's Stage 4 ("Multi-layer
   Reasoning and Cross-Verification").

9. RETRIEVAL / MULTI-MATCH ATTEMPT #1 (TRIED, VERIFIED NOT TO WORK,
   REVERTED). After an n=50 dev-run error taxonomy, three examples looked
   like "retrieval grounding failures": AMT/2005/page_105.pdf-2 (correct
   table row not retrieved), SPGI/2018/page_74.pdf-1 (model picked the
   wrong of two similar pension-trust facts), AON/2010/page_28.pdf-1
   (model only read one of two matching table rows). First attempt:
   raised TOP_K 8->12 for AMT, added two system-prompt rules for
   SPGI/AON (scan every matching line; disambiguate near-duplicate
   facts). Re-ran and diffed all 50 examples against the prior run to
   check whether it actually worked, instead of assuming from the
   post-fix summary numbers. It did not: AMT, SPGI, and AON were all
   STILL wrong afterward, with near-identical response text to before --
   the targeted mechanisms had no observable effect on the cases they
   targeted. What the change actually did was perturb the prompt for
   EVERY question (more chunks + longer instructions), which under
   greedy decoding shifted answers on unrelated examples: 8 flipped
   wrong->correct, 6 flipped correct->wrong elsewhere in the set, net
   +2/50 -- noise reshuffling that happened to net positive, not a
   verified fix. Reverted TOP_K to 8, MAX_PROMPT_TOKENS to 4096, and
   removed both prompt rules once this was confirmed, rather than
   keeping ineffective changes that only add prompt-perturbation risk.

10. RETRIEVAL / MULTI-MATCH ATTEMPT #2 (ALSO TRIED, ALSO REVERTED). Root
    cause of AMT/2005/page_105.pdf-2, confirmed by a direct embedding-rank
    measurement (not TOP_K guesswork): the document has only 25 total
    chunks, and the correct "total" table row still ranked outside the
    top 12 -- 18 near-duplicate prose sentences about "net operating loss
    carryforwards" out-ranked the terser, numeric table row on plain
    cosine similarity. That diagnosis is real and still stands. The
    attempted fix -- a FULL_TABLE_MAX_ROWS mechanism giving small tables
    one extra synthetic "table_full" chunk always included regardless of
    rank (see chunk_document/retrieve in an earlier version of this file)
    -- was claimed to be "targeted" (only affecting documents with a
    small table) but that assumption was never checked against how many
    FinQA documents actually have small tables. Re-running n=50 showed
    accuracy DROP to 30% (from the 34% verified baseline), because most
    FinQA tables are small, so this perturbed most of the 50 prompts, not
    just AMT's -- the same failure mode as attempt #1, just netting
    negative this time instead of positive. Reverted. AMT's root cause
    (per-row ranking is the wrong tool for a question needing one
    specific row from a small table) remains real and unfixed as of this
    file version; a genuinely narrow fix (e.g. only forcing inclusion for
    tables at or below ~5-6 rows, checked against the actual row-count
    distribution across the dev set first) is future work, not done here.

Lesson from both attempts, stated explicitly rather than glossed over:
"the model used the wrong number" does not automatically mean "retrieval
failed" -- always check the actual evidence_text before attributing an
error to retrieval vs. generation. And: always re-verify a targeted fix
by diffing individual examples against the prior run, not just by
reading the before/after summary accuracy -- a small net change can hide
a much larger amount of unrelated churn underneath it.

11. ALTERNATIVE ENGINE: llm_backend="anthropic" (see ANTHROPIC_MODEL_NAME
    above and FinancialRAG._generate_anthropic). Added after the local
    3B/4-bit model showed enough run-to-run answer instability (point 9
    above) that it became hard to distinguish a real pipeline fix from
    noise on n=50. Calls the Anthropic API (Claude Opus) for the
    generation step ONLY -- retrieval, chunking, the NRF scoring
    framework, and the gold-free verifier are all identical to the local
    path, so results are directly comparable as a "local small model vs.
    frontier hosted model" reliability comparison, not a different
    experiment. Requires `pip install anthropic` and an ANTHROPIC_API_KEY
    environment variable; never falls back to the local model silently if
    those are missing -- see FinancialRAG._init_anthropic_client().
"""

import os
import hashlib
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required. Install with:\n"
        "    pip install sentence-transformers"
    ) from e

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# Only needed for llm_backend="anthropic" -- imported lazily (not at module
# level as a hard dependency) so the "local" backend keeps working on a
# machine that never installed the anthropic package.
try:
    import anthropic
    _ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    _ANTHROPIC_SDK_AVAILABLE = False

from framework.nrf import NumericalReliabilityFramework
from framework.verifier import PostHocArithmeticVerifier


# ---------------------------------------------------------------------------
# Configuration constants (surfaced here rather than buried in code, so they
# can be tuned or overridden from experiment scripts without editing this file)
# ---------------------------------------------------------------------------

# Primary model: Qwen2.5-3B-Instruct, loaded 4-bit quantized. Chosen after
# the 1.5B baseline showed clear reasoning errors (wrong denominator on
# ratio-of-total questions, wrong-period value attribution on
# percentage-change questions -- see Experiment 1 dev-run diagnosis) and
# Phi-3-mini (3.8B, fp16) proved too slow on a 4GB RTX 3050 (almost
# certainly VRAM-exhausted and falling back to CPU/shared-memory paging).
# 3B in 4-bit (~2GB weights) is sized to comfortably fit alongside the
# embedding model and KV cache on a 4GB card, unlike a 7B model (~4.3GB+
# even 4-bit) which would risk repeating that same slowdown.
#
# FALLBACK_LLM_MODEL_NAME is used automatically, with a clear printed
# warning (never silently), if 4-bit loading fails for any reason
# (bitsandbytes not installed/incompatible, insufficient VRAM, etc.) --
# see FinancialRAG._load_llm().
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LLM_USE_4BIT = True
FALLBACK_LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Alternative generation backend: instead of running a local quantized
# model, call the Anthropic API and use Claude Opus for the generation
# step. Everything else (retrieval, chunking, the NRF scoring framework,
# the gold-free verifier) is IDENTICAL either way -- only generate() takes
# a different path (see FinancialRAG._generate_anthropic). This exists
# because the local 3B/4-bit model showed real answer instability across
# runs (see the n=50 dev-run diffs in Experiment 1 -- ~28% of answers
# changed between two runs that differed only in prompt length, unrelated
# to the specific question), which makes it hard to tell a genuine fix
# from noise. Comparing the SAME pipeline against a frontier hosted model
# is also a legitimate baseline-vs-upper-bound comparison for the
# dissertation, not just a way to get a better score.
# Model ID confirmed current as of this writing via Anthropic's own
# model-ID reference (github.com/anthropics/skills) -- check
# platform.claude.com/docs for the latest alias if this ever 404s.
ANTHROPIC_MODEL_NAME = "claude-opus-4-8"
ANTHROPIC_API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 8                   # number of chunks retrieved per question
# Raised from 5 -> 8 after an earlier dev-run showed two examples where
# the model correctly DECLINED to answer because a needed value wasn't
# among the top-5 retrieved chunks -- a real retrieval-recall gap.
#
# REVERTED an 8 -> 12 change tried in a later round. That attempt was
# made to fix AMT/2005/page_105.pdf-2 (correct table row not in top-8),
# but verification (re-running and diffing all 50 dev examples against
# the prior run) showed it did NOT fix AMT -- a direct embedding-rank
# check confirmed the correct row still wasn't in the top-12 of only 25
# total chunks in that document (18 near-duplicate prose sentences
# outranked it). What TOP_K=12 actually did was change the evidence text
# fed to EVERY question, which under greedy decoding perturbed answers
# on many unrelated examples (8 flipped correct, 6 flipped wrong, net
# +2/50) -- noise reshuffling, not a targeted fix. Reverted to 8 to avoid
# that blind churn. A follow-up attempt at a more surgical fix for AMT
# (a small-table full-inclusion mechanism) was ALSO tried and ALSO
# reverted after it hurt accuracy the same way -- see the module
# docstring, section 10, for the full history. AMT's root cause is real
# and diagnosed; a working fix for it is not yet in this file.
MAX_NEW_TOKENS = 500
# Raised 220 -> 320 -> 500. Two SEPARATE dev-run examples (MAS, a 5-step
# chained percentage-point-difference question, and CMCSA, a
# multi-line-item reconciliation) were still cut off mid-Reasoning at
# 320 tokens, before ever reaching a Final Answer line -- confirmed by
# inspecting the raw response text, not assumed. Genuinely multi-step
# financial questions need more room than a single-subtraction question
# does; 500 gives real headroom without being unbounded.
# Raised from 3500 to give headroom for the two few-shot examples now
# included in every prompt (~300-400 tokens). Qwen2.5's context window
# (32k) comfortably supports this.
# A 4096 -> 4608 bump tried alongside the (reverted) TOP_K=12 attempt is
# reverted back to 4096 too, along with the rest of that experiment.
# truncation_side="left" still protects the question/instructions at the
# tail if this ceiling is ever hit regardless.
MAX_PROMPT_TOKENS = 4096
SEED = 42

# Post-hoc verifier + retry (see framework/verifier.py). At most one
# retry is attempted per question, to bound extra inference cost -- if
# the retry also fails verification, the retry's response is still
# returned (never silently discarded), just flagged as unverified.
VERIFY_AND_RETRY = True
MAX_VERIFICATION_RETRIES = 1

# Embedding model device: kept separate from the LLM device because on a
# 4GB RTX 3050, running both on GPU simultaneously is fine given MiniLM's
# tiny footprint (~90MB), but this is exposed as a constant in case a user
# wants to force the embedder onto CPU to save VRAM headroom.
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_type: str   # "pre_text" | "table_row" | "post_text"


def _format_table_row(row: List[Any], header: Optional[List[Any]]) -> str:
    """Format a single table row for retrieval/reading.

    FinQA tables commonly use the first column as a row label with a
    blank first header cell, e.g.:
        header = ["",            "2015",  "2014",  "2013"]
        row    = ["net revenue", "$5829", "$5735", "$5600"]
    In that case the first cell is treated as a row label rather than a
    value paired with a blank header, producing:
        "net revenue -- 2015: $5829 | 2014: $5735 | 2013: $5600"
    which is far more interpretable in isolation (as a retrieved chunk)
    than pairing it with an empty header string.

    Falls back to bare pipe-joined values if no header row is available or
    the row/header lengths don't line up (which happens in some FinQA
    tables) — silently misaligning columns would be worse than dropping
    the header labels.
    """
    if not header or len(header) != len(row):
        return " | ".join(str(v).strip() for v in row)

    header = [str(h).strip() for h in header]
    row = [str(v).strip() for v in row]

    if header[0] == "" and row[0] != "":
        label = row[0]
        parts = [f"{h}: {v}" for h, v in zip(header[1:], row[1:]) if h]
        return f"{label} -- {' | '.join(parts)}" if parts else label

    parts = [f"{h}: {v}" for h, v in zip(header, row) if h]
    return " | ".join(parts) if parts else " | ".join(row)


# REVERTED: a FULL_TABLE_MAX_ROWS mechanism (guaranteeing a "show the
# whole table" chunk for tables with few rows) was tried here to fix
# AMT/2005/page_105.pdf-2 (its correct "total" row never ranked in the
# top-12 of 25 chunks under embedding similarity). Re-running the n=50
# dev set showed it made accuracy WORSE (30%, down from the 34% verified
# baseline) rather than better, for the same reason the earlier TOP_K=12
# attempt failed: most FinQA tables are small, so "tables with few rows"
# is not the narrow subset it sounds like -- it perturbed the prompt on
# most of the 50 questions, not just AMT, and the churn landed negative
# this time. Reverted rather than kept "just in case it helps sometimes"
# -- an unverified change that measurably hurt accuracy once has no
# business staying in a pipeline whose whole point is producing reliable
# numbers. AMT's specific root cause (per-row ranking is the wrong tool
# for a question needing one specific row from a small table) is real and
# still true; it just needs a more surgical fix than this was, which is
# untried as of this file version -- see rag.py's module docstring,
# section 10, for the full history of both attempts.


def chunk_document(item: Dict[str, Any]) -> List[Chunk]:
    """
    Split a FinQA item's report into retrievable chunks.

    - Each pre_text list element -> one chunk (already sentence-segmented
      by FinQA; not re-split heuristically).
    - Each table row (excluding the header row, which is used as column
      labels rather than indexed itself) -> one chunk.
    - Each post_text list element -> one chunk.
    """
    chunks: List[Chunk] = []

    pre_text = item.get("pre_text", []) or []
    for i, sentence in enumerate(pre_text):
        sentence = str(sentence).strip()
        if sentence:
            chunks.append(Chunk(chunk_id=f"pre_{i}", text=sentence, source_type="pre_text"))

    table = item.get("table", []) or []
    header = table[0] if table else None
    data_rows = table[1:] if len(table) > 1 else []
    for i, row in enumerate(data_rows):
        row_text = _format_table_row(row, header)
        if row_text.strip():
            chunks.append(Chunk(chunk_id=f"table_{i}", text=row_text, source_type="table_row"))

    post_text = item.get("post_text", []) or []
    for i, sentence in enumerate(post_text):
        sentence = str(sentence).strip()
        if sentence:
            chunks.append(Chunk(chunk_id=f"post_{i}", text=sentence, source_type="post_text"))

    return chunks


def _document_key(item: Dict[str, Any]) -> str:
    """Stable hash of a FinQA item's underlying report content, used to
    cache chunk embeddings across questions that share the same report."""
    raw = (
        "".join(str(s) for s in item.get("pre_text", []) or [])
        + "".join(str(s) for s in item.get("post_text", []) or [])
        + "".join(
            "".join(str(c) for c in row)
            for row in (item.get("table", []) or [])
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Vector index (FAISS if available, exact NumPy cosine-similarity fallback
# otherwise). Both paths perform exact nearest-neighbor search over
# L2-normalized vectors, so results are equivalent.
# ---------------------------------------------------------------------------

class _NumpyFlatIndex:
    """Exact cosine-similarity search, used when faiss is not installed."""

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors: Optional[np.ndarray] = None

    def add(self, vectors: np.ndarray) -> None:
        self._vectors = vectors.astype(np.float32)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._vectors is None or len(self._vectors) == 0:
            return np.array([[]]), np.array([[]])
        k = min(k, len(self._vectors))
        scores = self._vectors @ query[0].astype(np.float32)
        top_idx = np.argsort(-scores)[:k]
        top_scores = scores[top_idx]
        return top_scores[np.newaxis, :], top_idx[np.newaxis, :]


class DocumentIndex:
    """Wraps chunk embeddings + a per-document vector index."""

    def __init__(self, chunks: List[Chunk], embeddings: np.ndarray):
        assert len(chunks) == embeddings.shape[0]
        self.chunks = chunks
        self.embeddings = embeddings
        dim = embeddings.shape[1]

        if _FAISS_AVAILABLE:
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings.astype(np.float32))
        else:
            self._index = _NumpyFlatIndex(dim)
            self._index.add(embeddings)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    @classmethod
    def build(cls, chunks: List[Chunk], embedder: "SentenceTransformer") -> "DocumentIndex":
        if not chunks:
            # Degenerate case: no retrievable content at all. Store an
            # empty index rather than raising, so downstream code can
            # handle "no evidence found" explicitly.
            return cls(chunks=[], embeddings=np.zeros((0, embedder.get_sentence_embedding_dimension()), dtype=np.float32))

        texts = [c.text for c in chunks]
        raw = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        normalized = cls._normalize(raw)
        return cls(chunks=chunks, embeddings=normalized)

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        query_normalized = self._normalize(query_embedding.reshape(1, -1))
        scores, indices = self._index.search(query_normalized, min(top_k, len(self.chunks)))
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.chunks[int(idx)], float(score)))
        return results


class _DocumentIndexCache:
    """Small in-memory cache of DocumentIndex objects keyed by document
    content hash, so repeated questions against the same source report
    don't re-embed it. Unbounded for now (dev/full sample sizes here are
    at most low hundreds of documents, which is trivial to hold in RAM;
    revisit if scaling to the full 1147-example test set in one process
    causes memory pressure)."""

    def __init__(self):
        self._cache: Dict[str, DocumentIndex] = {}

    def get_or_build(self, item: Dict[str, Any], embedder: "SentenceTransformer") -> DocumentIndex:
        key = _document_key(item)
        if key not in self._cache:
            chunks = chunk_document(item)
            self._cache[key] = DocumentIndex.build(chunks, embedder)
        return self._cache[key]

    def __len__(self):
        return len(self._cache)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class FinancialRAG:

    def __init__(
        self,
        top_k: int = TOP_K,
        seed: int = SEED,
        verify_and_retry: bool = VERIFY_AND_RETRY,
        max_verification_retries: int = MAX_VERIFICATION_RETRIES,
        llm_backend: str = "local",
    ):
        """llm_backend: "local" (default, unchanged behaviour -- Qwen2.5-3B
        4-bit on your own GPU) or "anthropic" (calls the Anthropic API with
        ANTHROPIC_MODEL_NAME instead of loading any local LLM weights; the
        embedding model is still local either way, since it's tiny and
        retrieval needs to stay identical for a fair comparison). See the
        ANTHROPIC_MODEL_NAME comment above for why this option exists."""
        if llm_backend not in ("local", "anthropic"):
            raise ValueError(f"llm_backend must be 'local' or 'anthropic', got {llm_backend!r}")
        set_seed(seed)
        self.top_k = top_k
        self.verify_and_retry = verify_and_retry
        self.max_verification_retries = max_verification_retries
        self.llm_backend = llm_backend

        print("Loading embedding model:", EMBEDDING_MODEL_NAME)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)

        if self.llm_backend == "local":
            self.llm_model_name = self._load_llm()
            self.model.eval()
            print("LLM running on:", next(self.model.parameters()).device)
        else:
            self.llm_model_name = self._init_anthropic_client()
            print("LLM backend: Anthropic API,", self.llm_model_name)

        print("Embedding model running on:", EMBEDDING_DEVICE)
        print(f"FAISS available: {_FAISS_AVAILABLE} "
              f"({'using faiss.IndexFlatIP' if _FAISS_AVAILABLE else 'using NumPy cosine fallback'})")

        self.nrf = NumericalReliabilityFramework()
        self.verifier = PostHocArithmeticVerifier(nrf=self.nrf)

        self._index_cache = _DocumentIndexCache()

    def _init_anthropic_client(self) -> str:
        """Set up the Anthropic API client for llm_backend="anthropic".
        Never falls back silently to the local model -- if the SDK isn't
        installed or the API key isn't set, this raises immediately with
        an actionable message, rather than quietly running a different
        experiment than the one you asked for."""
        if not _ANTHROPIC_SDK_AVAILABLE:
            raise ImportError(
                "llm_backend='anthropic' requires the anthropic package. Install with:\n"
                "    pip install anthropic"
            )
        api_key = os.environ.get(ANTHROPIC_API_KEY_ENV_VAR)
        if not api_key:
            raise RuntimeError(
                f"llm_backend='anthropic' requires the {ANTHROPIC_API_KEY_ENV_VAR} "
                f"environment variable to be set to a valid Anthropic API key.\n"
                f"On Windows (PowerShell): $env:{ANTHROPIC_API_KEY_ENV_VAR} = \"your-key-here\"\n"
                f"Then re-run in the SAME terminal session."
            )
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        return ANTHROPIC_MODEL_NAME

    def _load_llm(self) -> str:
        """Load the primary (4-bit quantized 3B) model, falling back to
        FALLBACK_LLM_MODEL_NAME in fp16 if that fails for any reason. The
        fallback is never silent: it always prints which model actually
        ended up loaded and why, so a run's console output/log is a
        truthful record of what was actually used -- important since this
        directly affects Experiment 1's numbers."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME if LLM_USE_4BIT else FALLBACK_LLM_MODEL_NAME
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # If the prompt (few-shot examples + evidence + question) ever
        # exceeds MAX_PROMPT_TOKENS, truncate from the LEFT (the earliest,
        # least-critical content -- system prompt / few-shot examples)
        # rather than the default right-truncation, which would silently
        # cut off the actual question and output-format instructions at
        # the end of the prompt.
        self.tokenizer.truncation_side = "left"

        if LLM_USE_4BIT and LLM_DEVICE == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                print(f"Loading {LLM_MODEL_NAME} in 4-bit (NF4)...")
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    LLM_MODEL_NAME,
                    quantization_config=quant_config,
                    device_map=LLM_DEVICE,
                )
                print(f"Loaded {LLM_MODEL_NAME} (4-bit).")
                return LLM_MODEL_NAME
            except Exception as e:
                print(
                    f"WARNING: 4-bit load of {LLM_MODEL_NAME} failed ({type(e).__name__}: {e}). "
                    f"Falling back to {FALLBACK_LLM_MODEL_NAME} (fp16). "
                    f"If this is a missing dependency, try: pip install bitsandbytes accelerate"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_LLM_MODEL_NAME)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.truncation_side = "left"

        model_name = FALLBACK_LLM_MODEL_NAME if (LLM_USE_4BIT and LLM_DEVICE == "cuda") else LLM_MODEL_NAME
        if LLM_DEVICE != "cuda":
            print(f"WARNING: No CUDA device detected; loading {model_name} on CPU. This will be slow.")
        print(f"Loading {model_name} (fp16)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map=LLM_DEVICE,
        )
        print(f"Loaded {model_name} (fp16).")
        return model_name

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, item: Dict[str, Any], question: str) -> List[Tuple[Chunk, float]]:
        index = self._index_cache.get_or_build(item, self.embedder)
        if len(index.chunks) == 0:
            return []
        query_embedding = self.embedder.encode([question], convert_to_numpy=True, show_progress_bar=False)
        results = index.search(query_embedding, top_k=self.top_k)

        return results

    @staticmethod
    def build_evidence_text(retrieved: List[Tuple[Chunk, float]]) -> str:
        """Render retrieved chunks as the evidence block shown to the LLM.
        Ordered by retrieval rank (most relevant first) — this is a
        deliberate choice over restoring original document order, since
        the goal is to put the most useful evidence earliest in a short
        context window."""
        if not retrieved:
            return "(No relevant evidence was retrieved for this question.)"
        lines = []
        for chunk, score in retrieved:
            lines.append(f"- [{chunk.source_type}] {chunk.text}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    # Two few-shot examples, added as prior conversation turns rather than
    # buried in the system prompt (more reliably followed by small
    # instruct models). Each targets a SPECIFIC error class observed in
    # the 1.5B baseline's dev-run results, not a generic "be careful"
    # nudge:
    #
    #  - Example A targets picking the WRONG denominator on a
    #    ratio-of-a-subset-of-total question (the baseline divided by
    #    "owned facilities" instead of "total facilities" for a
    #    "% of total facilities leased" question).
    #  - Example B targets misattributing which value belongs to which
    #    period when evidence lists two values "respectively" for two
    #    dates (the baseline swapped the 2011/2010 values).
    #
    # These are real, generic reasoning patterns in financial numerical
    # QA (denominator selection; matching values to labels in a
    # "X and Y, respectively" construction), not answers memorized for
    # any specific FinQA test example -- the numbers here do not appear
    # in the FinQA test set.
    _FEW_SHOT_EXAMPLES = [
        {
            "evidence": (
                "- [table_row] leased facilities -- united states: 2.1 | other countries: 6.0 | total: 8.1\n"
                "- [table_row] owned facilities -- united states: 30.7 | other countries: 17.2 | total: 47.9\n"
                "- [table_row] total facilities -- united states: 32.8 | other countries: 23.2 | total: 56.0"
            ),
            "question": "what percentage of total facilities (in square feet) are leased?",
            "answer": (
                "Reasoning: The question asks for leased facilities as a percentage of TOTAL facilities, "
                "not owned facilities -- I need the 'total facilities' row as the denominator, not the "
                "'owned facilities' row. Leased total = 8.1. Total facilities total = 56.0. "
                "8.1 / 56.0 * 100 = 14.46\n\n"
                "Final Answer: 14.46%\n\n"
                "Confidence: 0.9"
            ),
        },
        {
            "evidence": (
                "- [pre_text] the total notional amounts of cash flow hedges as of october 29 , 2011 and "
                "october 30 , 2010 were $153.7 million and $139.9 million , respectively ."
            ),
            "question": "what was the percentage change in cash flow hedges from 2010 to 2011?",
            "answer": (
                "Reasoning: The evidence lists two dates then two values 'respectively' -- the FIRST value "
                "matches the FIRST date and the SECOND value matches the SECOND date. So 2011 = $153.7 "
                "million (matches 'october 29, 2011', listed first) and 2010 = $139.9 million (matches "
                "'october 30, 2010', listed second). I must not assume the more recent year is the larger "
                "or smaller number -- I read it directly from the pairing. "
                "Change = (153.7 - 139.9) / 139.9 * 100 = 9.86\n\n"
                "Final Answer: 9.86%\n\n"
                "Confidence: 0.9"
            ),
        },
    ]

    _SYSTEM_PROMPT = (
        "You are a financial numerical reasoning assistant. "
        "Use ONLY the retrieved evidence below. Do not invent numbers that "
        "are not present in or directly derivable from the evidence. "
        "Before calculating: (1) identify exactly which row/line the "
        "question refers to -- do not substitute a similarly-named row "
        "(e.g. do not use 'owned' when the question asks about 'total'); "
        "(2) if the evidence lists two values for two periods/entities "
        "using a construction like 'X and Y, respectively', match each "
        "value to its correct period in the SAME order they are listed, "
        "do not guess based on which number seems larger or more recent. "
        "If the question asks for a percentage, a rate, or a share of a "
        "total, your Final Answer MUST include a '%' sign (e.g. 'Final "
        "Answer: 9.86%', not 'Final Answer: 9.86') -- omitting it when a "
        "percentage was asked for is treated as a wrong unit, not a close "
        "answer. If the evidence genuinely does not contain a value you "
        "need, say so and give your best-supported answer with a low "
        "Confidence rather than a high-confidence guess. "
        "Put your full calculation in the Reasoning section, ending it "
        "with an explicit 'expression = result' (e.g. '8.1 / 56.0 * 100 "
        "= 14.46'), so the result can be checked. The Final Answer line "
        "must then contain ONLY that final value and its unit (e.g. "
        "'Final Answer: 14.46%') -- do not repeat the full expression "
        "there."
    )

    def _build_messages(self, question: str, evidence_text: str, correction_note: Optional[str] = None) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [{"role": "system", "content": self._SYSTEM_PROMPT}]

        for example in self._FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": f"Retrieved Financial Evidence:\n\n{example['evidence']}\n\nQuestion:\n\n{example['question']}\n\n"
                           "Return your answer in EXACTLY this format:\n\n"
                           "Reasoning: [brief calculation using only the retrieved evidence]\n\n"
                           "Final Answer: [one numerical answer]\n\n"
                           "Confidence: [number between 0 and 1]",
            })
            messages.append({"role": "assistant", "content": example["answer"]})

        user_content = f"""
Retrieved Financial Evidence:

{evidence_text}

Question:

{question}

Return your answer in EXACTLY this format:

Reasoning: [brief calculation using only the retrieved evidence]

Final Answer: [one numerical answer]

Confidence: [number between 0 and 1]

Do not stop before providing Final Answer and Confidence.
"""
        if correction_note:
            user_content += f"""

NOTE ON YOUR PREVIOUS ATTEMPT: An automated check could not verify your previous
answer to this exact question -- {correction_note}
Please re-identify which specific values from the evidence apply to this question
and recompute carefully before answering again.
"""
        messages.append({"role": "user", "content": user_content})
        return messages

    def generate(self, question: str, evidence_text: str, correction_note: Optional[str] = None) -> str:
        messages = self._build_messages(question, evidence_text, correction_note=correction_note)

        if self.llm_backend == "anthropic":
            return self._generate_anthropic(messages)
        return self._generate_local(messages)

    def _generate_local(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        ).to(LLM_DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response

    def _generate_anthropic(self, messages: List[Dict[str, str]]) -> str:
        """Same _build_messages() output as the local path, just sent to
        the Anthropic API instead of a local model. The Anthropic Messages
        API takes the system prompt as a separate top-level `system`
        argument rather than as a "system"-role message in the messages
        list, so that first message is pulled out here; everything else
        (few-shot user/assistant turns, the real question, the optional
        retry correction note) is passed through unchanged -- the model
        sees an identical conversation either way, so this is a fair
        like-for-like comparison against the local backend.
        do_sample=False locally maps to temperature=0 here for the same
        reason: minimize run-to-run variance from decoding randomness so
        that any accuracy difference reflects the model, not sampling luck.
        Uses greedy generation with a modest retry on transient API errors
        (rate limits, overloaded) rather than failing an entire experiment
        run over one flaky request."""
        system_prompt = messages[0]["content"]
        conversation = messages[1:]

        last_error = None
        for attempt in range(3):
            try:
                response = self.anthropic_client.messages.create(
                    model=self.llm_model_name,
                    max_tokens=MAX_NEW_TOKENS,
                    temperature=0,
                    system=system_prompt,
                    messages=conversation,
                )
                return "".join(
                    block.text for block in response.content if getattr(block, "type", None) == "text"
                )
            except Exception as e:  # noqa: BLE001 -- deliberately broad: any
                # API/network failure should retry a few times, then
                # surface clearly rather than being swallowed.
                last_error = e
                print(f"WARNING: Anthropic API call failed (attempt {attempt + 1}/3): "
                      f"{type(e).__name__}: {e}")
        raise RuntimeError(
            f"Anthropic API call failed after 3 attempts: {type(last_error).__name__}: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------

    def answer(self, item: Dict[str, Any], extra_evidence: Optional[str] = None,
               evidence_override: Optional[str] = None) -> Dict[str, Any]:
        """
        extra_evidence: text appended to the retrieved evidence block AFTER
        retrieval, i.e. content the model is forced to see regardless of
        whether it would have been retrieved.

        This exists for adversarial-robustness experiments (experiment2 /
        experiment4) and its purpose is worth stating precisely, because the
        distinction it enables is easy to lose:

          - Injecting distractors into the DOCUMENT (item["pre_text"])
            tests the pipeline END-TO-END, and in practice mostly tests
            retrieval, since retrieval may simply never surface them.
            Measured on the n=10 dev set: 0 of 10 naive distractor
            sentences were retrieved, and 8 of 10 model responses were
            byte-identical to the clean condition -- the noise never
            reached the model at all.

          - Injecting distractors HERE, into the evidence block, bypasses
            retrieval and tests the DETECTOR, which is what experiment2's
            research question actually asks about. A hallucination-detection
            module cannot be validated on inputs that contain nothing to
            detect.

        Both are legitimate and they answer different questions; do not
        report one as if it were the other. `extra_evidence` is recorded in
        the returned dict so a run's evidence block is always auditable.
        """
        question = item["qa"]["question"]

        # evidence_override REPLACES the retrieved evidence block entirely,
        # rather than appending to it as extra_evidence does. This exists for
        # the comparative-baseline experiment (experiment5), which needs to
        # hold the generator, prompt, scoring and verifier fixed while varying
        # ONLY what evidence reaches the model:
        #   - "" (empty)      -> closed-book condition, no evidence at all
        #   - gold evidence   -> oracle-retrieval condition, mirroring the
        #                        gold-retrieval setting reported in the FinQA
        #                        literature so the comparison is like-for-like
        # Retrieval still runs and its output is still returned, so the
        # retrieved_chunks field remains auditable even when it was not used.
        retrieved = self.retrieve(item, question)
        if evidence_override is not None:
            evidence_text = evidence_override
        else:
            evidence_text = self.build_evidence_text(retrieved)
        if extra_evidence:
            evidence_text = evidence_text + "\n" + extra_evidence

        response = self.generate(question, evidence_text)

        verification = None
        retry_count = 0

        # Verifier-guided retry (research design doc, Stage 4 --
        # "Multi-layer Reasoning and Cross-Verification"). The verifier
        # NEVER sees the gold answer (see framework/verifier.py) -- it
        # only checks that the model's own stated operands are grounded
        # in the retrieved evidence and that Python's own recomputation
        # of the model's own stated arithmetic matches what it reported.
        # This is a legitimate, gold-free improvement step, not a way of
        # nudging the model toward a known correct answer.
        if self.verify_and_retry:
            verification = self.verifier.verify(response, evidence_text)
            while (not verification.verified) and retry_count < self.max_verification_retries:
                response = self.generate(question, evidence_text, correction_note=verification.reason)
                verification = self.verifier.verify(response, evidence_text)
                retry_count += 1

        return {
            "id": item["id"],
            "question": question,
            "retrieved_chunks": [
                {"chunk_id": c.chunk_id, "source_type": c.source_type, "text": c.text, "score": score}
                for c, score in retrieved
            ],
            "evidence_text": evidence_text,
            "extra_evidence": extra_evidence,
            "response": response,
            "verification": verification.as_dict() if verification is not None else None,
            "retry_count": retry_count,
            "gold_answer": item["qa"]["answer"],
            "gold_program": item["qa"].get("program"),
            "gold_evidence": item["qa"].get("gold_inds", {}),
        }