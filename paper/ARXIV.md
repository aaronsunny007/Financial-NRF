# arXiv submission notes

**Package:** `arxiv.zip` (built and test-compiled by `.github/workflows/paper.yml`, published on the `paper-pdf` branch
together with `main.pdf`). It contains the sources, the unmodified ACL style files (`acl.sty`, `acl_natbib.bst`), the figures,
the generated tables and a pre-built `main.bbl`, and compiles with plain `pdflatex`.
Regenerate locally with `bash paper/make_arxiv.sh` after building `main.pdf` once.

**Format:** ACL template in `preprint` mode (non-anonymous, page numbers). For a venue submission change
`\usepackage[preprint]{acl}` in `main.tex` to `review` (anonymized) and remove the author block and acknowledgements.

## Metadata to paste into the arXiv form

- **Title:** Reference-Free Reliability Scores Cannot See Corrupted Evidence: A Controlled Test for Numerical Retrieval-Augmented QA in Finance
- **Authors:** Aaron Prakash Gandi
- **Primary category:** cs.CL (Computation and Language). **Cross-list:** cs.IR, cs.LG.
- **Comments:** 10 pages, 2 figures, 4 tables. Code and per-response outputs: https://github.com/aaronsunny007/Financial-NRF
- **License:** CC BY 4.0 is the usual choice for a paper whose code is already public; the arXiv non-exclusive licence is the
  minimum. Choose before submitting, because it cannot be withdrawn.
- **Abstract:** copy from `main.tex` (replace the `\aucFinoverlap{}`-style macros by their values in `generated/numbers.tex`).

## Before you submit

1. **Endorsement.** A first submission to cs.* usually needs an endorsement from an established author in the category.
   arXiv shows the endorsement request link during submission; you can also be endorsed automatically if you have an
   institutional email with prior arXiv-indexed papers.
2. **Verify the bibliography** (`references.bib` was written from memory) against the publisher records; arXiv does not check it.
3. **AI-assistance disclosure.** The Acknowledgements say the experiments, code and a first draft were produced with Claude
   (Anthropic) and that you reviewed and take responsibility. arXiv and ACL policies hold the human author responsible and
   expect AI assistance to be disclosed rather than listed as an author. Keep, edit or extend that sentence as you see fit; do
   not remove it if it is true.
4. **Read the whole paper once as the author**; check every claim you are not comfortable defending.
5. After the PDF is on arXiv you can tag the repository (`git tag v1.0-paper`) and archive a release on Zenodo for a DOI.
