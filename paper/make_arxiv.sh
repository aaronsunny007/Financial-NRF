#!/usr/bin/env bash
# Builds paper/arxiv.zip: everything arXiv needs to compile the paper (sources, ACL style files, figures, the
# generated tables and a pre-built main.bbl), then test-compiles the package in a clean directory.
# Usage (from the repository root, after paper/main.pdf has been built once so main.bbl exists):
#     bash paper/make_arxiv.sh
set -euo pipefail
cd "$(dirname "$0")"
[ -f main.bbl ] || { echo "main.bbl missing: build main.tex with latexmk first"; exit 1; }
rm -rf /tmp/arxiv_pkg arxiv.zip
mkdir -p /tmp/arxiv_pkg/figures /tmp/arxiv_pkg/generated
cp main.tex sections_front.tex sections_results.tex sections_back.tex references.bib acl.sty acl_natbib.bst main.bbl /tmp/arxiv_pkg/
cp figures/*.pdf /tmp/arxiv_pkg/figures/
cp generated/*.tex /tmp/arxiv_pkg/generated/
( cd /tmp/arxiv_pkg && zip -qr "$OLDPWD/arxiv.zip" . )
# test compile exactly as arXiv would: pdflatex only, using the supplied .bbl
rm -rf /tmp/arxiv_test && mkdir /tmp/arxiv_test && unzip -q arxiv.zip -d /tmp/arxiv_test
( cd /tmp/arxiv_test && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex | tail -3 )
grep -c "undefined" /tmp/arxiv_test/main.log || true
echo "arxiv.zip ok: $(unzip -l arxiv.zip | tail -1)"
