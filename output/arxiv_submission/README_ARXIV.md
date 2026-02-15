# arXiv Submission Package

This directory contains a source-ready arXiv package.

## Included in the clean tarball
- `main.tex` (primary manuscript)
- `main.bbl` (resolved bibliography)
- `references.bib` (source bibliography)
- `figures/fig_*.png` (all referenced figures)
- `README_ARXIV.md`
- `ARXIV_METADATA.md`

## arXiv compiler settings
- Compiler: `pdfLaTeX`
- Main file: `main.tex`

## Local compile command
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
