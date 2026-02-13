# arXiv Submission Package

This folder contains an arXiv-ready LaTeX source package for the report.

## Contents
- `main.tex`: Manuscript source.
- `references.bib`: Bibliography.
- `figures/*.png`: All figures referenced by `main.tex`.

## Compile locally (if LaTeX is installed)
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Upload to arXiv
1. Upload all files in this folder (or upload `arxiv_submission.tar.gz`).
2. Ensure compiler is set to `pdfLaTeX`.
3. Main document should be `main.tex`.

## Notes
- `\pdfoutput=1` is included for arXiv compatibility.
- Figure paths are relative to this folder.
