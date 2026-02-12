# Prompt for Gemini: Draft the Research Paper

Use only the attached repository artifacts to draft a research paper titled:

"Volatility Regimes and Trend-Following Performance in U.S. Equities"

## Hard requirements
- Use evidence only from the provided files.
- Cite every numeric claim with a file reference (example: `data/tables/unconditional_performance.csv`).
- Do not invent additional experiments.
- Keep methodology consistent with `metadata/summary_metrics.json`.
- Use all seven figures and cite by filename from `figures/`.

## Structure requirements
- Abstract
- Introduction
- Literature context
- Data and methodology
- Results
- Robustness checks
- Limitations
- Conclusion
- Appendix with table notes and figure notes

## Evidence map
- Headline setup and key metrics: `metadata/summary_metrics.json`
- Main performance tables: `data/tables/*.csv`
- Figure captions: `metadata/figure_index.csv`
- Time-series diagnostics: `data/time_series/*.csv`
- Existing manuscript baseline: `manuscript/main.tex`

## Style requirements
- Academic tone.
- Quantify all findings.
- Distinguish statistically supported findings from descriptive observations.
- Explicitly report p-values where available.

## Output format
- Return a complete LaTeX manuscript (`main.tex` compatible).
- Include a short "Data provenance" paragraph referencing this repo.
