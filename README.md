# Gemini Research Paper Repo

This repo packages the core evidence from the volatility-regime trend-following study so Gemini can draft a full paper with traceable sources.

## What is included
- `manuscript/main.tex`: arXiv-style manuscript source.
- `manuscript/references.bib`: bibliography entries.
- `figures/*.png`: publication-ready charts.
- `data/tables/*.csv`: all key result tables.
- `data/time_series/*.csv`: daily backtest series for SPY, QQQ, IWM.
- `metadata/summary_metrics.json`: headline metrics and setup.
- `metadata/figure_index.csv`: figure-to-caption mapping.
- `metadata/data_dictionary.md`: schema documentation.
- `GEMINI_PROMPT.md`: prompt template for Gemini.

## Recommended Gemini workflow
1. Load `GEMINI_PROMPT.md` as the base instruction.
2. Attach `metadata/summary_metrics.json` and all CSV files in `data/tables/`.
3. Attach figures from `figures/` and require citation by filename.
4. Use `data/time_series/` only for additional charting or sensitivity checks.
5. Output a draft in arXiv format aligned with `manuscript/main.tex` sections.

## Rebuild assets
Run:
```bash
python scripts/build_paper_assets.py
```

## Notes
- Regime classification uses out-of-sample expanding quantiles.
- Backtest assumptions: monthly rebalance, 10 bps transaction cost.
- If market data updates, rerun the build script before drafting.
