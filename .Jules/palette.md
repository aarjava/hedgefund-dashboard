
## 2024-03-15 - [Added concrete examples to statistical and financial parameter tooltips]
**Learning:** Found that users and even screen-readers encounter ambiguous parameter thresholds (like "High Volatility Quantile" or "Transaction Cost (bps)"). Adding concrete examples (e.g., "0.80 = top 20%" or "10 bps = 0.10%") improves clarity and provides an immediate mental anchor for complex inputs, making the dashboard more accessible for users with different levels of financial expertise.
**Action:** When adding or modifying inputs related to quantitative finance or statistical variables, always include a `help` tooltip with a concrete, numerical example of what a common value represents in plain language.
