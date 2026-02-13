"""
Report generation utilities (rules-based).
"""

from typing import Dict
import pandas as pd


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    """
    Render a simple markdown table without external deps.
    """
    if df is None or df.empty:
        return "_No data available._"

    head = df.head(max_rows).copy()
    head = head.reset_index()
    headers = list(head.columns)
    rows = head.values.tolist()

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    md = []
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(md)


def build_report_payload(summary: Dict[str, str], tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """
    Build Markdown and HTML report payloads.

    Args:
        summary: Dict of summary fields.
        tables: Dict of table name -> DataFrame.

    Returns:
        Dict with 'markdown' and 'html' keys.
    """
    md_lines = ["# HedgeFund Dashboard Report", ""]

    md_lines.append("## Summary")
    for k, v in summary.items():
        md_lines.append(f"- **{k}**: {v}")

    for name, df in tables.items():
        md_lines.append("")
        md_lines.append(f"## {name}")
        md_lines.append(_df_to_markdown(df, max_rows=10))

    markdown = "\n".join(md_lines)

    # Simple HTML version
    html_lines = ["<html><body>", "<h1>HedgeFund Dashboard Report</h1>"]
    html_lines.append("<h2>Summary</h2>")
    html_lines.append("<ul>")
    for k, v in summary.items():
        html_lines.append(f"<li><strong>{k}</strong>: {v}</li>")
    html_lines.append("</ul>")

    for name, df in tables.items():
        html_lines.append(f"<h2>{name}</h2>")
        if df is None or df.empty:
            html_lines.append("<p><em>No data available.</em></p>")
        else:
            html_lines.append(df.head(10).to_html(index=True))

    html_lines.append("</body></html>")
    html = "\n".join(html_lines)

    return {"markdown": markdown, "html": html}
