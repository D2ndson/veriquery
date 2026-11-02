"""Part F: deterministic, read-only pandas analysis of SQL/CSV results."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
import pandas as pd
import plotly.express as px

@dataclass
class AnalysisResult:
    question: str
    answer: str
    table: pd.DataFrame
    figure: object | None = None
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    operation: str = ""


def analyze(df: pd.DataFrame, question: str) -> AnalysisResult:
    if df.empty:
        return AnalysisResult(question, "No rows available for analysis.", df.copy(), warnings=["Empty input dataset."])
    q = question.lower().strip()
    cols = list(df.columns)
    # Match whole column names, preferring longest names to avoid substrings.
    mentioned = [c for c in sorted(cols, key=lambda x: -len(str(x))) if re.search(r'(?<!\w)' + re.escape(str(c).lower()) + r'(?!\w)', q)]
    numeric = list(df.select_dtypes(include='number').columns)
    categorical = [c for c in cols if c not in numeric]
    metric = next((c for c in mentioned if c in numeric), None)
    group = next((c for c in mentioned if c != metric and c in categorical), None)
    assumptions = []
    if not metric and numeric:
        metric = numeric[0]
        assumptions.append(f"Metric not explicitly matched; selected '{metric}'.")
    if not group and categorical and any(x in q for x in ('by ', 'per ', 'each ', 'group')):
        group = categorical[0]
        assumptions.append(f"Grouping column not explicitly matched; selected '{group}'.")
    if any(x in q for x in ('correlation', 'correlate')):
        if len(numeric) < 2:
            return AnalysisResult(question, "Correlation requires at least two numeric columns.", df.head(0), warnings=["Insufficient numeric columns."])
        result = df[numeric].corr(numeric_only=True)
        fig = px.imshow(result, text_auto='.2f', title='Correlation matrix')
        return AnalysisResult(question, 'Pearson correlation matrix calculated.', result.reset_index(), fig, assumptions, operation='correlation')
    if any(x in q for x in ('count', 'how many', 'number of rows')):
        op = 'count'
    elif any(x in q for x in ('average', 'mean', 'avg')):
        op = 'mean'
    elif any(x in q for x in ('total', 'sum')):
        op = 'sum'
    elif any(x in q for x in ('maximum', 'highest', 'max', 'top')):
        op = 'max'
    elif any(x in q for x in ('minimum', 'lowest', 'min')):
        op = 'min'
    elif any(x in q for x in ('median',)):
        op = 'median'
    else:
        return AnalysisResult(question, 'Question not recognized. Try: total sales by region, average revenue, count by category, highest price, or correlation.', df.head(10), warnings=['Unsupported question; no statistic inferred.'])
    if op != 'count' and metric is None:
        return AnalysisResult(question, 'A numeric column is required for this calculation.', df.head(0), warnings=['No numeric columns found.'])
    if group:
        grouped = df.groupby(group, dropna=False)
        series = grouped.size() if op == 'count' else getattr(grouped[metric], op)()
        out = series.reset_index(name=f'{op}_{metric or "rows"}')
        if op in ('max', 'sum', 'count'):
            out = out.sort_values(out.columns[-1], ascending=False)
        fig = px.bar(out.head(30), x=group, y=out.columns[-1], title=f'{op.title()} by {group}')
        answer = f'{op.title()} calculated for {len(out)} groups using {len(df)} rows.'
    else:
        value = len(df) if op == 'count' else getattr(df[metric], op)()
        out = pd.DataFrame({'statistic': [f'{op}_{metric or "rows"}'], 'value': [value]})
        fig = None
        answer = f'{op.title()} of {metric or "rows"}: {value:,.4f}' if isinstance(value, (int, float)) else f'{op.title()}: {value}'
    return AnalysisResult(question, answer, out, fig, assumptions, operation=op)
