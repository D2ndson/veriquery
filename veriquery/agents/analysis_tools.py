"""Analyst tool library: answers "...and why?" instead of only "what?".

contribution_analysis   which dimension values drove the change in a metric between two periods
pvm_decomposition       price / volume / mix effects of a revenue change (exact identity, tested)
detect_anomalies        robust (median / MAD) outlier detection on a series
forecast_series         ETS forecast with 80% prediction intervals (linear fallback for short series)
auto_analysis           picks the tool from the question and the shape of the SQL result

All functions are deterministic pandas / numpy / statsmodels. None of them calls an LLM, so every number
they report can be reproduced from the result table.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from veriquery.agents.python_analyst import AnalysisResult

_WHY = re.compile(r"\b(why|driver|drivers|drove|driven|driving|drives|reason|cause|caused|explain|what changed|what happened|contribut\w*|due to|behind)\b", re.I)
_CHANGE = re.compile(r"\b(drop\w*|declin\w*|decreas\w*|increas\w*|grew|growth|fell|rose|lost|gain\w*|change\w*|chang)\b", re.I)
_FORECAST = re.compile(r"\b(forecast|predict\w*|next (month|quarter|year|week)|projection|project)\b", re.I)
_ANOMALY = re.compile(r"\b(anomal\w*|outlier\w*|spike\w*|unusual|abnormal)\b", re.I)
_PERIOD_NAME = re.compile(r"(date|month|year|quarter|period|week|day|time|ym)", re.I)


def wants_decomposition(question: str) -> bool:
    return bool(_WHY.search(question))


# --------------------------------------------------------------------------- contribution analysis
def contribution_analysis(df: pd.DataFrame, period_col: str, dim_col: str, value_col: str,
                          base=None, current=None) -> dict:
    """Split the change in sum(value) between two periods across the values of dim_col."""
    periods = sorted(df[period_col].dropna().unique(), key=lambda x: str(x))
    if len(periods) < 2:
        raise ValueError("need at least two periods")
    base = periods[-2] if base is None else base
    current = periods[-1] if current is None else current
    pv = df.pivot_table(index=dim_col, columns=period_col, values=value_col, aggfunc="sum", fill_value=0.0)
    for p in (base, current):
        if p not in pv.columns:
            raise ValueError(f"period {p!r} not in data")
    out = pd.DataFrame({dim_col: pv.index, "base": pv[base].values, "current": pv[current].values})
    out["delta"] = out["current"] - out["base"]
    total = float(out["delta"].sum())
    out["share_of_change"] = out["delta"] / total if total else np.nan
    out = out.reindex(out["delta"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    return {"table": out, "base": base, "current": current, "total_delta": total,
            "base_total": float(out["base"].sum()), "current_total": float(out["current"].sum())}


def _waterfall(res: dict, dim_col: str, value_col: str, top: int = 8) -> go.Figure:
    t = res["table"].head(top)
    rest = res["table"].iloc[top:]["delta"].sum()
    labels = [str(res["base"])] + [str(x) for x in t[dim_col]] + (["other"] if len(res["table"]) > top else []) + [str(res["current"])]
    vals = [res["base_total"]] + list(t["delta"]) + ([rest] if len(res["table"]) > top else []) + [res["current_total"]]
    measure = ["absolute"] + ["relative"] * (len(labels) - 2) + ["total"]
    fig = go.Figure(go.Waterfall(x=labels, y=vals, measure=measure))
    fig.update_layout(title=f"What drove the change in {value_col}: {res['base']} to {res['current']}", showlegend=False)
    return fig


# --------------------------------------------------------------------------- price / volume / mix
def pvm_decomposition(df: pd.DataFrame, dim_col: str, period_col: str, qty_col: str, revenue_col: str,
                      base=None, current=None) -> dict:
    """revenue change = price effect + volume effect + mix effect (+ new / lost items), exactly.

    For an item present in both periods with price P = revenue / quantity and total quantity Q:
        price  = (P1 - P0) * q1
        volume = (Q1 - Q0) * s0 * P0           where s = q / Q is the item's quantity share
        mix    = (s1 - s0) * Q1 * P0
    and price + volume + mix = r1 - r0 identically. Items sold in only one period go to `new_lost`."""
    periods = sorted(df[period_col].dropna().unique(), key=lambda x: str(x))
    if len(periods) < 2:
        raise ValueError("need at least two periods")
    base = periods[-2] if base is None else base
    current = periods[-1] if current is None else current
    agg = df.groupby([dim_col, period_col])[[qty_col, revenue_col]].sum()
    q0 = agg.xs(base, level=period_col)[qty_col] if base in agg.index.get_level_values(1) else pd.Series(dtype=float)
    q1 = agg.xs(current, level=period_col)[qty_col]
    r0 = agg.xs(base, level=period_col)[revenue_col]
    r1 = agg.xs(current, level=period_col)[revenue_col]
    items = sorted(set(q0.index) | set(q1.index), key=str)
    Q0, Q1 = float(q0.sum()), float(q1.sum())
    rows = []
    for it in items:
        a0, a1 = float(q0.get(it, 0.0)), float(q1.get(it, 0.0))
        v0, v1 = float(r0.get(it, 0.0)), float(r1.get(it, 0.0))
        row = {dim_col: it, "revenue_base": v0, "revenue_current": v1, "delta": v1 - v0,
               "price": 0.0, "volume": 0.0, "mix": 0.0, "new_lost": 0.0}
        if a0 > 0 and a1 > 0:
            p0, p1 = v0 / a0, v1 / a1
            s0, s1 = a0 / Q0, a1 / Q1
            row["price"] = (p1 - p0) * a1
            row["volume"] = (Q1 - Q0) * s0 * p0
            row["mix"] = (s1 - s0) * Q1 * p0
        else:
            row["new_lost"] = v1 - v0
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("delta", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    totals = {k: float(table[k].sum()) for k in ("price", "volume", "mix", "new_lost", "delta")}
    return {"table": table, "totals": totals, "base": base, "current": current}


# --------------------------------------------------------------------------- anomalies & forecast
def detect_anomalies(series: pd.Series, z: float = 3.5) -> pd.DataFrame:
    x = series.astype(float)
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0:
        return pd.DataFrame(columns=["index", "value", "robust_z"])
    rz = 0.6745 * (x - med) / mad
    hit = rz.abs() >= z
    return pd.DataFrame({"index": series.index[hit].tolist(), "value": x[hit].tolist(), "robust_z": rz[hit].round(2).tolist()})


def forecast_series(series: pd.Series, horizon: int = 3, alpha: float = 0.2) -> pd.DataFrame:
    """Forecast with prediction intervals (central 1 - alpha)."""
    y = series.astype(float).reset_index(drop=True)
    n = len(y)
    if n < 4:
        raise ValueError("need at least 4 points to forecast")
    try:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        fit = ETSModel(y, error="add", trend="add", damped_trend=True).fit(disp=False)
        pred = fit.get_prediction(start=n, end=n + horizon - 1).summary_frame(alpha=alpha)
        return pd.DataFrame({"step": range(1, horizon + 1), "forecast": pred["mean"].values,
                             "lower": pred["pi_lower"].values, "upper": pred["pi_upper"].values, "method": "ETS (damped trend)"})
    except Exception:
        x = np.arange(n)
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        sd = float(resid.std(ddof=2)) if n > 2 else 0.0
        zq = 1.2816  # 80% two-sided
        fx = np.arange(n, n + horizon)
        mean = slope * fx + intercept
        return pd.DataFrame({"step": range(1, horizon + 1), "forecast": mean, "lower": mean - zq * sd,
                             "upper": mean + zq * sd, "method": "linear trend"})


# --------------------------------------------------------------------------- tool selection
def _id_like(name) -> bool:
    n = str(name).lower()
    return n == "id" or n.endswith("_id") or n.endswith(" id")


def _shape(df: pd.DataFrame) -> dict:
    numeric = [c for c in df.select_dtypes(include="number").columns if not _id_like(c)]
    others = [c for c in df.columns if c not in numeric]
    period = next((c for c in df.columns if _PERIOD_NAME.search(str(c)) and df[c].nunique() >= 2), None)
    if period is None:
        for c in others:  # text columns that look like YYYY / YYYY-MM
            sample = df[c].dropna().astype(str).head(20)
            if len(sample) and sample.str.match(r"^\d{4}([-/]\d{1,2})?$").all():
                period = c
                break
    dims = [c for c in others if c != period]
    return {"numeric": numeric, "period": period, "dims": dims}


def auto_analysis(df: pd.DataFrame, question: str) -> AnalysisResult | None:
    """Choose and run an analysis tool, or return None when the result does not suit one."""
    if df is None or df.empty or len(df) < 3:
        return None
    sh = _shape(df)
    num, period, dims = sh["numeric"], sh["period"], sh["dims"]
    if not num:
        return None
    value = num[-1]
    if wants_decomposition(question) and period and dims:
        dim = dims[0]
        qty = next((c for c in num if re.search(r"(qty|quantity|units|volume|count)", str(c), re.I)), None)
        rev = next((c for c in num if re.search(r"(revenue|sales|amount|total)", str(c), re.I)), None)
        try:
            if qty and rev and qty != rev:
                res = pvm_decomposition(df, dim, period, qty, rev)
                t = res["totals"]
                top = res["table"].iloc[0]
                text = (f"Revenue changed by {t['delta']:,.2f} from {res['base']} to {res['current']}: "
                        f"price effect {t['price']:,.2f}, volume effect {t['volume']:,.2f}, mix effect {t['mix']:,.2f}, "
                        f"new/lost items {t['new_lost']:,.2f}. Largest single mover: {top[dim]} ({top['delta']:,.2f}).")
                fig = go.Figure(go.Bar(x=["price", "volume", "mix", "new/lost"],
                                       y=[t["price"], t["volume"], t["mix"], t["new_lost"]]))
                fig.update_layout(title="Price / volume / mix decomposition")
                return AnalysisResult(question, text, res["table"], fig, operation="pvm_decomposition",
                                      extras={"tool": "pvm_decomposition", "totals": t, "base": str(res["base"]), "current": str(res["current"])})
            res = contribution_analysis(df, period, dim, value)
            top = res["table"].head(3)
            parts = ", ".join(f"{r[dim]} ({r['delta']:+,.2f})" for _, r in top.iterrows())
            text = (f"{value} changed by {res['total_delta']:+,.2f} from {res['base']} to {res['current']}. "
                    f"Biggest contributors: {parts}.")
            return AnalysisResult(question, text, res["table"], _waterfall(res, dim, value),
                                  operation="contribution_analysis",
                                  extras={"tool": "contribution_analysis", "total_delta": res["total_delta"],
                                          "base": str(res["base"]), "current": str(res["current"])})
        except Exception:
            return None
    if period and not dims:
        series = df.sort_values(period).set_index(period)[value]
        if _FORECAST.search(question) and len(series) >= 4:
            try:
                fc = forecast_series(series.reset_index(drop=True))
                text = (f"{fc['method'].iloc[0]} forecast for {value}: next {len(fc)} steps "
                        + ", ".join(f"{v:,.2f}" for v in fc["forecast"]) + " (80% intervals in the table).")
                fig = go.Figure()
                fig.add_scatter(x=list(range(len(series))), y=series.values, name="actual")
                fig.add_scatter(x=list(range(len(series), len(series) + len(fc))), y=fc["forecast"], name="forecast")
                fig.add_scatter(x=list(range(len(series), len(series) + len(fc))), y=fc["upper"], name="upper", line=dict(dash="dot"))
                fig.add_scatter(x=list(range(len(series), len(series) + len(fc))), y=fc["lower"], name="lower", line=dict(dash="dot"))
                return AnalysisResult(question, text, fc, fig, operation="forecast", extras={"tool": "forecast"})
            except Exception:
                return None
        if _ANOMALY.search(question):
            an = detect_anomalies(series)
            where = ", ".join(str(i) for i in an["index"].tolist()[:6])
            text = (f"{len(an)} anomalous point(s) in {value} (robust z >= 3.5): {where}." if len(an) else f"No anomalies found in {value}.")
            return AnalysisResult(question, text, an, None, operation="anomalies", extras={"tool": "anomalies", "n": int(len(an))})
    return None
