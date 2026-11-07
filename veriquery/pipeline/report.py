"""Build results/REPORT.md, results/summary.json and figures from everything under results/ and logs/.

Comparisons are PAIRED: two systems are compared only on the questions both answered, with a bootstrap 95% CI for the
difference. Quick-run outputs (artifacts under quick_run/) never mix with full runs.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from veriquery.config import path

ORDER = [("minidev_base_once", "Base 7B, write SQL once (no agents)"),
         ("minidev_base_original", "Base 7B + original pipeline (as delivered)"),
         ("minidev_base_final", "Base 7B + full system"),
         ("minidev_sft_once", "LoRA-SFT writer, write once"),
         ("minidev_sft_final", "LoRA-SFT writer + full system"),
         ("minidev_dpo_once", "SFT+DPO writer, write once"),
         ("minidev_dpo_final_norepair", "SFT+DPO writer + full system (no repairer adapter)"),
         ("minidev_dpo_final", "SFT+DPO writer + repairer adapter + full system"),
         ("minidev_dpolf_final", "Label-free-reward DPO writer + full system")]
SHOP = [("shoptrap_base_original", "Base 7B + original pipeline"), ("shoptrap_base_final", "Base 7B + full system"),
        ("shoptrap_sft_final", "SFT writer + full system"), ("shoptrap_dpo_final", "SFT+DPO writer + full system")]


def _load(tag: str) -> dict[str, dict]:
    f = path(f"results/{tag}/predictions.jsonl")
    if not f.is_file():
        return {}
    return {str(r["id"]): r for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip())
            if r.get("final_correct") is not None}


def _metrics(tag: str) -> dict | None:
    f = path(f"results/{tag}/metrics.json")
    return json.loads(f.read_text()) if f.is_file() else None


def paired_diff(a: dict, b: dict, n_boot: int = 2000, seed: int = 0) -> dict | None:
    ids = sorted(set(a) & set(b))
    if len(ids) < 10:
        return None
    d = [int(bool(b[i]["final_correct"])) - int(bool(a[i]["final_correct"])) for i in ids]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"n": len(ids), "diff": sum(d) / len(d), "ci95": [boots[int(.025 * n_boot)], boots[int(.975 * n_boot)]]}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _table(rows: list[tuple[str, str]]) -> list[str]:
    out = ["| System | n | Accuracy | 1st-attempt | Confidently wrong | Catch rate | Tokens (M) | s/question |",
           "|---|---|---|---|---|---|---|---|"]
    for tag, label in rows:
        m = _metrics(tag)
        if not m:
            continue
        u = m.get("usage") or {}
        tok = (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)) / 1e6
        out.append(f"| {label} | {m.get('n')} | {_pct(m.get('accuracy'))} | {_pct(m.get('first_attempt_accuracy'))} | "
                   f"{_pct(m.get('confidently_wrong_rate'))} | {_pct(m.get('catch_rate'))} | {tok:.2f} | "
                   f"{m.get('mean_seconds') and round(m['mean_seconds'], 1) or '-'} |")
    return out if len(out) > 2 else []


def _figures(out: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    made = []
    rows = [(l, _metrics(t)) for t, l in ORDER if _metrics(t)]
    if rows:
        fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.5))
        ax.barh([r[0] for r in rows][::-1], [100 * (r[1].get("accuracy") or 0) for r in rows][::-1], color="#3b82f6")
        ax.set_xlabel("Execution accuracy on BIRD Mini-Dev (%)")
        fig.tight_layout(); fig.savefig(out / "fig_accuracy.png", dpi=160); plt.close(fig); made.append("fig_accuracy.png")
    logs = sorted(path("logs").glob("train_*.jsonl"))
    if logs:
        fig, ax = plt.subplots(figsize=(7, 4))
        for f in logs:
            pts = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            if pts:
                ax.plot([p["step"] for p in pts], [p["loss"] for p in pts], label=f.stem.replace("train_", ""))
        ax.set_xlabel("step"); ax.set_ylabel("training loss"); ax.set_yscale("log"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "fig_training_loss.png", dpi=160); plt.close(fig); made.append("fig_training_loss.png")
    for tag in ("minidev_dpo_final", "minidev_sft_final", "minidev_base_final"):
        m = _metrics(tag)
        if m and m.get("calibration"):
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.plot([c["mean_confidence"] for c in m["calibration"]], [c["accuracy"] for c in m["calibration"]], "o-")
            ax.set_xlabel("stated confidence"); ax.set_ylabel("actual accuracy"); ax.set_title(f"Calibration: {tag}")
            fig.tight_layout(); fig.savefig(out / "fig_calibration.png", dpi=160); plt.close(fig); made.append("fig_calibration.png")
            break
    return made


def build_report() -> Path:
    out = path("results")
    out.mkdir(exist_ok=True)
    lines = ["# VeriQuery results", "", "_Generated automatically from `results/*/metrics.json`. Execution accuracy follows BIRD "
             "(result-set equality). All figures come from the runs in this folder; nothing here is copied from a paper._", ""]
    t = _table(ORDER)
    lines += ["## BIRD Mini-Dev", ""] + (t or ["(no BIRD runs yet)"]) + [""]
    lines += ["## ShopTrap (our trap set: units, fan-out, date formats, ambiguity)", ""] + (_table(SHOP) or ["(no ShopTrap runs yet)"]) + [""]
    cmp_rows, summary = [], {}
    for a, b, label in [("minidev_base_once", "minidev_sft_once", "LoRA-SFT vs base (writer only)"),
                        ("minidev_sft_once", "minidev_dpo_once", "DPO vs SFT (writer only)"),
                        ("minidev_base_final", "minidev_sft_final", "LoRA-SFT vs base (full system)"),
                        ("minidev_sft_final", "minidev_dpo_final", "DPO + repairer vs SFT (full system)"),
                        ("minidev_dpo_final_norepair", "minidev_dpo_final", "repairer adapter on vs off"),
                        ("minidev_base_original", "minidev_dpo_final", "original pipeline -> final system")]:
        d = paired_diff(_load(a), _load(b))
        if d:
            summary[label] = d
            cmp_rows.append(f"| {label} | {d['n']} | {100 * d['diff']:+.1f} pts | [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] |")
    if cmp_rows:
        lines += ["## Paired differences (same questions, bootstrap 95% CI)", "", "| Comparison | n | Δ accuracy | 95% CI |", "|---|---|---|---|"] + cmp_rows + [""]
    for name, f in (("Data build", "artifacts/sft/build_stats.json"), ("Mining", "artifacts/trajectories/mine_stats.json"),
                    ("Preference data / label-free reward quality", "artifacts/sft/pref_stats.json")):
        if path(f).is_file():
            lines += [f"## {name}", "", "```json", path(f).read_text().strip(), "```", ""]
    cal = path("artifacts/calibrator.json")
    if cal.is_file():
        meta = json.loads(cal.read_text()).get("meta", {})
        lines += ["## Calibrator (cross-validated)", "", "```json", json.dumps(meta, indent=1), "```", ""]
    figs = _figures(out)
    lines += [f"![{f}]({f})" for f in figs]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"paired": summary,
                                                  "runs": {t: _metrics(t) for t, _ in ORDER + SHOP if _metrics(t)}},
                                                 indent=1, default=str))
    print(f"[report] wrote {out / 'REPORT.md'}")
    return out / "REPORT.md"


if __name__ == "__main__":
    build_report()
