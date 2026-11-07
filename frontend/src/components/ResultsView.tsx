import { useEffect, useState } from "react";
import { api } from "../api";

const LABEL: Record<string, string> = {
  minidev_base_once: "Base 7B model, writes SQL once", minidev_base_original: "Base 7B in the original pipeline", minidev_base_final: "Base 7B in the full system",
  minidev_sft_once: "LoRA fine-tuned writer, writes once", minidev_sft_final: "LoRA fine-tuned writer in the full system",
  minidev_dpo_once: "LoRA + preference tuning, writes once", minidev_dpo_final_norepair: "LoRA + preference tuning, full system, no repairer",
  minidev_dpo_final: "LoRA + preference tuning + repairer, full system", minidev_dpolf_final: "Label-free preference tuning, full system",
  shoptrap_base_original: "ShopTrap: base in the original pipeline", shoptrap_base_final: "ShopTrap: base in the full system",
  shoptrap_sft_final: "ShopTrap: fine-tuned, full system", shoptrap_dpo_final: "ShopTrap: preference tuned, full system",
};
const pct = (x: any) => (x == null ? "-" : `${(100 * x).toFixed(1)}%`);

export default function ResultsView() {
  const [r, setR] = useState<any>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api("/api/results").then(setR).catch((e) => setErr(e.message)); }, []);
  if (err) return <div className="page"><div className="error">{err}</div></div>;
  if (!r) return <div className="page"><p className="muted">Loading results…</p></div>;
  const runs = Object.entries<any>(r.runs).filter(([, m]) => m.accuracy != null);
  if (!runs.length) return (
    <div className="page"><section className="sheet"><h2>No results yet</h2>
      <p>Results appear here after the evaluation stages have run on the GPU machine. From the project folder:</p>
      <pre className="sql">python -m veriquery.pipeline.runner status{"\n"}python -m veriquery.pipeline.runner run</pre>
      <p className="muted">Until then, use the Scripted demo on the first tab. It is clearly labelled and never counts as a benchmark result.</p></section></div>);
  const mini = runs.filter(([t]) => t.startsWith("minidev")).sort((a, b) => Object.keys(LABEL).indexOf(a[0]) - Object.keys(LABEL).indexOf(b[0]));
  const best = Math.max(...mini.map(([, m]) => m.accuracy));
  return (
    <div className="page">
      <section className="sheet"><h2>Execution accuracy on BIRD Mini-Dev</h2>
        <p className="muted small">A query counts as right only when its result equals the reference result. Each bar is a separate run in the results folder.</p>
        <div className="bars">{mini.map(([t, m]) => (
          <div className="b" key={t}><span>{LABEL[t] || t}</span><div className="track"><div className={`fill ${m.accuracy === best ? "best" : ""}`} style={{ width: `${100 * m.accuracy}%` }} /></div><b>{pct(m.accuracy)}</b></div>))}</div>
      </section>
      <section className="sheet"><h2>All runs</h2>
        <div style={{ overflow: "auto" }}><table className="runs"><thead><tr><th>Run</th><th>Questions</th><th>Accuracy</th><th>First attempt</th><th>Confidently wrong</th><th>Wrong answers caught</th></tr></thead>
          <tbody>{runs.map(([t, m]) => <tr key={t}><td>{LABEL[t] || t}</td><td className="num">{m.n}</td><td className="num">{pct(m.accuracy)}</td><td className="num">{pct(m.first_attempt_accuracy)}</td><td className="num">{pct(m.confidently_wrong_rate)}</td><td className="num">{pct(m.catch_rate)}</td></tr>)}</tbody></table></div>
        <p className="muted small" style={{ marginTop: 8 }}>Confidently wrong means the answer was wrong and was not flagged. Lower is better.</p>
      </section>
      {r.summary?.paired && Object.keys(r.summary.paired).length > 0 && <section className="sheet"><h2>Paired differences</h2>
        <p className="muted small">Same questions in both runs, with a bootstrap 95% interval.</p>
        <table className="runs"><tbody>{Object.entries<any>(r.summary.paired).map(([k, d]) => (
          <tr key={k}><td>{k}</td><td className="num">{d.diff >= 0 ? "+" : ""}{(100 * d.diff).toFixed(1)} points</td><td className="num">[{(100 * d.ci95[0]).toFixed(1)}, {(100 * d.ci95[1]).toFixed(1)}]</td><td className="num">n={d.n}</td></tr>))}</tbody></table></section>}
      {r.figures?.length > 0 && <section className="sheet"><h2>Figures</h2><div className="figs">{r.figures.map((f: string) => <img key={f} src={`/results-files/${f}`} alt={f.replace(/^fig_|\.png$/g, "").replace(/_/g, " ")} />)}</div></section>}
      {[["Training data", r.build_stats], ["Mined trajectories", r.mine_stats], ["Preference data and label-free reward quality", r.pref_stats]].filter(([, v]) => v).map(([k, v]: any) => (
        <section className="sheet" key={k}><h2>{k}</h2><pre className="sql">{JSON.stringify(v, null, 1)}</pre></section>))}
    </div>
  );
}
