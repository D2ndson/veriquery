import { Suspense, lazy, useMemo, useState } from "react";
import { DataTable, DiffView, SqlView, Tick, fmt } from "./common";
import type { RunState } from "../state";

const Plot: any = lazy(async () => {
  const Plotly: any = (await import("plotly.js-basic-dist-min")).default;
  const factory: any = (await import("react-plotly.js/factory")).default;
  return { default: factory(Plotly) };
});

const CHECK_NAME: Record<string, string> = {
  dual_route: "Independent query", dual_route_units: "Units differ between the two queries", join_fanout: "Join fan-out",
  execution: "Query ran", null_order: "NULLs in the sort column", int_div: "Integer division", metamorphic_scale: "Scale check",
  empty_result: "Result is empty", literal_mismatch: "Filter value not found in the data", date_format: "Date format",
};
const human = (n: string) => CHECK_NAME[n] || n.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const DECISION: Record<string, string> = { accept: "Signed off", repair: "Sent back for repair", low_confidence: "Flagged as low confidence" };

export function VerificationLedger({ rounds }: { rounds: any[] }) {
  if (!rounds.length) return <p className="placeholder">The verifier has not looked at a query yet. Its checks are ticked off here, one by one.</p>;
  return (
    <div>
      {rounds.map((r, i) => (
        <div key={i}>
          <div className="round">Check {i + 1}<span className={`res ${r.decision}`}>{DECISION[r.decision] || r.decision}</span>
            <span className="muted small">confidence {Number(r.confidence).toFixed(2)}</span></div>
          <ul className="checks">
            {(r.checks || []).length === 0 && <li className="check"><Tick /><div><div className="nm">No problems found</div></div></li>}
            {(r.checks || []).map((c: any, k: number) => {
              const kind = c.severity === "fail" ? "cross" : c.severity === "warn" ? "flag" : "tick";
              return (
                <li key={k} className={`check ${c.severity}`}>
                  <Tick kind={kind as any} delay={0.12 * k} />
                  <div><div className="nm"><span>{human(c.name)}</span></div><div className="msg">{c.message}</div></div>
                </li>
              );
            })}
          </ul>
          {r.second_route?.result && <p className="small muted">The independent query returned {r.second_route.result.n_rows} row(s); first value {fmt(r.second_route.result.rows?.[0]?.[0])}.</p>}
        </div>
      ))}
    </div>
  );
}

function Decomposition({ extras }: { extras: any }) {
  const t = extras.totals || {};
  const keys = ["price", "volume", "mix", "new_lost"].filter((k) => k in t && Math.abs(t[k]) > 0.005);
  const max = Math.max(...keys.map((k) => Math.abs(t[k])), Math.abs(t.delta || 0), 1);
  const label: Record<string, string> = { price: "Price", volume: "Volume", mix: "Mix", new_lost: "New and lost" };
  return (
    <div>
      <h3>What drove the change from {extras.base} to {extras.current}</h3>
      <div className="decomp">
        {keys.map((k) => (<div key={k} style={{ display: "contents" }}>
          <span>{label[k]}</span>
          <div className="bar"><i style={{ left: t[k] >= 0 ? "50%" : `${50 - 50 * Math.abs(t[k]) / max}%`, width: `${50 * Math.abs(t[k]) / max}%`, background: t[k] >= 0 ? "#2f6b4f" : "#c23b2e" }} /></div>
          <span className="v">{t[k] >= 0 ? "+" : ""}{fmt(Math.round(t[k]))}</span></div>))}
        <b>Total</b><span /><b className="v">{t.delta >= 0 ? "+" : ""}{fmt(Math.round(t.delta))}</b>
      </div>
      <p className="muted small">The three parts add up exactly to the total change, so nothing is left unexplained.</p>
    </div>
  );
}

function Chart({ figure }: { figure: any }) {
  const layout = { ...figure.layout, template: undefined, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#ffffff", height: 290, autosize: true,
    margin: { l: 56, r: 12, t: 44, b: 52 }, font: { family: "Bricolage Grotesque Variable, system-ui, sans-serif", color: "#16213a", size: 12 } };
  return (
    <Suspense fallback={<p className="muted small">Loading chart…</p>}>
      <Plot data={figure.data} layout={layout} config={{ displayModeBar: false, responsive: true }} useResizeHandler style={{ width: "100%" }} />
    </Suspense>
  );
}

export default function AnswerSheet({ state }: { state: RunState }) {
  const res = state.result;
  const answers: any[] = res?.answers || [];
  const [idx, setIdx] = useState(0);
  const [tab, setTab] = useState("final");
  const a = answers[Math.min(idx, answers.length - 1)];
  const rep = a?.report || {};
  const repaired = state.repairs.length > 0;
  const last = state.repairs[state.repairs.length - 1];
  const firstVal = state.firstResult?.rows?.[0]?.[0];
  const finalVal = a?.table?.rows?.[0]?.[0];
  const firstFails = state.verifications[0]?.checks?.filter((c: any) => c.severity === "fail") || [];
  const secondSql = useMemo(() => [...state.verifications].reverse().find((v) => v.second_route?.code)?.second_route?.code, [state.verifications]);
  const fixes: string[] = state.drafts?.fixes || [];

  if (state.status === "idle") return <section className="sheet"><h2>Answer sheet</h2><p className="placeholder">Pick a question on the left. The answer appears here once the verifier has signed off, together with the evidence.</p></section>;

  return (
    <>
      <section className="sheet" aria-live="polite">
        {state.status === "error" && <div className="error"><b>The run stopped.</b> {state.error}</div>}
        {!a && state.status !== "error" && <p className="placeholder">Working on it. The agents are still talking, the answer appears when the verifier is done.</p>}
        {answers.length > 1 && <div className="subtabs" role="tablist">{answers.map((x, i) => <button key={i} role="tab" aria-selected={i === idx} onClick={() => setIdx(i)}>Reading {x.label || i + 1}</button>)}</div>}
        {a && <>
          {a.question && answers.length > 1 && <p className="muted small">{a.question}</p>}
          <div className="answer-big">{rep.answer}</div>
          <div className="row">
            <span className={`conf ${rep.confidence}`}>{rep.confidence === "high" ? "Verified" : rep.confidence === "medium" ? "Partly verified" : rep.confidence === "low" ? "Check this one" : "Not verified"}</span>
            {rep.p_correct != null && <span className="muted small">calibrated chance of being right: {Math.round(rep.p_correct * 100)}%</span>}
            {a.path && <span className="muted small">{a.path === "fast" ? "Answered on the fast path (one draft)" : a.path === "escalated" ? "Needed extra drafts and a vote" : "Full pipeline"}</span>}
          </div>
          {rep.abstain && <p className="notice">Below the calibrated confidence threshold. Treat this as unverified.</p>}
          {repaired && firstVal !== undefined && finalVal !== undefined && firstVal !== finalVal && (
            <div className="callout" style={{ marginTop: 12 }}>
              The first draft returned <b className="val">{fmt(firstVal)}</b>. The verifier flagged {firstFails.length ? firstFails.map((c: any) => human(c.name).toLowerCase()).join(" and ") : "a problem"}, and the repaired query returns <b className="val">{fmt(finalVal)}</b>.
            </div>)}
          {(rep.assumptions?.length > 0) && <><h3 style={{ marginTop: 14 }}>Assumptions</h3><ul className="plain small">{rep.assumptions.map((x: string, i: number) => <li key={i}>{x}</li>)}</ul></>}
          {(a.warnings?.length > 0) && <ul className="plain small">{a.warnings.map((x: string, i: number) => <li key={i}><b>Warning:</b> {x}</li>)}</ul>}
          {a.extras?.tool === "pvm_decomposition" && <div style={{ marginTop: 14 }}><Decomposition extras={a.extras} /></div>}
          {a.figure && <div style={{ marginTop: 12 }}><Chart figure={a.figure} /></div>}
          <h3 style={{ marginTop: 14 }}>Result</h3>
          <DataTable table={a.table} />
        </>}
      </section>

      <section className="sheet">
        <h2>Query</h2>
        <div className="subtabs" role="tablist">
          {[["final", "Final"], ...(repaired || fixes.length ? [["changes", "What changed"]] : []), ...(secondSql ? [["second", "Independent query"]] : []), ["drafts", "Drafts"]].map(([k, l]) =>
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{l}</button>)}
        </div>
        {tab === "final" && <SqlView sql={a?.sql || last?.new_sql || state.drafts?.candidates?.[0]?.sql || ""} />}
        {tab === "changes" && <>
          {fixes.length > 0 && <><p className="small muted">Fixed automatically before running (no model call):</p><ul className="plain small">{fixes.map((f, i) => <li key={i}>{f}</li>)}</ul></>}
          {repaired && <><p className="small muted">Repair after verifier feedback. Removed text is struck through, added text is green.</p><DiffView a={state.repairs[0].old_sql} b={last.new_sql} /></>}
        </>}
        {tab === "second" && <><p className="small muted">Written independently from the question alone. If it disagrees, the answer is not accepted.</p><SqlView sql={secondSql} /></>}
        {tab === "drafts" && ((state.drafts?.candidates || []).length ? state.drafts.candidates.map((c: any, i: number) =>
          <div key={i} style={{ marginBottom: 8 }}><p className="small muted">Draft {i + 1}: {c.group === 0 ? "same result as the winner" : "different result"}</p><SqlView sql={c.sql} /></div>)
          : <p className="placeholder">No drafts yet.</p>)}
        {tab === "drafts" && state.drafts?.plan && <><p className="small muted">The writer's plan before the query</p><pre className="sql">{state.drafts.plan}</pre></>}
      </section>

      <section className="sheet">
        <h2>Verification</h2>
        <VerificationLedger rounds={state.verifications} />
        {(state.seconds != null || res?.usage?.total) && <p className="muted small" style={{ marginTop: 12 }}>
          Finished in {state.seconds?.toFixed(1)}s{res?.usage?.total?.completion_tokens ? `, ${fmt((res.usage.total.prompt_tokens || 0) + (res.usage.total.completion_tokens || 0))} tokens` : ""}.</p>}
      </section>
    </>
  );
}
