import { useEffect, useRef, useState } from "react";
import { Chip, DataTable, SqlView } from "./common";
import type { Row, RunState } from "../state";

function Detail({ row }: { row: Row }) {
  const p = row.payload || {};
  return (
    <div onClick={(e) => e.stopPropagation()}>
      {p.sql && <SqlView sql={p.sql} />}
      {p.result && <DataTable table={p.result} />}
      {p.feedback && <pre className="sql">{p.feedback}</pre>}
      {p.old_sql && <><p className="small muted">Before</p><SqlView sql={p.old_sql} /><p className="small muted">After</p><SqlView sql={p.new_sql} /></>}
      {row.kind === "drafts" && (p.candidates || []).map((c: any, i: number) => <div key={i}><p className="small muted">Draft {i + 1}{c.group === 0 ? " (agrees with the winner)" : " (different result)"}</p><SqlView sql={c.sql} /></div>)}
      {row.kind === "drafts" && p.plan && <><p className="small muted">The writer's plan</p><pre className="sql">{p.plan}</pre></>}
      {row.kind === "verdict" && (p.checks || []).map((c: any, i: number) => <p key={i} className="small"><b>{c.name}</b>: {c.message}</p>)}
      {p.tables && <p className="small muted">Tables: {p.tables.join(", ")}</p>}
      {p.definitions && p.definitions.map((d: any, i: number) => <p key={i} className="small"><b>{d.term}</b>: {d.definition}</p>)}
      {p.readings && p.readings.map((r: any, i: number) => <p key={i} className="small"><b>Reading {i + 1}:</b> {r.question}</p>)}
    </div>
  );
}

export default function Ledger({ state }: { state: RunState }) {
  const [open, setOpen] = useState<number | null>(null);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ block: "end" }); }, [state.rows.length]);
  return (
    <section className="sheet ledger-wrap" aria-label="Conversation between agents">
      <div className="ledger-head"><h2>What the agents are saying</h2><span className="muted small">{state.rows.length} message{state.rows.length === 1 ? "" : "s"}. Click one to see the data behind it.</span></div>
      <div className="ledger" role="log" aria-live="polite">
        {state.rows.length === 0 && <div className="empty">Ask a question and each hand-off between agents appears here as it happens.</div>}
        {state.rows.map((r, i) => (
          <div key={i} className="lrow" onClick={() => setOpen(open === i ? null : i)}>
            <span className="tm">{r.t.toFixed(1)}s</span>
            <div>
              <div className="who"><Chip agent={r.from} />{r.to && <><span>to</span><Chip agent={r.to} /></>}</div>
              <div className="sum">{r.summary}</div>
              {open === i && r.payload && <Detail row={r} />}
            </div>
          </div>
        ))}
        <div ref={end} />
      </div>
    </section>
  );
}
