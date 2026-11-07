import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError, setPasscode } from "./api";
import { useRun } from "./useRun";
import AskPanel, { type Mode } from "./components/AskPanel";
import AgentGraph from "./components/AgentGraph";
import Ledger from "./components/Ledger";
import AnswerSheet from "./components/AnswerSheet";
import ResultsView from "./components/ResultsView";
import MemoryView from "./components/MemoryView";

export default function App() {
  const { state, start, replay, answer, reset } = useRun();
  const [tab, setTab] = useState<"run" | "results" | "memory">("run");
  const [mode, setMode] = useState<Mode>("scripted");
  const [config, setConfig] = useState<any>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [golden, setGolden] = useState<any[]>([]);
  const [dataset, setDataset] = useState("shoptrap");
  const [locked, setLocked] = useState(false);
  const [code, setCode] = useState("");

  const load = useCallback(async () => {
    try {
      const [c, d, g] = await Promise.all([api("/api/config"), api("/api/datasets"), api("/api/golden")]);
      setConfig(c); setDatasets(d); setGolden(g); setLocked(false);
      if (c.live_ready && c.has_golden === false) setMode((m) => m);
    } catch (e: any) { if (e instanceof ApiError && e.status === 401) setLocked(true); }
  }, []);
  useEffect(() => { load(); }, [load]);
  const refreshDatasets = useCallback(async () => setDatasets(await api("/api/datasets")), []);

  const models = useMemo(() => {
    const m: Record<string, string> = {};
    if (mode === "scripted") return Object.fromEntries(["orchestrator", "clarifier", "sql_writer", "verifier", "reporter"].map((k) => [k, "scripted demo"]));
    (config?.roles || []).forEach((r: any) => {
      const k = ({ writer: "sql_writer", clarifier: "clarifier", checker: "verifier", reporter: "reporter", planner: "orchestrator" } as any)[r.role];
      if (k) m[k] = r.model.split("/").pop();
    });
    return m;
  }, [config, mode]);

  if (locked) return (
    <div className="page" style={{ maxWidth: 420, marginTop: 80 }}><section className="sheet"><h2>This demo needs a passcode</h2>
      <p className="muted">Live runs spend the owner's model credits, so the demo is protected.</p>
      <input type="password" value={code} onChange={(e) => setCode(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") { setPasscode(code); load(); } }} aria-label="Passcode" autoFocus />
      <div className="row" style={{ marginTop: 10 }}><button className="btn" onClick={() => { setPasscode(code); load(); }}>Open the demo</button></div></section></div>);

  const badge = mode === "scripted" ? { cls: "", text: "Scripted demo: no model calls. Not a benchmark result." }
    : mode === "live" ? { cls: "live", text: "Live: every agent calls a real model" } : { cls: "rec", text: "Recorded run: replay of a real run" };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="#f3f5ee" /><path d="M8 17l6 6 11-14" fill="none" stroke="#c23b2e" strokeWidth="3.6" strokeLinecap="round" strokeLinejoin="round" /></svg>
          VeriQuery
        </div>
        <nav className="tabs" role="tablist">
          {([["run", "Ask and watch"], ["results", "Results"], ["memory", "Memory"]] as const).map(([k, l]) =>
            <button key={k} className="tab" role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{l}</button>)}
        </nav>
        <span className={`badge ${badge.cls}`}>{badge.text}</span>
      </header>
      {tab === "run" && (
        <main className="workspace">
          <div className="col"><AskPanel state={state} datasets={datasets} config={config} golden={golden} mode={mode} setMode={setMode}
            onRun={start} onReplay={replay} onAnswer={answer} onReset={reset} refreshDatasets={refreshDatasets} dataset={dataset} setDataset={setDataset} /></div>
          <div className="col">
            <section className="sheet graph"><AgentGraph state={state} models={models} /></section>
            <Ledger state={state} />
          </div>
          <div className="col"><AnswerSheet state={state} /></div>
        </main>)}
      {tab === "results" && <ResultsView />}
      {tab === "memory" && <MemoryView datasets={datasets} dataset={dataset} setDataset={setDataset} />}
    </div>
  );
}
