import { useEffect, useState } from "react";
import type { RunRequest } from "../useRun";
import type { RunState } from "../state";

export type Mode = "scripted" | "live" | "recorded";

interface Props {
  state: RunState; datasets: any[]; config: any; golden: any[]; mode: Mode; setMode: (m: Mode) => void;
  onRun: (r: RunRequest) => void; onReplay: (id: string, q: string, speed: number) => void; onAnswer: (t: string) => void; onReset: () => void;
  refreshDatasets: () => void; dataset: string; setDataset: (d: string) => void;
}

const WRITERS: [string, string][] = [["writer-qlora", "QLoRA fine-tuned (default)"], ["base", "Base model"], ["writer-sft", "LoRA fine-tuned"], ["writer-dpo", "LoRA + preference tuning"]];
const SYSTEMS: [string, string][] = [["final", "Full system"], ["improved", "Better prompts, same agents"], ["original", "Original pipeline"], ["writer_only", "Writer alone, no agents"]];

export default function AskPanel(p: Props) {
  const ds = p.datasets.find((d) => d.id === p.dataset) || p.datasets[0];
  const [q, setQ] = useState("");
  const [ev, setEv] = useState("");
  const [writer, setWriter] = useState("");
  const [system, setSystem] = useState("final");
  const [speed, setSpeed] = useState(1);
  const [reply, setReply] = useState("");
  const [uploading, setUploading] = useState(false);
  const busy = p.state.status === "running" || p.state.status === "waiting";
  const adapters: string[] = p.config?.adapters || [];
  const writers = WRITERS.filter(([k]) => k === "base" || adapters.includes(k));
  useEffect(() => { if (!writer && writers.length) setWriter(writers[writers.length - 1][0]); }, [adapters.join()]);
  useEffect(() => { setReply(""); }, [p.state.pendingQuestion]);

  const run = (question = q, evidence = ev) => {
    if (!question.trim() || busy) return;
    p.onRun({ question, evidence, dataset: ds.id, mode: p.mode === "live" ? "live" : "scripted", system, writer: p.mode === "live" ? writer : null });
  };
  const upload = async (files: FileList | null) => {
    if (!files || !files.length) return;
    setUploading(true);
    try {
      const fd = new FormData();
      Array.from(files).forEach((f) => fd.append("files", f));
      fd.append("name", files[0].name.replace(/\.[^.]+$/, ""));
      const r = await fetch("/api/datasets/upload", { method: "POST", body: fd, headers: localStorage.getItem("vq_pass") ? { "x-passcode": localStorage.getItem("vq_pass")! } : {} });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail);
      await p.refreshDatasets(); p.setDataset(j.id);
    } catch (e: any) { alert("Upload failed: " + e.message); }
    setUploading(false);
  };

  return (
    <>
      <section className="sheet">
        <h2>Ask a question</h2>
        <div className="seg" role="group" aria-label="How to run">
          <button aria-pressed={p.mode === "scripted"} onClick={() => p.setMode("scripted")}>Scripted demo</button>
          <button aria-pressed={p.mode === "live"} onClick={() => p.setMode("live")}>Live models</button>
          <button aria-pressed={p.mode === "recorded"} onClick={() => p.setMode("recorded")} disabled={!p.golden.length} title={p.golden.length ? "" : "No recorded runs yet"}>Recorded</button>
        </div>
        {p.mode === "scripted" && <p className="muted small">No model calls: the answers are scripted for the sample questions. Everything else (schema lookup, SQL execution, checks, analysis) really runs.</p>}
        {p.mode === "live" && !p.config?.live_ready && <p className="notice">{p.config?.live_problem}</p>}

        {p.mode !== "recorded" ? <>
          <label className="field" htmlFor="ds">Data</label>
          <select id="ds" value={ds?.id} onChange={(e) => p.setDataset(e.target.value)}>{p.datasets.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}</select>
          <p className="muted small" style={{ margin: "4px 0 0" }}>{ds?.tables?.length ? ds.tables.map((t: any) => `${t.name} (${t.rows.toLocaleString()})`).join(", ") : ""}</p>
          <label className="field" htmlFor="up">Or bring your own CSV files</label>
          <input id="up" type="file" accept=".csv,.xlsx" multiple disabled={uploading} onChange={(e) => upload(e.target.files)} />
          <label className="field" htmlFor="q">Your question</label>
          <textarea id="q" value={q} placeholder="For example: What was our revenue last quarter?" onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run(); }} />
          <details style={{ marginTop: 6 }}><summary className="small muted" style={{ cursor: "pointer" }}>Add context for the analyst</summary>
            <textarea value={ev} onChange={(e) => setEv(e.target.value)} placeholder="For example: Today is 2025-07-01. Net revenue means after discounts." /></details>
          {p.mode === "live" && <>
            <label className="field" htmlFor="wr">SQL writer model</label>
            <select id="wr" value={writer} onChange={(e) => setWriter(e.target.value)}>{writers.map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
          </>}
          <label className="field" htmlFor="sy">Pipeline version</label>
          <select id="sy" value={system} onChange={(e) => setSystem(e.target.value)}>{SYSTEMS.filter(([k]) => (p.config?.systems || []).includes(k)).map(([k, l]) => <option key={k} value={k}>{l}</option>)}</select>
          <div className="row" style={{ marginTop: 14 }}>
            <button className="btn" disabled={busy || !q.trim() || (p.mode === "live" && !p.config?.live_ready)} onClick={() => run()}>{busy ? "Working" : "Ask"}</button>
            {p.state.status !== "idle" && <button className="btn ghost" onClick={p.onReset}>Clear</button>}
          </div>
        </> : <>
          <label className="field" htmlFor="sp">Playback speed</label>
          <select id="sp" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}><option value={1}>Real time</option><option value={2}>2 times faster</option><option value={4}>4 times faster</option></select>
          <ul className="samples">{p.golden.map((g) => (
            <li key={g.id}><button className="sample" disabled={busy || !g.ok} onClick={() => p.onReplay(g.id, g.question, speed)}>{g.question}<span className="hint">Recorded in {g.seconds}s</span></button></li>))}</ul>
        </>}
      </section>

      {p.state.pendingQuestion && (
        <section className="sheet ask-card" aria-live="assertive">
          <h3>The clarifier needs an answer</h3>
          <p>{p.state.pendingQuestion}</p>
          <input type="text" value={reply} autoFocus onChange={(e) => setReply(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && reply.trim()) p.onAnswer(reply); }} placeholder="Type your answer" aria-label="Your answer" />
          <div className="row" style={{ marginTop: 8 }}><button className="btn" disabled={!reply.trim()} onClick={() => p.onAnswer(reply)}>Send answer</button></div>
        </section>)}

      {p.mode !== "recorded" && ds?.samples?.length > 0 && (
        <section className="sheet">
          <h3>Try one of these</h3>
          <ul className="samples">{ds.samples.map((s: any) => (
            <li key={s.id}><button className="sample" disabled={busy} onClick={() => { setQ(s.question); setEv(s.evidence || ""); run(s.question, s.evidence || ""); }}>
              {s.question}{s.expect && <span className="hint">{s.expect.split(": ").slice(1).join(": ") || s.expect}</span>}</button></li>))}</ul>
        </section>)}
    </>
  );
}
