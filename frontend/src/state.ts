export type Ev = { seq: number; t: number; type: string; agent: string | null; data: any };
export type AgentStatus = { status: "idle" | "active" | "done"; note?: string; seconds?: number };
export type Row = { seq: number; t: number; from: string; to?: string; kind: string; summary: string; payload?: any };

export const AGENTS = ["user", "orchestrator", "schema_scout", "clarifier", "sql_writer", "verifier", "python_analyst", "reporter"] as const;
export const AGENT_LABEL: Record<string, string> = {
  user: "You", orchestrator: "Orchestrator", schema_scout: "Schema scout", clarifier: "Clarifier", sql_writer: "SQL writer",
  verifier: "Verifier", python_analyst: "Analyst", reporter: "Reporter",
};

export interface RunState {
  status: "idle" | "running" | "waiting" | "done" | "error";
  runId?: string;
  question: string;
  agents: Record<string, AgentStatus>;
  rows: Row[];
  edge: { from: string; to: string; summary: string; seq: number } | null;
  plan?: { plan?: string; intent?: string; difficulty?: number };
  pendingQuestion?: string;
  verifications: any[];
  repairs: any[];
  drafts?: any;
  firstResult?: any;       // table the SQL writer handed to the verifier first
  usage?: any;
  seconds?: number;
  error?: string;
  result?: any;
}

export const initial = (): RunState => ({
  status: "idle", question: "", rows: [], edge: null, verifications: [], repairs: [],
  agents: Object.fromEntries(AGENTS.map((a) => [a, { status: "idle" } as AgentStatus])),
});

export type Action = { type: "reset" } | { type: "start"; question: string; runId?: string } | { type: "event"; ev: Ev }
  | { type: "result"; result: any } | { type: "end" } | { type: "fail"; message: string };

export function reducer(s: RunState, a: Action): RunState {
  switch (a.type) {
    case "reset": return initial();
    case "start": return { ...initial(), status: "running", question: a.question, runId: a.runId,
      agents: { ...initial().agents, user: { status: "done" } } };
    case "result": return { ...s, result: a.result };
    case "fail": return { ...s, status: "error", error: a.message };
    case "end": return s.status === "running" || s.status === "waiting" ? { ...s, status: s.error ? "error" : "done" } : s;
    case "event": return onEvent(s, a.ev);
  }
  return s;
}

function onEvent(s: RunState, ev: Ev): RunState {
  const d = ev.data || {};
  const agents = { ...s.agents };
  let rows = s.rows, edge = s.edge, next: Partial<RunState> = {};
  const add = (r: Omit<Row, "seq" | "t">) => { rows = [...rows, { seq: ev.seq, t: ev.t, ...r }]; };
  switch (ev.type) {
    case "run_start": agents.orchestrator = { status: "active" }; add({ from: "user", to: "orchestrator", kind: "ask", summary: d.question });
      edge = { from: "user", to: "orchestrator", summary: "question", seq: ev.seq }; break;
    case "plan": next.plan = { ...(s.plan || {}), ...d };
      add({ from: "orchestrator", kind: "plan", summary: `Plan: ${d.plan}${d.intent ? `, intent ${d.intent}` : ""}${d.difficulty ? `, difficulty ${d.difficulty}/5` : ""}`, payload: d }); break;
    case "agent_start": if (ev.agent && agents[ev.agent]) agents[ev.agent] = { status: "active" }; break;
    case "agent_end": if (ev.agent && agents[ev.agent]) agents[ev.agent] = { status: "done", note: d.note, seconds: d.seconds }; break;
    case "message": {
      const to = d.to as string;
      add({ from: ev.agent || "?", to, kind: "message", summary: d.summary, payload: d });
      edge = { from: ev.agent || "", to, summary: d.summary, seq: ev.seq };
      if (ev.agent === "sql_writer" && to === "verifier" && !s.firstResult && d.result) next.firstResult = d.result;
      break;
    }
    case "clarify_question": next.pendingQuestion = d.question; next.status = "waiting";
      add({ from: "clarifier", to: "user", kind: "ask", summary: d.question });
      edge = { from: "clarifier", to: "user", summary: "needs your answer", seq: ev.seq }; break;
    case "clarify_answer": next.pendingQuestion = undefined; next.status = "running";
      add({ from: "user", to: "clarifier", kind: "answer", summary: d.answer });
      edge = { from: "user", to: "clarifier", summary: d.answer, seq: ev.seq }; break;
    case "sql_candidates": next.drafts = d;
      add({ from: "sql_writer", kind: "drafts", summary: `${(d.candidates || []).length} draft(s)${d.path ? `, ${d.path} path` : ""}${(d.fixes || []).length ? `, auto-fixed: ${d.fixes.join("; ")}` : ""}`, payload: d }); break;
    case "verification": next.verifications = [...s.verifications, d];
      add({ from: "verifier", kind: "verdict", summary: `Round ${d.round + 1}: ${d.decision.replace("_", " ")} (confidence ${Number(d.confidence).toFixed(2)})`, payload: d }); break;
    case "repair": next.repairs = [...s.repairs, d];
      add({ from: "sql_writer", kind: "repair", summary: `Repair ${d.round}: rewrote the query using the verifier's feedback`, payload: d });
      edge = { from: "sql_writer", to: "verifier", summary: "repaired query", seq: ev.seq }; break;
    case "analysis": add({ from: "python_analyst", kind: "analysis", summary: `${d.operation}: ${d.answer}`, payload: d }); break;
    case "run_end": agents.orchestrator = { ...agents.orchestrator, status: "done", seconds: d.seconds }; next.usage = d.usage; next.seconds = d.seconds; break;
    case "error": next.error = d.message; add({ from: "orchestrator", kind: "error", summary: d.message }); break;
  }
  return { ...s, ...next, agents, rows, edge };
}
