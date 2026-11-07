import { AGENT_LABEL } from "../state";

export const Chip = ({ agent }: { agent?: string }) =>
  agent ? <span className={`chip ${agent}`}>{AGENT_LABEL[agent] || agent}</span> : null;

export function fmt(v: any): string {
  if (v === null || v === undefined) return "null";
  if (typeof v === "number") return Number.isInteger(v) ? v.toLocaleString("en-US") : v.toLocaleString("en-US", { maximumFractionDigits: 2 });
  return String(v);
}

/** Red-pencil tick (or cross) drawn stroke by stroke. */
export function Tick({ kind = "tick", delay = 0 }: { kind?: "tick" | "cross" | "flag"; delay?: number }) {
  const style = { animationDelay: `${delay}s` };
  return (
    <svg className={`tick ${kind}`} viewBox="0 0 28 28" aria-hidden="true">
      {kind === "cross" ? <><path d="M7 7l14 14" style={style} /><path d="M21 7L7 21" style={{ animationDelay: `${delay + 0.2}s` }} /></>
        : <path d="M5 15l6 6L23 6" style={style} />}
    </svg>
  );
}

const KW = new Set(("select from where group by order having join left right inner outer on as and or not in is null like between case when then else end " +
  "distinct limit union all with asc desc count sum avg min max cast over partition exists").split(" "));

export function SqlView({ sql }: { sql: string }) {
  if (!sql) return <pre className="sql placeholder">No SQL yet.</pre>;
  const parts = sql.split(/('(?:[^']|'')*'|--[^\n]*|\b\d+(?:\.\d+)?\b|\b[A-Za-z_]+\b)/g);
  return (
    <pre className="sql">
      {parts.map((p, i) => {
        if (!p) return null;
        if (p.startsWith("'")) return <span key={i} className="s">{p}</span>;
        if (p.startsWith("--")) return <span key={i} className="c">{p}</span>;
        if (/^\d/.test(p)) return <span key={i} className="n">{p}</span>;
        if (KW.has(p.toLowerCase())) return <span key={i} className="k">{p}</span>;
        return <span key={i}>{p}</span>;
      })}
    </pre>
  );
}

/** Word-level diff (LCS) between two SQL strings. */
export function DiffView({ a, b }: { a: string; b: string }) {
  const ta = a.split(/(\s+)/), tb = b.split(/(\s+)/);
  const n = ta.length, m = tb.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = ta[i] === tb[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out: { t: string; k: "" | "del" | "ins" }[] = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (ta[i] === tb[j]) { out.push({ t: ta[i], k: "" }); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) out.push({ t: ta[i++], k: "del" });
    else out.push({ t: tb[j++], k: "ins" });
  }
  while (i < n) out.push({ t: ta[i++], k: "del" });
  while (j < m) out.push({ t: tb[j++], k: "ins" });
  return <pre className="sql">{out.map((o, k) => <span key={k} className={o.k}>{o.t}</span>)}</pre>;
}

export function DataTable({ table }: { table: { columns: string[]; rows: any[][]; n_rows?: number } | null | undefined }) {
  if (!table || !table.columns?.length) return <p className="muted small">No rows.</p>;
  const num = table.columns.map((_, c) => table.rows.slice(0, 20).every((r) => r[c] === null || typeof r[c] === "number"));
  return (
    <>
      <div className="tablewrap" tabIndex={0}>
        <table className="data">
          <thead><tr>{table.columns.map((c, i) => <th key={i} className={num[i] ? "num" : ""}>{c}</th>)}</tr></thead>
          <tbody>{table.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className={num[j] ? "num" : ""}>{fmt(v)}</td>)}</tr>)}</tbody>
        </table>
      </div>
      {table.n_rows !== undefined && table.n_rows > table.rows.length &&
        <p className="muted small">Showing {table.rows.length} of {table.n_rows.toLocaleString()} rows.</p>}
    </>
  );
}
