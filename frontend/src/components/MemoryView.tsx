import { useEffect, useState } from "react";
import { api } from "../api";

const stem = (id: string) => id.split(":").pop() || id;

export default function MemoryView({ datasets, dataset, setDataset }: { datasets: any[]; dataset: string; setDataset: (d: string) => void }) {
  const [items, setItems] = useState<any[]>([]);
  const [term, setTerm] = useState("");
  const [def, setDef] = useState("");
  const load = () => api(`/api/semantic/${stem(dataset)}`).then(setItems).catch(() => setItems([]));
  useEffect(() => { load(); }, [dataset]);
  const add = async () => { await api(`/api/semantic/${stem(dataset)}`, { method: "POST", body: JSON.stringify({ term, definition: def }) }); setTerm(""); setDef(""); load(); };
  const del = async (t: string) => { await api(`/api/semantic/${stem(dataset)}/${encodeURIComponent(t)}`, { method: "DELETE" }); load(); };
  return (
    <div className="page">
      <section className="sheet"><h2>Business definitions the system remembers</h2>
        <p className="muted">When the clarifier asks what a term means, such as "revenue" or "big order", your answer is saved here and applied to every later question, so you are only asked once. You can also add or remove definitions yourself.</p>
        <label className="field" htmlFor="mds">Data</label>
        <select id="mds" value={dataset} onChange={(e) => setDataset(e.target.value)}>{datasets.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}</select>
      </section>
      <section className="sheet">
        {items.length === 0 ? <p className="placeholder">Nothing remembered yet for this data. Ask "What was our revenue last quarter?" in the scripted demo, answer the clarifier, and the definition shows up here.</p> :
          <table className="runs"><thead><tr><th>Term</th><th>Definition</th><th>Learned from</th><th>Used</th><th /></tr></thead><tbody>
            {items.map((e) => <tr key={e.term}><td><b>{e.term}</b></td><td>{e.definition}</td><td>{e.source}</td><td className="num">{e.uses ?? 0}</td><td><button className="btn ghost" onClick={() => del(e.term)}>Forget</button></td></tr>)}</tbody></table>}
        <h3 style={{ marginTop: 18 }}>Add a definition</h3>
        <div className="row"><input type="text" style={{ maxWidth: 220 }} placeholder="Term, for example big order" value={term} onChange={(e) => setTerm(e.target.value)} aria-label="Term" />
          <input type="text" className="grow" placeholder="Definition, for example order value of at least $500" value={def} onChange={(e) => setDef(e.target.value)} aria-label="Definition" />
          <button className="btn" disabled={!term.trim() || !def.trim()} onClick={add}>Save definition</button></div>
      </section>
    </div>
  );
}
