import { memo, useMemo } from "react";
import ReactFlow, { Handle, Position, MarkerType, type Edge, type Node } from "reactflow";
import { AGENT_LABEL, type RunState } from "../state";

const POS: Record<string, [number, number]> = {
  user: [0, 0], orchestrator: [200, 0], schema_scout: [400, 0], clarifier: [600, 0],
  reporter: [0, 210], python_analyst: [200, 210], verifier: [400, 210], sql_writer: [600, 210],
};
const STATUS_TEXT: Record<string, string> = { idle: "waiting", active: "working", done: "done" };

const AgentNode = memo(({ data }: any) => (
  <div className={`node ${data.status} ${data.id === "user" ? "you" : ""}`}>
    {(["Left", "Right", "Top", "Bottom"] as const).map((side) => ["source", "target"].map((t) => (
      <Handle key={side + t} id={`${side[0].toLowerCase()}${t[0]}`} type={t as any} position={Position[side]} />)))}
    <div className="nm"><span className="dot" />{AGENT_LABEL[data.id]}</div>
    <div className="md" title={data.model}>{data.model || (data.id === "user" ? "the person asking" : "rules + tools")}</div>
    <div className="st">{data.seconds != null ? `done in ${data.seconds.toFixed(1)}s` : STATUS_TEXT[data.status]}</div>
  </div>
));
const nodeTypes = { agent: AgentNode };

// [from, to, sourceHandle, targetHandle]
const LINKS: [string, string, string, string][] = [
  ["user", "orchestrator", "rs", "lt"], ["orchestrator", "schema_scout", "rs", "lt"], ["schema_scout", "clarifier", "rs", "lt"],
  ["clarifier", "sql_writer", "bs", "tt"], ["sql_writer", "verifier", "ls", "rt"], ["verifier", "sql_writer", "bs", "bt"],
  ["verifier", "python_analyst", "ls", "rt"], ["python_analyst", "reporter", "ls", "rt"], ["reporter", "user", "ts", "bt"],
  ["clarifier", "user", "ts", "tt"],
];

export default function AgentGraph({ state, models }: { state: RunState; models: Record<string, string> }) {
  const nodes: Node[] = useMemo(() => Object.entries(POS).map(([id, [x, y]]) => ({
    id, type: "agent", position: { x, y }, draggable: false, selectable: false,
    data: { id, model: models[id], ...state.agents[id] },
  })), [state.agents, models]);
  const edges: Edge[] = useMemo(() => LINKS.map(([s, t, sh, th]) => {
    const hot = state.edge && state.edge.from === s && state.edge.to === t;
    return {
      id: `${s}>${t}${sh}`, source: s, target: t, sourceHandle: sh, targetHandle: th, type: s === "clarifier" && t === "user" ? "default" : "smoothstep",
      animated: !!hot, label: hot ? state.edge!.summary.slice(0, 44) : undefined,
      labelBgStyle: { fill: "#f2d45c" }, labelStyle: { fontSize: 11.5, fontWeight: 600, fill: "#16213a" }, labelBgPadding: [6, 3] as [number, number],
      style: { stroke: hot ? "#c23b2e" : "#9aa3b2", strokeWidth: hot ? 3 : 1.6 },
      markerEnd: { type: MarkerType.ArrowClosed, color: hot ? "#c23b2e" : "#9aa3b2", width: 16, height: 16 },
    } as Edge;
  }), [state.edge]);
  return (
    <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.12 }} nodesConnectable={false}
      zoomOnScroll={false} panOnDrag={false} zoomOnDoubleClick={false} preventScrolling={false} minZoom={0.3} aria-label="Agents and the messages passed between them" />
  );
}
