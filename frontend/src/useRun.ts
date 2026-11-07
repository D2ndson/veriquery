import { useCallback, useReducer, useRef } from "react";
import { api, stream } from "./api";
import { initial, reducer } from "./state";

export interface RunRequest { question: string; dataset: string; evidence: string; mode: "scripted" | "live"; system: string; writer?: string | null; latency?: number }

export function useRun() {
  const [state, dispatch] = useReducer(reducer, undefined, initial);
  const close = useRef<null | (() => void)>(null);

  const finish = useCallback(async (kind: "run" | "golden", id: string) => {
    try {
      const url = kind === "run" ? `/api/runs/${id}` : `/api/golden/${id}/result`;
      const r = await api(url);
      dispatch({ type: "result", result: kind === "run" ? r.result : r });
    } catch {}
    dispatch({ type: "end" });
  }, []);

  const start = useCallback(async (req: RunRequest) => {
    close.current?.();
    dispatch({ type: "start", question: req.question });
    try {
      const { run_id } = await api("/api/runs", { method: "POST", body: JSON.stringify({ ...req, interactive: true }) });
      dispatch({ type: "start", question: req.question, runId: run_id });
      close.current = stream(`/api/runs/${run_id}/events`, (ev) => ev.type !== "done" && dispatch({ type: "event", ev }), () => finish("run", run_id));
    } catch (e: any) { dispatch({ type: "fail", message: e.message }); }
  }, [finish]);

  const replay = useCallback((gid: string, question: string, speed: number) => {
    close.current?.();
    dispatch({ type: "start", question });
    close.current = stream(`/api/golden/${gid}/events?speed=${speed}`, (ev) => ev.type !== "done" && dispatch({ type: "event", ev }), () => finish("golden", gid));
  }, [finish]);

  const answer = useCallback(async (text: string) => {
    if (!state.runId) return;
    try { await api(`/api/runs/${state.runId}/answer`, { method: "POST", body: JSON.stringify({ text }) }); }
    catch (e: any) { dispatch({ type: "fail", message: e.message }); }
  }, [state.runId]);

  const reset = useCallback(() => { close.current?.(); dispatch({ type: "reset" }); }, []);
  return { state, start, replay, answer, reset };
}
